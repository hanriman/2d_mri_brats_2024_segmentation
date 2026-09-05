import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEFAULT_NUM_WORKERS,
    LOGS_DIR,
    METRICS_DIR,
    ensure_directories,
    get_metadata_path,
)
from brats_jepa.data import BraTS2DDataset, RandomModalityDropout
from brats_jepa.losses import CombinedDiceBCELoss, DeepSupervisionLoss
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import IJEPA, BraTS2DnnUNet, BraTS2DUNet, JEPASegmentationModel, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Low-Data Label Efficiency Benchmark Runner")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv")
    parser.add_argument("--epochs", type=int, default=30, help="Downstream fine-tuning epochs per label fraction")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--p_drop", type=float, default=0.25, help="Modality dropout probability during training")
    parser.add_argument("--exp_version", type=str, default="v2_low_data_efficiency", help="Experiment version directory tag")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to search for pre-trained checkpoints")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Number of DataLoader worker processes (default: 2 on Linux, 0 on macOS)")
    parser.add_argument("--cache_data", action="store_true", default=True,
                        help="Cache loaded slices in RAM to eliminate disk I/O bottlenecks")
    parser.add_argument("--no_cache_data", action="store_false", dest="cache_data",
                        help="Disable RAM caching of slices")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP (mixed precision)")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Enable cuDNN benchmark for static-sized convolutions on CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Versioned experiment directories
    base_out = Path(args.output_dir) if args.output_dir else Path("outputs")
    exp_dir = base_out / "experiments" / args.exp_version
    ckpt_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"
    for d in [exp_dir, ckpt_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    logger = get_logger("evaluate_low_data", logs_dir / "evaluate_low_data.log")
    logger.info(f"Starting Low-Data Label Efficiency Benchmark ({args.exp_version}) on device: {device}")
    logger.info(f"DataLoader settings: num_workers={args.num_workers}, cache_in_memory={args.cache_data}")
    
    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).resolve()
    else:
        metadata_path = get_metadata_path("brats_gli_2d")
        
    logger.info(f"Using dataset metadata: {metadata_path}")
    full_train_ds = BraTS2DDataset(metadata_csv=metadata_path, split="train", cache_in_memory=args.cache_data)
    test_ds = BraTS2DDataset(metadata_csv=metadata_path, split="test", cache_in_memory=args.cache_data)

    use_cuda = (device.type == "cuda")
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    
    total_train_samples = len(full_train_ds)
    label_fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    mod_drop = RandomModalityDropout(p_drop=args.p_drop)
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        logger.info("CUDA Mixed Precision (AMP) enabled for low-data benchmark.")
    
    ssl_base_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CHECKPOINTS_DIR
    
    results = []
    
    for frac in label_fractions:
        frac_tag = f"{int(round(frac*100))}pct"
        n_samples = max(1, int(round(frac * total_train_samples)))
        
        # Seeded stratified sampling across training dataset to preserve positive tumor ratio
        rng = np.random.RandomState(args.seed)
        tumor_indices = [i for i, r in enumerate(full_train_ds.records) if r.get("has_tumor", True) in (True, "True", 1, "1")]
        non_tumor_indices = [i for i in range(total_train_samples) if i not in set(tumor_indices)]
        
        if len(tumor_indices) > 0 and len(non_tumor_indices) > 0:
            tumor_frac = len(tumor_indices) / total_train_samples
            n_tumor = max(1, int(round(n_samples * tumor_frac))) if n_samples > 1 else 1
            n_non_tumor = max(0, n_samples - n_tumor)
            
            chosen_tumor = rng.choice(tumor_indices, size=min(n_tumor, len(tumor_indices)), replace=False).tolist()
            chosen_non_tumor = rng.choice(non_tumor_indices, size=min(n_non_tumor, len(non_tumor_indices)), replace=False).tolist() if n_non_tumor > 0 else []
            indices = chosen_tumor + chosen_non_tumor
            rng.shuffle(indices)
        else:
            indices = rng.choice(total_train_samples, size=n_samples, replace=False).tolist()
        
        sub_train_ds = Subset(full_train_ds, indices)
        train_loader = DataLoader(
            sub_train_ds,
            batch_size=min(args.batch_size, n_samples),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        
        logger.info(f"\n" + "="*80)
        logger.info(f"EVALUATING LABEL FRACTION: {frac*100:.0f}% ({n_samples}/{total_train_samples} training slices)")
        logger.info("="*80)
        
        # A. UNet Baseline
        logger.info(f"Training UNet Baseline on {frac*100:.0f}% labels...")
        t_start = time.perf_counter()
        unet = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
        loss_fn_bce = CombinedDiceBCELoss()
        opt_u = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, T_max=args.epochs)
        scaler_u = torch.amp.GradScaler('cuda', enabled=use_amp)
        for _ in range(args.epochs):
            unet.train()
            for b in train_loader:
                opt_u.zero_grad()
                imgs = mod_drop(b["image"].to(device, non_blocking=True))
                targets = b["label"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds = unet(imgs)
                    l = loss_fn_bce(preds, targets)
                if use_amp:
                    scaler_u.scale(l).backward()
                    scaler_u.unscale_(opt_u)
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    scaler_u.step(opt_u)
                    scaler_u.update()
                else:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                    opt_u.step()
            sched_u.step()
        u_time = time.perf_counter() - t_start
        
        unet.eval()
        d_list, i_list, h_list = [], [], []
        with torch.no_grad():
            for b in test_loader:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds = unet(b["image"].to(device, non_blocking=True))
                m = compute_segmentation_metrics(preds, b["label"].to(device, non_blocking=True))
                d_list.extend(m["dice_per_sample"])
                i_list.extend(m["iou_per_sample"])
                h_list.extend(m["hd95_per_sample"])
        
        unet_ckpt = ckpt_dir / f"unet_{frac_tag}.pt"
        torch.save({"model_state_dict": unet.state_dict(), "test_dice": float(np.mean(d_list))}, unet_ckpt)
        logger.info(f"Saved unet_{frac_tag}.pt | Training Time: {u_time:.2f}s ({u_time/args.epochs:.2f}s/epoch)")
        
        results.append({
            "label_fraction": f"{frac*100:.0f}%",
            "n_samples": n_samples,
            "model": "UNet Baseline",
            "test_dice": float(np.mean(d_list)),
            "test_iou": float(np.mean(i_list)),
            "hd95_px": float(np.mean(h_list)),
            "train_time_sec": round(u_time, 2),
            "sec_per_epoch": round(u_time / args.epochs, 2),
        })
        
        # B. nnU-Net SOTA Baseline
        logger.info(f"Training nnU-Net SOTA Baseline on {frac*100:.0f}% labels...")
        t_start = time.perf_counter()
        nnunet = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
        loss_fn_ds = DeepSupervisionLoss()
        opt_nn = torch.optim.AdamW(nnunet.parameters(), lr=2e-4, weight_decay=1e-5)
        sched_nn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_nn, T_max=args.epochs)
        scaler_nn = torch.amp.GradScaler('cuda', enabled=use_amp)
        for _ in range(args.epochs):
            nnunet.train()
            for b in train_loader:
                opt_nn.zero_grad()
                imgs = mod_drop(b["image"].to(device, non_blocking=True))
                targets = b["label"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds = nnunet(imgs)
                    l = loss_fn_ds(preds, targets)
                if use_amp:
                    scaler_nn.scale(l).backward()
                    scaler_nn.unscale_(opt_nn)
                    torch.nn.utils.clip_grad_norm_(nnunet.parameters(), max_norm=1.0)
                    scaler_nn.step(opt_nn)
                    scaler_nn.update()
                else:
                    l.backward()
                    torch.nn.utils.clip_grad_norm_(nnunet.parameters(), max_norm=1.0)
                    opt_nn.step()
            sched_nn.step()
        nn_time = time.perf_counter() - t_start
        
        nnunet.eval()
        d_list, i_list, h_list = [], [], []
        with torch.no_grad():
            for b in test_loader:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds = nnunet(b["image"].to(device, non_blocking=True))
                m = compute_segmentation_metrics(preds, b["label"].to(device, non_blocking=True))
                d_list.extend(m["dice_per_sample"])
                i_list.extend(m["iou_per_sample"])
                h_list.extend(m["hd95_per_sample"])
                
        nnunet_ckpt = ckpt_dir / f"nnunet_{frac_tag}.pt"
        torch.save({"model_state_dict": nnunet.state_dict(), "test_dice": float(np.mean(d_list))}, nnunet_ckpt)
        logger.info(f"Saved nnunet_{frac_tag}.pt | Training Time: {nn_time:.2f}s ({nn_time/args.epochs:.2f}s/epoch)")
        
        results.append({
            "label_fraction": f"{frac*100:.0f}%",
            "n_samples": n_samples,
            "model": "nnU-Net (Supervised SOTA)",
            "test_dice": float(np.mean(d_list)),
            "test_iou": float(np.mean(i_list)),
            "hd95_px": float(np.mean(h_list)),
            "train_time_sec": round(nn_time, 2),
            "sec_per_epoch": round(nn_time / args.epochs, 2),
        })
        
        # C. Pre-trained JEPA Variants (Fine-tuned)
        jepa_variants = ["ijepa", "sigreg_jepa", "visreg_jepa"]
        for type_name in jepa_variants:
            logger.info(f"Fine-tuning Pre-trained {type_name.upper()} on {frac*100:.0f}% labels...")
            t_start = time.perf_counter()
            model = JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1).to(device)
            
            # Check ssl checkpoint locations
            candidates = [
                ssl_base_dir / f"best_{type_name}.pt",
                ssl_base_dir / f"{type_name}_100pct.pt",
                CHECKPOINTS_DIR / f"best_{type_name}.pt",
                CHECKPOINTS_DIR / f"{type_name}_100pct.pt",
            ]
            ssl_ckpt = next((c for c in candidates if c.exists()), None)
            if ssl_ckpt is not None:
                logger.info(f"Loaded SSL encoder weights from: {ssl_ckpt}")
                ckpt = torch.load(ssl_ckpt, map_location=device)
                model.load_pretrained_encoder(ckpt["context_encoder_state_dict"])
            else:
                logger.warning(f"No SSL pre-trained weights found for {type_name} in {ssl_base_dir}. Training from scratch!")
                
            opt_j = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched_j = torch.optim.lr_scheduler.CosineAnnealingLR(opt_j, T_max=args.epochs)
            scaler_j = torch.amp.GradScaler('cuda', enabled=use_amp)
            for _ in range(args.epochs):
                model.train()
                for b in train_loader:
                    opt_j.zero_grad()
                    imgs = mod_drop(b["image"].to(device, non_blocking=True))
                    targets = b["label"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        preds = model(imgs)
                        l = loss_fn_bce(preds, targets)
                    if use_amp:
                        scaler_j.scale(l).backward()
                        scaler_j.unscale_(opt_j)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler_j.step(opt_j)
                        scaler_j.update()
                    else:
                        l.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        opt_j.step()
                sched_j.step()
            j_time = time.perf_counter() - t_start
            
            model.eval()
            d_list, i_list, h_list = [], [], []
            with torch.no_grad():
                for b in test_loader:
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        preds = model(b["image"].to(device, non_blocking=True))
                    m = compute_segmentation_metrics(preds, b["label"].to(device, non_blocking=True))
                    d_list.extend(m["dice_per_sample"])
                    i_list.extend(m["iou_per_sample"])
                    h_list.extend(m["hd95_per_sample"])
            
            jepa_ckpt = ckpt_dir / f"finetuned_{type_name}_{frac_tag}.pt"
            torch.save({"model_state_dict": model.state_dict(), "test_dice": float(np.mean(d_list))}, jepa_ckpt)
            logger.info(f"Saved finetuned_{type_name}_{frac_tag}.pt | Training Time: {j_time:.2f}s ({j_time/args.epochs:.2f}s/epoch)")
            
            clean_name = "I-JEPA (Fine-tuned)" if type_name == "ijepa" else ("SigReg JEPA (Fine-tuned)" if type_name == "sigreg_jepa" else "VisReg JEPA (Fine-tuned)")
            results.append({
                "label_fraction": f"{frac*100:.0f}%",
                "n_samples": n_samples,
                "model": clean_name,
                "test_dice": float(np.mean(d_list)),
                "test_iou": float(np.mean(i_list)),
                "hd95_px": float(np.mean(h_list)),
                "train_time_sec": round(j_time, 2),
                "sec_per_epoch": round(j_time / args.epochs, 2),
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*105)
    print("         LOW-DATA LABEL EFFICIENCY BENCHMARK SUMMARY (With Execution Timings)")
    print("="*105)
    print(summary_df.to_string(index=False))
    print("="*105 + "\n")
    
    out_csv = metrics_dir / "low_data_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved low-data benchmark summary to: {out_csv}")

if __name__ == "__main__":
    main()
