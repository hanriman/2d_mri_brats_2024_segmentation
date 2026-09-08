import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_WORKERS,
    get_metadata_path,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa.data import BraTS2DDataset, RandomModalityDropout
from brats_jepa.losses import CombinedDiceBCELoss, DeepSupervisionLoss
from brats_jepa.metrics import (
    compute_patient_volume_metrics,
    compute_segmentation_metrics,
)
from brats_jepa.models import (
    BraTS2DnnUNet,
    BraTS2DUNet,
    JEPASegmentationModel,
)
from brats_jepa.utils import get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Low-Data Label Efficiency Benchmark Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
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
    parser.add_argument("--decoder_type", type=str, choices=["bottleneck", "multiscale"], default="bottleneck",
                        help="Downstream decoder architecture: standard bottleneck or hierarchical multiscale feature pyramid")
    parser.add_argument("--encoder_source", type=str, choices=["target", "context"], default="target",
                        help="Source encoder weights to load: target (EMA teacher) or context (online student)")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Limit batches per epoch/eval for rapid smoke testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluates a model computing slice-wise, tumor-stratified, and 3D volumetric metrics."""
    model.eval()
    dice_list, iou_list, hd95_list = [], [], []
    dice_tumor_list, iou_tumor_list, hd95_tumor_list = [], [], []
    all_logits, all_labels = [], []
    all_patients, all_slice_idxs = [], []

    with torch.no_grad():
        for batch_idx, b in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            imgs = b["image"].to(device, non_blocking=True)
            targets = b["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(imgs)

            m = compute_segmentation_metrics(preds, targets)
            dice_list.extend(m["dice_per_sample"])
            iou_list.extend(m["iou_per_sample"])
            hd95_list.extend(m["hd95_per_sample"])

            if "has_tumor_per_sample" in m:
                for d, iou, h, has_t in zip(
                    m["dice_per_sample"],
                    m["iou_per_sample"],
                    m["hd95_per_sample"],
                    m["has_tumor_per_sample"],
                ):
                    if has_t:
                        dice_tumor_list.append(d)
                        iou_tumor_list.append(iou)
                        hd95_tumor_list.append(h)

            all_logits.append(preds.detach().cpu())
            all_labels.append(targets.detach().cpu())
            all_patients.extend(b["patient_id"])
            all_slice_idxs.extend(
                b["slice_index"].tolist() if torch.is_tensor(b["slice_index"]) else b["slice_index"]
            )

    vol_metrics = compute_patient_volume_metrics(
        slice_preds=all_logits,
        slice_targets=all_labels,
        patient_ids=all_patients,
        slice_indices=all_slice_idxs,
    )

    return {
        "test_dice": float(np.mean(dice_list)) if dice_list else 0.0,
        "test_dice_tumor": float(np.mean(dice_tumor_list)) if dice_tumor_list else float("nan"),
        "test_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "hd95_px": float(np.mean(hd95_list)) if hd95_list else 0.0,
        "patient_3d_dice": vol_metrics.get("dice_3d", vol_metrics.get("patient_3d_dice_mean", float("nan"))),
        "patient_3d_hd95_px": vol_metrics.get("hd95_3d", vol_metrics.get("patient_3d_hd95_mean", float("nan"))),
    }


def main():
    args = parse_args()
    if args.config:
        cfg = load_yaml_config(args.config)
        args = merge_config_with_args(cfg, args)

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
    logger.info(f"Decoder type: {args.decoder_type}, Encoder source: {args.encoder_source}")

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
        frac_tag = f"{round(frac * 100)}pct"
        n_samples = max(1, round(frac * total_train_samples))

        # Seeded stratified sampling across training dataset to preserve positive tumor ratio
        rng = np.random.RandomState(args.seed)
        tumor_indices = [i for i, r in enumerate(full_train_ds.records) if r.get("has_tumor", True) in (True, "True", 1, "1")]
        non_tumor_indices = [i for i in range(total_train_samples) if i not in set(tumor_indices)]

        if len(tumor_indices) > 0 and len(non_tumor_indices) > 0:
            tumor_frac = len(tumor_indices) / total_train_samples
            n_tumor = max(1, round(n_samples * tumor_frac)) if n_samples > 1 else 1
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

        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING LABEL FRACTION: {frac*100:.0f}% ({n_samples}/{total_train_samples} training slices)")
        logger.info("="*80)

        # A. UNet Baseline
        logger.info(f"Training UNet Baseline on {frac*100:.0f}% labels...")
        t_start = time.perf_counter()
        unet = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
        loss_fn_bce = CombinedDiceBCELoss()
        opt_u = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
        sched_u = torch.optim.lr_scheduler.CosineAnnealingLR(opt_u, T_max=args.epochs, eta_min=1e-6)
        scaler_u = torch.amp.GradScaler('cuda', enabled=use_amp)
        for _ in range(args.epochs):
            unet.train()
            for b_idx, b in enumerate(train_loader):
                if args.max_batches and b_idx >= args.max_batches:
                    break
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

        m_unet = evaluate_model(unet, test_loader, device, use_amp=use_amp, max_batches=args.max_batches)
        unet_ckpt = ckpt_dir / f"unet_{frac_tag}.pt"
        torch.save({
            "model_state_dict": unet.state_dict(),
            "test_dice": m_unet["test_dice"],
            "test_dice_tumor": m_unet["test_dice_tumor"],
            "patient_3d_dice": m_unet["patient_3d_dice"],
        }, unet_ckpt)
        logger.info(
            f"Saved unet_{frac_tag}.pt | Test Dice: {m_unet['test_dice']:.4f} "
            f"(Tumor: {m_unet['test_dice_tumor']:.4f}, 3D: {m_unet['patient_3d_dice']:.4f}) | "
            f"Training Time: {u_time:.2f}s ({u_time/args.epochs:.2f}s/epoch)"
        )

        results.append({
            "label_fraction": f"{frac*100:.0f}%",
            "n_samples": n_samples,
            "model": "UNet Baseline",
            "test_dice": m_unet["test_dice"],
            "test_dice_tumor": m_unet["test_dice_tumor"],
            "test_iou": m_unet["test_iou"],
            "hd95_px": m_unet["hd95_px"],
            "patient_3d_dice": m_unet["patient_3d_dice"],
            "patient_3d_hd95_px": m_unet["patient_3d_hd95_px"],
            "train_time_sec": round(u_time, 2),
            "sec_per_epoch": round(u_time / args.epochs, 2),
        })

        # B. nnU-Net SOTA Baseline
        logger.info(f"Training nnU-Net SOTA Baseline on {frac*100:.0f}% labels...")
        t_start = time.perf_counter()
        nnunet = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
        loss_fn_ds = DeepSupervisionLoss()
        opt_nn = torch.optim.AdamW(nnunet.parameters(), lr=2e-4, weight_decay=1e-5)
        sched_nn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_nn, T_max=args.epochs, eta_min=1e-6)
        scaler_nn = torch.amp.GradScaler('cuda', enabled=use_amp)
        for _ in range(args.epochs):
            nnunet.train()
            for b_idx, b in enumerate(train_loader):
                if args.max_batches and b_idx >= args.max_batches:
                    break
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

        m_nnunet = evaluate_model(nnunet, test_loader, device, use_amp=use_amp, max_batches=args.max_batches)
        nnunet_ckpt = ckpt_dir / f"nnunet_{frac_tag}.pt"
        torch.save({
            "model_state_dict": nnunet.state_dict(),
            "test_dice": m_nnunet["test_dice"],
            "test_dice_tumor": m_nnunet["test_dice_tumor"],
            "patient_3d_dice": m_nnunet["patient_3d_dice"],
        }, nnunet_ckpt)
        logger.info(
            f"Saved nnunet_{frac_tag}.pt | Test Dice: {m_nnunet['test_dice']:.4f} "
            f"(Tumor: {m_nnunet['test_dice_tumor']:.4f}, 3D: {m_nnunet['patient_3d_dice']:.4f}) | "
            f"Training Time: {nn_time:.2f}s ({nn_time/args.epochs:.2f}s/epoch)"
        )

        results.append({
            "label_fraction": f"{frac*100:.0f}%",
            "n_samples": n_samples,
            "model": "nnU-Net (Supervised SOTA)",
            "test_dice": m_nnunet["test_dice"],
            "test_dice_tumor": m_nnunet["test_dice_tumor"],
            "test_iou": m_nnunet["test_iou"],
            "hd95_px": m_nnunet["hd95_px"],
            "patient_3d_dice": m_nnunet["patient_3d_dice"],
            "patient_3d_hd95_px": m_nnunet["patient_3d_hd95_px"],
            "train_time_sec": round(nn_time, 2),
            "sec_per_epoch": round(nn_time / args.epochs, 2),
        })

        # C. Pre-trained JEPA Variants (Fine-tuned)
        jepa_variants = ["ijepa", "sigreg_jepa", "visreg_jepa"]
        for type_name in jepa_variants:
            logger.info(f"Fine-tuning Pre-trained {type_name.upper()} ({args.decoder_type}) on {frac*100:.0f}% labels...")
            t_start = time.perf_counter()
            model = JEPASegmentationModel(
                img_size=240,
                patch_size=16,
                in_channels=4,
                embed_dim=384,
                out_channels=1,
                decoder_type=args.decoder_type,
            ).to(device)

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
                state_key = "target_encoder_state_dict" if (args.encoder_source == "target" and "target_encoder_state_dict" in ckpt) else "context_encoder_state_dict"
                logger.info(f"Using encoder weights from checkpoint key: '{state_key}' (requested source: {args.encoder_source})")
                model.load_pretrained_encoder(ckpt[state_key])
            else:
                logger.warning(f"No SSL pre-trained weights found for {type_name} in {ssl_base_dir}. Training from scratch!")

            opt_j = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched_j = torch.optim.lr_scheduler.CosineAnnealingLR(opt_j, T_max=args.epochs, eta_min=1e-6)
            scaler_j = torch.amp.GradScaler('cuda', enabled=use_amp)
            for _ in range(args.epochs):
                model.train()
                for b_idx, b in enumerate(train_loader):
                    if args.max_batches and b_idx >= args.max_batches:
                        break
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

            m_jepa = evaluate_model(model, test_loader, device, use_amp=use_amp, max_batches=args.max_batches)
            jepa_ckpt = ckpt_dir / f"finetuned_{type_name}_{frac_tag}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "decoder_type": args.decoder_type,
                "test_dice": m_jepa["test_dice"],
                "test_dice_tumor": m_jepa["test_dice_tumor"],
                "patient_3d_dice": m_jepa["patient_3d_dice"],
            }, jepa_ckpt)
            logger.info(
                f"Saved finetuned_{type_name}_{frac_tag}.pt | Test Dice: {m_jepa['test_dice']:.4f} "
                f"(Tumor: {m_jepa['test_dice_tumor']:.4f}, 3D: {m_jepa['patient_3d_dice']:.4f}) | "
                f"Training Time: {j_time:.2f}s ({j_time/args.epochs:.2f}s/epoch)"
            )

            clean_name = "I-JEPA (Fine-tuned)" if type_name == "ijepa" else ("SigReg JEPA (Fine-tuned)" if type_name == "sigreg_jepa" else "VisReg JEPA (Fine-tuned)")
            display_model_name = f"{clean_name} ({args.decoder_type})" if args.decoder_type != "bottleneck" else clean_name
            results.append({
                "label_fraction": f"{frac*100:.0f}%",
                "n_samples": n_samples,
                "model": display_model_name,
                "test_dice": m_jepa["test_dice"],
                "test_dice_tumor": m_jepa["test_dice_tumor"],
                "test_iou": m_jepa["test_iou"],
                "hd95_px": m_jepa["hd95_px"],
                "patient_3d_dice": m_jepa["patient_3d_dice"],
                "patient_3d_hd95_px": m_jepa["patient_3d_hd95_px"],
                "train_time_sec": round(j_time, 2),
                "sec_per_epoch": round(j_time / args.epochs, 2),
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*115)
    print("         LOW-DATA LABEL EFFICIENCY BENCHMARK SUMMARY (With 3D Volume & Slice Stratification)")
    print("="*115)
    print(summary_df.to_string(index=False))
    print("="*115 + "\n")

    out_csv = metrics_dir / "low_data_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved low-data benchmark summary to: {out_csv}")


if __name__ == "__main__":
    main()

