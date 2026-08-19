import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from brats_jepa.config import CHECKPOINTS_DIR, DATA_DIR, LOGS_DIR, METRICS_DIR, ensure_directories
from brats_jepa.data import BraTS2DDataset
from brats_jepa.losses import CombinedDiceBCELoss, DeepSupervisionLoss
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import IJEPA, BraTS2DnnUNet, BraTS2DUNet, JEPASegmentationModel, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


class RandomModalityDropout(nn.Module):
    """Randomly zero out 1, 2, or 3 modality channels during training with probability p_drop."""
    def __init__(self, p_drop: float = 0.25):
        super().__init__()
        self.p_drop = p_drop
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_drop == 0:
            return x
        B, C, H, W = x.shape
        mask = (torch.rand(B, C, 1, 1, device=x.device) > self.p_drop).float()
        all_zero = (mask.sum(dim=1, keepdim=True) == 0)
        mask = torch.where(all_zero, torch.ones_like(mask), mask)
        return x * mask


def parse_args():
    parser = argparse.ArgumentParser(description="Low-Data Label Efficiency Benchmark Runner")
    parser.add_argument("--epochs", type=int, default=30, help="Downstream fine-tuning epochs per label fraction")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--p_drop", type=float, default=0.25, help="Modality dropout probability during training")
    parser.add_argument("--exp_version", type=str, default="v2_low_data_efficiency", help="Experiment version directory tag")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    
    # Versioned experiment directories
    exp_dir = Path("outputs/experiments") / args.exp_version
    ckpt_dir = exp_dir / "checkpoints"
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"
    for d in [exp_dir, ckpt_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    logger = get_logger("evaluate_low_data", logs_dir / "evaluate_low_data.log")
    logger.info(f"Starting Low-Data Label Efficiency Benchmark ({args.exp_version}) on device: {device}")
    
    metadata_path = (DATA_DIR / "processed" / "brats_gli_2d" / "metadata.csv").resolve()
    if not metadata_path.exists():
        metadata_path = Path("data/processed/2d_slices/metadata.csv").resolve()
        
    full_train_ds = BraTS2DDataset(metadata_csv=metadata_path, split="train")
    test_ds = BraTS2DDataset(metadata_csv=metadata_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    total_train_samples = len(full_train_ds)
    label_fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    mod_drop = RandomModalityDropout(p_drop=args.p_drop)
    
    results = []
    
    for frac in label_fractions:
        frac_tag = f"{int(round(frac*100))}pct"
        n_samples = max(1, int(round(frac * total_train_samples)))
        indices = list(range(n_samples))
        sub_train_ds = Subset(full_train_ds, indices)
        train_loader = DataLoader(sub_train_ds, batch_size=min(args.batch_size, n_samples), shuffle=True, num_workers=0)
        
        logger.info(f"\n" + "="*80)
        logger.info(f"EVALUATING LABEL FRACTION: {frac*100:.0f}% ({n_samples}/{total_train_samples} training slices)")
        logger.info("="*80)
        
        # A. UNet Baseline
        logger.info(f"Training UNet Baseline on {frac*100:.0f}% labels...")
        unet = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
        loss_fn_bce = CombinedDiceBCELoss()
        opt_u = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
        for _ in range(args.epochs):
            unet.train()
            for b in train_loader:
                opt_u.zero_grad()
                imgs = mod_drop(b["image"].to(device))
                l = loss_fn_bce(unet(imgs), b["label"].to(device))
                l.backward()
                opt_u.step()
        unet.eval()
        d_list, i_list, h_list = [], [], []
        with torch.no_grad():
            for b in test_loader:
                m = compute_segmentation_metrics(unet(b["image"].to(device)), b["label"].to(device))
                d_list.append(m["dice"])
                i_list.append(m["iou"])
                h_list.append(m["hd95"])
        
        unet_ckpt = ckpt_dir / f"unet_{frac_tag}.pt"
        torch.save({"model_state_dict": unet.state_dict(), "test_dice": float(np.mean(d_list))}, unet_ckpt)
        logger.info(f"Saved checkpoint to: {unet_ckpt.name}")
        
        results.append({
            "label_fraction": f"{frac*100:.0f}%",
            "n_samples": n_samples,
            "model": "UNet Baseline",
            "test_dice": float(np.mean(d_list)),
            "test_iou": float(np.mean(i_list)),
            "hd95_px": float(np.mean(h_list)),
        })
        
        # B. nnU-Net SOTA Baseline
        logger.info(f"Training nnU-Net SOTA Baseline on {frac*100:.0f}% labels...")
        nnunet = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
        loss_fn_ds = DeepSupervisionLoss()
        opt_nn = torch.optim.AdamW(nnunet.parameters(), lr=2e-4, weight_decay=1e-5)
        for _ in range(args.epochs):
            nnunet.train()
            for b in train_loader:
                opt_nn.zero_grad()
                imgs = mod_drop(b["image"].to(device))
                l = loss_fn_ds(nnunet(imgs), b["label"].to(device))
                l.backward()
                opt_nn.step()
        nnunet.eval()
        d_list, i_list, h_list = [], [], []
        with torch.no_grad():
            for b in test_loader:
                m = compute_segmentation_metrics(nnunet(b["image"].to(device)), b["label"].to(device))
                d_list.append(m["dice"])
                i_list.append(m["iou"])
                h_list.append(m["hd95"])
                
        nnunet_ckpt = ckpt_dir / f"nnunet_{frac_tag}.pt"
        torch.save({"model_state_dict": nnunet.state_dict(), "test_dice": float(np.mean(d_list))}, nnunet_ckpt)
        logger.info(f"Saved checkpoint to: {nnunet_ckpt.name}")
        
        results.append({
            "label_fraction": f"{frac*100:.0f}%",
            "n_samples": n_samples,
            "model": "nnU-Net (Supervised SOTA)",
            "test_dice": float(np.mean(d_list)),
            "test_iou": float(np.mean(i_list)),
            "hd95_px": float(np.mean(h_list)),
        })
        
        # C. Pre-trained JEPA Variants (Fine-tuned)
        jepa_variants = ["ijepa", "sigreg_jepa", "visreg_jepa"]
        for type_name in jepa_variants:
            logger.info(f"Fine-tuning Pre-trained {type_name.upper()} on {frac*100:.0f}% labels...")
            model = JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1).to(device)
            ssl_ckpt = CHECKPOINTS_DIR / f"best_{type_name}.pt"
            if ssl_ckpt.exists():
                ckpt = torch.load(ssl_ckpt, map_location=device)
                model.load_pretrained_encoder(ckpt["context_encoder_state_dict"])
                
            opt_j = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            for _ in range(args.epochs):
                model.train()
                for b in train_loader:
                    opt_j.zero_grad()
                    imgs = mod_drop(b["image"].to(device))
                    l = loss_fn_bce(model(imgs), b["label"].to(device))
                    l.backward()
                    opt_j.step()
            model.eval()
            d_list, i_list, h_list = [], [], []
            with torch.no_grad():
                for b in test_loader:
                    m = compute_segmentation_metrics(model(b["image"].to(device)), b["label"].to(device))
                    d_list.append(m["dice"])
                    i_list.append(m["iou"])
                    h_list.append(m["hd95"])
            
            jepa_ckpt = ckpt_dir / f"finetuned_{type_name}_{frac_tag}.pt"
            torch.save({"model_state_dict": model.state_dict(), "test_dice": float(np.mean(d_list))}, jepa_ckpt)
            logger.info(f"Saved checkpoint to: {jepa_ckpt.name}")
            
            clean_name = "I-JEPA (Fine-tuned)" if type_name == "ijepa" else ("SigReg JEPA (Fine-tuned)" if type_name == "sigreg_jepa" else "VisReg JEPA (Fine-tuned)")
            results.append({
                "label_fraction": f"{frac*100:.0f}%",
                "n_samples": n_samples,
                "model": clean_name,
                "test_dice": float(np.mean(d_list)),
                "test_iou": float(np.mean(i_list)),
                "hd95_px": float(np.mean(h_list)),
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*90)
    print("      LOW-DATA LABEL EFFICIENCY BENCHMARK SUMMARY (1% to 100% Labels)")
    print("="*90)
    print(summary_df.to_string(index=False))
    print("="*90 + "\n")
    
    out_csv = metrics_dir / "low_data_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved low-data benchmark summary to: {out_csv}")

if __name__ == "__main__":
    main()
