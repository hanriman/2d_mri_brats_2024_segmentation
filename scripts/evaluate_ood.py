import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa.config import CHECKPOINTS_DIR, DATA_DIR, LOGS_DIR, METRICS_DIR, ensure_directories
from brats_jepa.data import BraTS2DDataset
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import IJEPA, BraTS2DnnUNet, BraTS2DUNet, JEPASegmentationModel, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import get_device, get_logger, set_seed


def apply_rician_noise(img: torch.Tensor, noise_std: float = 0.15) -> torch.Tensor:
    """Applies synthetic Rician noise to simulate low SNR / 1.5T MRI scanner acquisitions."""
    n1 = torch.randn_like(img) * noise_std
    n2 = torch.randn_like(img) * noise_std
    return torch.sqrt((img + n1) ** 2 + n2 ** 2)

def apply_bias_field(img: torch.Tensor, scale: float = 0.3) -> torch.Tensor:
    """Applies synthetic B1 intensity bias field to simulate coil sensitivity variations across scanner vendors."""
    B, C, H, W = img.shape
    y = torch.linspace(-1, 1, H, device=img.device).view(1, 1, H, 1)
    x = torch.linspace(-1, 1, W, device=img.device).view(1, 1, 1, W)
    bias = 1.0 + scale * (x ** 2 + y ** 2)
    return img * bias

def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-Distribution (OOD) Scanner Domain Generalization Benchmark")
    parser.add_argument("--exp_version", type=str, default="v3_ood_generalization", help="Experiment version directory tag")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    
    exp_dir = Path("outputs/experiments") / args.exp_version
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"
    for d in [exp_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    logger = get_logger("evaluate_ood", logs_dir / "evaluate_ood.log")
    logger.info(f"Starting OOD Domain Generalization Benchmark ({args.exp_version}) on device: {device}")
    
    metadata_path = (DATA_DIR / "processed" / "brats_gli_2d" / "metadata.csv").resolve()
    if not metadata_path.exists():
        metadata_path = Path("data/processed/2d_slices/metadata.csv").resolve()
        
    test_ds = BraTS2DDataset(metadata_csv=metadata_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    
    domain_shifts = {
        "Clean Standard Test": lambda x: x,
        "Rician Noise (1.5T Shift)": lambda x: apply_rician_noise(x, noise_std=0.15),
        "Bias Field Inhomogeneity (Coil Shift)": lambda x: apply_bias_field(x, scale=0.35),
    }
    
    models = {
        "UNet Baseline": (BraTS2DUNet(in_channels=4, out_channels=1), CHECKPOINTS_DIR / "best_unet.pt"),
        "nnU-Net (Supervised SOTA)": (BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True), CHECKPOINTS_DIR / "best_nnunet.pt"),
        "I-JEPA (Fine-tuned)": (JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1), CHECKPOINTS_DIR / "best_finetuned_ijepa.pt"),
        "SigReg JEPA (Fine-tuned)": (JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1), CHECKPOINTS_DIR / "best_finetuned_sigreg_jepa.pt"),
        "VisReg JEPA (Fine-tuned)": (JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1), CHECKPOINTS_DIR / "best_finetuned_visreg_jepa.pt"),
    }
    
    results = []
    
    for shift_name, transform_fn in domain_shifts.items():
        logger.info(f"\nEvaluating Domain Shift: {shift_name}...")
        for model_name, (model, ckpt_path) in models.items():
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint {ckpt_path.name} not found. Skipping {model_name}.")
                continue
                
            model = model.to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            d_list, i_list, h_list = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    perturbed_images = transform_fn(images)
                    logits = model(perturbed_images)
                    m = compute_segmentation_metrics(logits, labels)
                    d_list.append(m["dice"])
                    i_list.append(m["iou"])
                    h_list.append(m["hd95"])
                    
            results.append({
                "domain_shift": shift_name,
                "model": model_name,
                "test_dice": float(np.mean(d_list)),
                "test_iou": float(np.mean(i_list)),
                "hd95_px": float(np.mean(h_list)),
            })
            
    summary_df = pd.DataFrame(results)
    print("\n" + "="*95)
    print("      OUT-OF-DISTRIBUTION (OOD) SCANNER DOMAIN GENERALIZATION SUMMARY")
    print("="*95)
    print(summary_df.to_string(index=False))
    print("="*95 + "\n")
    
    out_csv = metrics_dir / "ood_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved OOD benchmark summary to: {out_csv}")

if __name__ == "__main__":
    main()
