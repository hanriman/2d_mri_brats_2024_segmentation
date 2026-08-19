import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa.config import CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, ensure_directories
from brats_jepa.data import BraTS2DDataset
from brats_jepa.metrics import compute_representation_collapse_metrics, compute_segmentation_metrics
from brats_jepa.models import IJEPA, BraTS2DnnUNet, BraTS2DUNet, JEPASegmentationModel, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Downstream Segmentation, Representation Quality, and Runtime Benchmarks")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_directories()
    set_seed(args.seed)
    device = get_device(args.device)
    logger = get_logger("evaluate", LOGS_DIR / "evaluate.log")
    
    logger.info(f"Running evaluation benchmark on device: {device}")
    
    metadata_path = Path("data/processed/2d_slices/metadata.csv").resolve()
    test_ds = BraTS2DDataset(metadata_csv=metadata_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    num_samples = len(test_ds)
    
    logger.info(f"Loaded {num_samples} test slices.")
    results = []
    
    # 1. Evaluate Supervised UNet Baseline
    unet_ckpt = CHECKPOINTS_DIR / "best_unet.pt"
    if unet_ckpt.exists():
        logger.info(f"Evaluating standard UNet baseline from {unet_ckpt.name}...")
        unet = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
        ckpt = torch.load(unet_ckpt, map_location=device)
        unet.load_state_dict(ckpt["model_state_dict"])
        unet.eval()
        
        dummy_in = torch.randn(8, 4, 240, 240, device=device)
        with torch.no_grad():
            _ = unet(dummy_in)
            
        dice_list, iou_list, prec_list, rec_list, hd95_list = [], [], [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                logits = unet(images)
                m = compute_segmentation_metrics(logits, labels)
                dice_list.append(m["dice"])
                iou_list.append(m["iou"])
                prec_list.append(m["precision"])
                rec_list.append(m["recall"])
                hd95_list.append(m["hd95"])
        infer_time = time.perf_counter() - t0
        ms_per_slice = (infer_time / num_samples) * 1000.0 if num_samples > 0 else 0.0
        
        unet_json = METRICS_DIR / "unet_train_metrics.json"
        sec_per_epoch = "N/A"
        if unet_json.exists():
            with open(unet_json, "r") as f:
                d = json.load(f)
                if "epoch_duration_sec" in d:
                    sec_per_epoch = f"{pd.Series(d['epoch_duration_sec']).mean():.2f}s"
                
        results.append({
            "model": "UNet Baseline",
            "test_dice": f"{pd.Series(dice_list).mean():.4f}",
            "test_iou": f"{pd.Series(iou_list).mean():.4f}",
            "hd95_px": f"{pd.Series(hd95_list).mean():.2f}",
            "effective_rank": "N/A (CNN)",
            "avg_cosine_sim": "N/A",
            "infer_ms_per_slice": f"{ms_per_slice:.2f} ms",
            "train_sec_per_epoch": sec_per_epoch,
        })

    # 2. Evaluate Supervised SOTA 2D nnU-Net Baseline
    nnunet_ckpt = CHECKPOINTS_DIR / "best_nnunet.pt"
    if nnunet_ckpt.exists():
        logger.info(f"Evaluating 2D nnU-Net baseline from {nnunet_ckpt.name}...")
        nnunet = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
        ckpt = torch.load(nnunet_ckpt, map_location=device)
        nnunet.load_state_dict(ckpt["model_state_dict"])
        nnunet.eval()
        
        dummy_in = torch.randn(8, 4, 240, 240, device=device)
        with torch.no_grad():
            _ = nnunet(dummy_in)
            
        dice_list, iou_list, prec_list, rec_list, hd95_list = [], [], [], [], []
        t0 = time.perf_counter()
        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                logits = nnunet(images)
                m = compute_segmentation_metrics(logits, labels)
                dice_list.append(m["dice"])
                iou_list.append(m["iou"])
                prec_list.append(m["precision"])
                rec_list.append(m["recall"])
                hd95_list.append(m["hd95"])
        infer_time = time.perf_counter() - t0
        ms_per_slice = (infer_time / num_samples) * 1000.0 if num_samples > 0 else 0.0
        
        nnunet_json = METRICS_DIR / "nnunet_train_metrics.json"
        sec_per_epoch = "N/A"
        if nnunet_json.exists():
            with open(nnunet_json, "r") as f:
                d = json.load(f)
                if "epoch_duration_sec" in d:
                    sec_per_epoch = f"{pd.Series(d['epoch_duration_sec']).mean():.2f}s"
                
        results.append({
            "model": "nnU-Net (Supervised SOTA)",
            "test_dice": f"{pd.Series(dice_list).mean():.4f}",
            "test_iou": f"{pd.Series(iou_list).mean():.4f}",
            "hd95_px": f"{pd.Series(hd95_list).mean():.2f}",
            "effective_rank": "N/A (CNN)",
            "avg_cosine_sim": "N/A",
            "infer_ms_per_slice": f"{ms_per_slice:.2f} ms",
            "train_sec_per_epoch": sec_per_epoch,
        })
        
    # 3. Evaluate Self-Supervised JEPA Variants
    jepa_variants = {
        "I-JEPA": ("ijepa", IJEPA),
        "SigReg JEPA": ("sigreg_jepa", SigRegJEPA),
        "VisReg JEPA": ("visreg_jepa", VisRegJEPA),
    }
    
    for name, (type_name, ssl_model_cls) in jepa_variants.items():
        finetuned_ckpt = CHECKPOINTS_DIR / f"best_finetuned_{type_name}.pt"
        ssl_ckpt = CHECKPOINTS_DIR / f"best_{type_name}.pt"
        
        eff_rank_str, cosine_sim_str = "N/A", "N/A"
        if ssl_ckpt.exists():
            ssl_model = ssl_model_cls(img_size=240, patch_size=16, in_channels=4, embed_dim=384).to(device)
            ckpt = torch.load(ssl_ckpt, map_location=device)
            ssl_model.context_encoder.load_state_dict(ckpt["context_encoder_state_dict"])
            ssl_model.eval()
            
            ranks, sims = [], []
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["image"].to(device)
                    tokens = ssl_model.context_encoder(images)
                    rep_metrics = compute_representation_collapse_metrics(tokens)
                    ranks.append(rep_metrics["effective_rank"])
                    sims.append(rep_metrics["avg_cosine_sim"])
            eff_rank_str = f"{pd.Series(ranks).mean():.2f}"
            cosine_sim_str = f"{pd.Series(sims).mean():.4f}"
            
        if finetuned_ckpt.exists():
            logger.info(f"Evaluating fine-tuned downstream segmentation for {name} from {finetuned_ckpt.name}...")
            model = JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1).to(device)
            ckpt = torch.load(finetuned_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            
            dummy_in = torch.randn(8, 4, 240, 240, device=device)
            with torch.no_grad():
                _ = model(dummy_in)
                
            dice_list, iou_list, prec_list, rec_list, hd95_list = [], [], [], [], []
            t0 = time.perf_counter()
            with torch.no_grad():
                for batch in test_loader:
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    logits = model(images)
                    m = compute_segmentation_metrics(logits, labels)
                    dice_list.append(m["dice"])
                    iou_list.append(m["iou"])
                    prec_list.append(m["precision"])
                    rec_list.append(m["recall"])
                    hd95_list.append(m["hd95"])
            infer_time = time.perf_counter() - t0
            ms_per_slice = (infer_time / num_samples) * 1000.0 if num_samples > 0 else 0.0
            
            ft_json = METRICS_DIR / f"finetuned_{type_name}_metrics.json"
            sec_per_epoch = "N/A"
            if ft_json.exists():
                with open(ft_json, "r") as f:
                    d = json.load(f)
                    if "epoch_duration_sec" in d:
                        sec_per_epoch = f"{pd.Series(d['epoch_duration_sec']).mean():.2f}s"
                        
            results.append({
                "model": f"{name} (Fine-tuned)",
                "test_dice": f"{pd.Series(dice_list).mean():.4f}",
                "test_iou": f"{pd.Series(iou_list).mean():.4f}",
                "hd95_px": f"{pd.Series(hd95_list).mean():.2f}",
                "effective_rank": eff_rank_str,
                "avg_cosine_sim": cosine_sim_str,
                "infer_ms_per_slice": f"{ms_per_slice:.2f} ms",
                "train_sec_per_epoch": sec_per_epoch,
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*110)
    print("      RESEARCH EVALUATION BENCHMARK & RUNTIME TIMING SUMMARY (2D BraTS GLI)")
    print("="*110)
    print(summary_df.to_string(index=False))
    print("="*110 + "\n")
    
    out_csv = METRICS_DIR / "evaluation_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved benchmark summary to: {out_csv}")

if __name__ == "__main__":
    main()
