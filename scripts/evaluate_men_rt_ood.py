import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from brats_jepa.config import CHECKPOINTS_DIR, DATA_DIR
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import IJEPA, BraTS2DnnUNet, BraTS2DUNet, JEPASegmentationModel, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import get_device, get_logger, set_seed


class BraTSMENRTDataset(Dataset):
    """Dataset for processed 2D BraTS-MEN-RT (Meningioma) slices."""
    def __init__(self, metadata_csv: Path, max_samples: int = 1000, tumor_only: bool = True):
        df = pd.read_csv(metadata_csv)
        if tumor_only:
            df = df[df["has_tumor"] == True].copy()
        if max_samples > 0 and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            
        self.df = df
        self.data_dir = metadata_csv.parent
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = Path(row["file_path"]).name
        file_path = self.data_dir / file_name
        data = np.load(file_path)
        
        image = torch.from_numpy(data["image"]).float()  # [1, 240, 240]
        mask = torch.from_numpy(data["mask"]).float()    # [1, 240, 240]
        return {
            "image": image,
            "label": mask,
            "patient_id": row["patient_id"],
            "has_tumor": row["has_tumor"],
        }

def parse_args():
    parser = argparse.ArgumentParser(description="BraTS-MEN-RT Cross-Pathology & Missing-Modality OOD Benchmark")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum tumor slices for fast evaluation")
    parser.add_argument("--exp_version", type=str, default="v4_men_rt_ood", help="Experiment version directory tag")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output base directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv manifest file")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    if args.output_dir:
        base_out = Path(args.output_dir).resolve()
        exp_dir = base_out / "experiments" / args.exp_version if args.exp_version else base_out
    else:
        exp_dir = Path("outputs/experiments") / args.exp_version
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"
    for d in [exp_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("evaluate_men_rt_ood", logs_dir / "evaluate_men_rt_ood.log")
    logger.info(f"Starting BraTS-MEN-RT Cross-Pathology & Missing-Modality OOD Benchmark ({args.exp_version})")

    ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else CHECKPOINTS_DIR
    logger.info(f"Loading evaluation checkpoints from: {ckpt_dir}")

    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).resolve()
    else:
        from brats_jepa.config import get_metadata_path
        metadata_path = get_metadata_path("brats_men_rt_2d")

    if not metadata_path.exists():
        logger.error(f"Metadata manifest not found at {metadata_path}. Please run prepare_brats_men_rt.py first.")
        return

    logger.info(f"Using Meningioma dataset metadata from: {metadata_path}")
    dataset = BraTSMENRTDataset(metadata_path, max_samples=args.max_samples, tumor_only=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    logger.info(f"Loaded {len(dataset)} Meningioma tumor 2D slices for OOD evaluation.")

    # 4-channel adaptation strategies
    strategies = {
        "Channel Replication [T1c, T1c, T1c, T1c]": lambda img1c: img1c.repeat(1, 4, 1, 1),
        "Zero-Padding Missing Channels [0, T1c, 0, 0]": lambda img1c: torch.cat([torch.zeros_like(img1c), img1c, torch.zeros_like(img1c), torch.zeros_like(img1c)], dim=1),
    }

    def resolve_ckpt(primary_name, fallback_name):
        p1 = ckpt_dir / primary_name
        if p1.exists():
            return p1
        p2 = ckpt_dir / fallback_name
        if p2.exists():
            return p2
        return p1

    models = {
        "UNet Baseline": (BraTS2DUNet(in_channels=4, out_channels=1), resolve_ckpt("best_unet.pt", "unet_100pct.pt")),
        "nnU-Net (Supervised SOTA)": (BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True), resolve_ckpt("best_nnunet.pt", "nnunet_100pct.pt")),
        "I-JEPA (Fine-tuned)": (JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1), resolve_ckpt("best_finetuned_ijepa.pt", "finetuned_ijepa_100pct.pt")),
        "SigReg JEPA (Fine-tuned)": (JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1), resolve_ckpt("best_finetuned_sigreg_jepa.pt", "finetuned_sigreg_jepa_100pct.pt")),
        "VisReg JEPA (Fine-tuned)": (JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=384, out_channels=1), resolve_ckpt("best_finetuned_visreg_jepa.pt", "finetuned_visreg_jepa_100pct.pt")),
    }

    
    results = []
    
    for strat_name, adapt_fn in strategies.items():
        logger.info(f"\n" + "="*80)
        logger.info(f"EVALUATING ADAPTATION STRATEGY: {strat_name}")
        logger.info("="*80)
        
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
                for batch in loader:
                    img_1c, labels = batch["image"].to(device), batch["label"].to(device)
                    img_4c = adapt_fn(img_1c)
                    logits = model(img_4c)
                    m = compute_segmentation_metrics(logits, labels)
                    d_list.extend(m["dice_per_sample"])
                    i_list.extend(m["iou_per_sample"])
                    h_list.extend(m["hd95_per_sample"])
                    
            results.append({
                "adaptation_strategy": strat_name,
                "model": model_name,
                "men_rt_test_dice": float(np.mean(d_list)),
                "men_rt_test_iou": float(np.mean(i_list)),
                "hd95_px": float(np.mean(h_list)),
            })
            
    summary_df = pd.DataFrame(results)
    print("\n" + "="*95)
    print("   BraTS-MEN-RT CROSS-PATHOLOGY & MISSING-MODALITY OOD SUMMARY")
    print("="*95)
    print(summary_df.to_string(index=False))
    print("="*95 + "\n")
    
    out_csv = metrics_dir / "men_rt_ood_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved BraTS-MEN-RT OOD benchmark summary to: {out_csv}")

if __name__ == "__main__":
    main()
