import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_WORKERS,
    OUTPUTS_DIR,
    get_metadata_path,
    load_yaml_config,
    merge_config_with_args,
)
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


class BraTSMENRTDataset(Dataset):
    """Dataset for processed 2D BraTS-MEN-RT (Meningioma) slices."""

    def __init__(
        self,
        metadata_csv: Path,
        max_samples: int = 1000,
        tumor_only: bool = True,
        cache_in_memory: bool = True,
    ):
        df = pd.read_csv(metadata_csv)
        if tumor_only:
            df = df[df["has_tumor"].isin([True, "True", 1, "1"])].copy()
        if max_samples > 0 and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.df = df
        self.data_dir = metadata_csv.parent
        self.cache_in_memory = cache_in_memory
        self.cache: dict[int, dict[str, torch.Tensor | str | int | bool]] = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]

        row = self.df.iloc[idx]
        file_name = Path(row["file_path"]).name
        file_path = self.data_dir / file_name
        data = np.load(file_path)

        image = torch.from_numpy(data["image"]).float()  # [1, 240, 240]
        mask = torch.from_numpy(data["mask"]).float()    # [1, 240, 240]
        sample = {
            "image": image,
            "label": mask,
            "patient_id": str(row["patient_id"]),
            "has_tumor": bool(row["has_tumor"]),
            "slice_index": int(row["slice_index"]) if "slice_index" in row else idx,
        }
        if self.cache_in_memory:
            self.cache[idx] = sample
        return sample


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS-MEN-RT Cross-Pathology & Missing-Modality OOD Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum tumor slices for fast evaluation")
    parser.add_argument("--exp_version", type=str, default="v4_men_rt_ood", help="Experiment version directory tag")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output base directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv manifest file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--cache_data", action="store_true", default=True,
                        help="Cache loaded slices in RAM to eliminate disk I/O bottlenecks")
    parser.add_argument("--no_cache_data", action="store_false", dest="cache_data",
                        help="Disable RAM caching of slices")
    parser.add_argument("--decoder_type", type=str, choices=["bottleneck", "multiscale"], default=None,
                        help="Decoder architecture override for JEPA downstream segmentation models")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP (mixed precision)")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches for rapid smoke testing")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enforce strict cuDNN determinism (disables cuDNN benchmark)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        cfg = load_yaml_config(args.config)
        args = merge_config_with_args(cfg, args)

    set_seed(args.seed)
    device = get_device(args.device)

    # Enable cuDNN benchmark for static-sized convolutions on CUDA (unless strict determinism is requested)
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    base_out = Path(args.output_dir).resolve() if args.output_dir else OUTPUTS_DIR
    exp_dir = base_out / "experiments" / args.exp_version if args.exp_version else base_out
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
        metadata_path = get_metadata_path("brats_men_rt_2d")

    if not metadata_path.exists():
        logger.error(f"Metadata manifest not found at {metadata_path}. Please run prepare_brats_men_rt.py first.")
        return

    logger.info(f"Using Meningioma dataset metadata from: {metadata_path}")
    dataset = BraTSMENRTDataset(metadata_path, max_samples=args.max_samples, tumor_only=True, cache_in_memory=args.cache_data)
    use_cuda = (device.type == "cuda")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
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

    model_candidates = [
        ("UNet Baseline", "unet", resolve_ckpt("best_unet.pt", "unet_100pct.pt")),
        ("nnU-Net (Supervised SOTA)", "nnunet", resolve_ckpt("best_nnunet.pt", "nnunet_100pct.pt")),
        ("I-JEPA (Fine-tuned)", "ijepa", resolve_ckpt("best_finetuned_ijepa.pt", "finetuned_ijepa_100pct.pt")),
        ("SigReg JEPA (Fine-tuned)", "sigreg_jepa", resolve_ckpt("best_finetuned_sigreg_jepa.pt", "finetuned_sigreg_jepa_100pct.pt")),
        ("VisReg JEPA (Fine-tuned)", "visreg_jepa", resolve_ckpt("best_finetuned_visreg_jepa.pt", "finetuned_visreg_jepa_100pct.pt")),
    ]

    use_amp = args.amp and device.type == "cuda"
    results = []

    for strat_name, adapt_fn in strategies.items():
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATING ADAPTATION STRATEGY: {strat_name}")
        logger.info("="*80)

        for model_name, model_type, ckpt_path in model_candidates:
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint {ckpt_path.name} not found. Skipping {model_name}.")
                continue

            ckpt = torch.load(ckpt_path, map_location=device)
            if model_type == "unet":
                model = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
                display_name = model_name
            elif model_type == "nnunet":
                model = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
                display_name = model_name
            else:
                dec_type = getattr(args, "decoder_type", None) or ckpt.get("decoder_type", "bottleneck")
                model = JEPASegmentationModel(
                    img_size=240,
                    patch_size=16,
                    in_channels=4,
                    embed_dim=384,
                    out_channels=1,
                    decoder_type=dec_type,
                ).to(device)
                display_name = f"{model_name} ({dec_type})" if dec_type != "bottleneck" else model_name

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            dice_list, iou_list, hd95_list = [], [], []
            dice_tumor_list = []
            all_logits, all_labels = [], []
            all_patients, all_slice_idxs = [], []

            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    if args.max_batches and batch_idx >= args.max_batches:
                        break
                    img_1c, labels = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
                    img_4c = adapt_fn(img_1c)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(img_4c)

                    m = compute_segmentation_metrics(logits, labels)
                    dice_list.extend(m["dice_per_sample"])
                    iou_list.extend(m["iou_per_sample"])
                    hd95_list.extend(m["hd95_per_sample"])

                    if "has_tumor_per_sample" in m:
                        for d, has_t in zip(m["dice_per_sample"], m["has_tumor_per_sample"]):
                            if has_t:
                                dice_tumor_list.append(d)

                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())
                    all_patients.extend(batch["patient_id"])
                    all_slice_idxs.extend(
                        batch["slice_index"].tolist()
                        if torch.is_tensor(batch["slice_index"])
                        else batch["slice_index"]
                    )

            vol_metrics = compute_patient_volume_metrics(
                slice_preds=all_logits,
                slice_targets=all_labels,
                patient_ids=all_patients,
                slice_indices=all_slice_idxs,
            )

            results.append({
                "adaptation_strategy": strat_name,
                "model": display_name,
                "men_rt_test_dice": float(np.mean(dice_list)) if dice_list else 0.0,
                "men_rt_test_dice_tumor": float(np.mean(dice_tumor_list)) if dice_tumor_list else float("nan"),
                "men_rt_test_iou": float(np.mean(iou_list)) if iou_list else 0.0,
                "hd95_px": float(np.mean(hd95_list)) if hd95_list else 0.0,
                "patient_3d_dice": vol_metrics.get("dice_3d", vol_metrics.get("patient_3d_dice_mean", float("nan"))),
                "patient_3d_hd95_px": vol_metrics.get("hd95_3d", vol_metrics.get("patient_3d_hd95_mean", float("nan"))),
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*115)
    print("   BraTS-MEN-RT CROSS-PATHOLOGY & MISSING-MODALITY OOD SUMMARY")
    print("="*115)
    print(summary_df.to_string(index=False))
    print("="*115 + "\n")

    out_csv = metrics_dir / "men_rt_ood_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved BraTS-MEN-RT OOD benchmark summary to: {out_csv}")


if __name__ == "__main__":
    main()

