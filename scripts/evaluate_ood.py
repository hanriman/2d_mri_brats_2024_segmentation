import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_WORKERS,
    get_metadata_path,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa.data import BraTS2DDataset
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


def apply_rician_noise(img: torch.Tensor, noise_std: float = 0.15) -> torch.Tensor:
    """Applies additive Gaussian noise on foreground tissue approximating high-SNR asymptotic Rician noise.

    Note: In magnitude MRI data, noise follows a Rician distribution: M = sqrt((S + n1)^2 + n2^2).
    However, on zero-mean standardized Z-score normalized inputs (where x can be negative), applying
    the magnitude Rician formula directly would destroy negative values and distort the distribution.
    In the high-SNR regime (SNR >> 1), the Rician distribution is asymptotically Gaussian:
    Rician(nu, sigma) -> Normal(nu, sigma^2). We apply additive foreground Gaussian noise to
    faithfully simulate scanner acquisition noise on standardized slices without non-physical rectification.
    """
    noise = torch.randn_like(img) * noise_std
    mask = (img != 0).float()
    return (img + noise) * mask


def apply_bias_field(img: torch.Tensor, scale: float = 0.3) -> torch.Tensor:
    """Applies synthetic B1 intensity bias field to simulate coil sensitivity variations across scanner vendors."""
    _B, _C, H, W = img.shape
    y = torch.linspace(-1, 1, H, device=img.device).view(1, 1, H, 1)
    x = torch.linspace(-1, 1, W, device=img.device).view(1, 1, 1, W)
    bias_grad = scale * (x ** 2 + y ** 2 - 0.5)
    mask = (img != 0).float()
    return (img + bias_grad) * mask


def parse_args():
    parser = argparse.ArgumentParser(description="Out-of-Distribution (OOD) Scanner Domain Generalization Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--exp_version", type=str, default="v3_ood_generalization", help="Experiment version directory tag")
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
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP (mixed precision)")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches for rapid smoke testing")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


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

    if args.output_dir:
        base_out = Path(args.output_dir).resolve()
        exp_dir = base_out / "experiments" / args.exp_version if args.exp_version else base_out
    else:
        exp_dir = Path("outputs/experiments") / args.exp_version
    metrics_dir = exp_dir / "metrics"
    logs_dir = exp_dir / "logs"
    for d in [exp_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("evaluate_ood", logs_dir / "evaluate_ood.log")
    logger.info(f"Starting OOD Domain Generalization Benchmark ({args.exp_version}) on device: {device}")

    ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else CHECKPOINTS_DIR
    logger.info(f"Loading evaluation checkpoints from: {ckpt_dir}")

    if args.metadata_csv:
        metadata_path = Path(args.metadata_csv).resolve()
    else:
        metadata_path = get_metadata_path("brats_gli_2d")

    logger.info(f"Using OOD test metadata from: {metadata_path}")
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

    domain_shifts = {
        "Clean Standard Test": lambda x: x,
        "Rician Noise (1.5T Shift Approx)": lambda x: apply_rician_noise(x, noise_std=0.15),
        "Bias Field Inhomogeneity (Coil Shift)": lambda x: apply_bias_field(x, scale=0.35),
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

    for shift_name, transform_fn in domain_shifts.items():
        logger.info(f"\nEvaluating Domain Shift: {shift_name}...")
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
                dec_type = ckpt.get("decoder_type", "bottleneck")
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
                for batch_idx, batch in enumerate(test_loader):
                    if args.max_batches and batch_idx >= args.max_batches:
                        break
                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)
                    perturbed_images = transform_fn(images)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        logits = model(perturbed_images)

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
                "domain_shift": shift_name,
                "model": display_name,
                "test_dice": float(np.mean(dice_list)) if dice_list else 0.0,
                "test_dice_tumor": float(np.mean(dice_tumor_list)) if dice_tumor_list else float("nan"),
                "test_iou": float(np.mean(iou_list)) if iou_list else 0.0,
                "hd95_px": float(np.mean(hd95_list)) if hd95_list else 0.0,
                "patient_3d_dice": vol_metrics.get("patient_3d_dice_mean", float("nan")),
                "patient_3d_hd95_px": vol_metrics.get("patient_3d_hd95_mean", float("nan")),
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*115)
    print("      OUT-OF-DISTRIBUTION (OOD) SCANNER DOMAIN GENERALIZATION SUMMARY")
    print("="*115)
    print(summary_df.to_string(index=False))
    print("="*115 + "\n")

    out_csv = metrics_dir / "ood_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved OOD benchmark summary to: {out_csv}")


if __name__ == "__main__":
    main()

