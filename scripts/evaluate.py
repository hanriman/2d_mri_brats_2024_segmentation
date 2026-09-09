import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_WORKERS,
    METRICS_DIR,
    OUTPUTS_DIR,
    get_metadata_path,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa.data import BraTS2DDataset
from brats_jepa.metrics import (
    compute_patient_volume_metrics,
    compute_representation_collapse_metrics,
    compute_segmentation_metrics,
)
from brats_jepa.models import (
    IJEPA,
    BraTS2DnnUNet,
    BraTS2DUNet,
    JEPASegmentationModel,
    SigRegJEPA,
    VisRegJEPA,
)
from brats_jepa.utils import get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Downstream Segmentation, Representation Quality, and Runtime Benchmarks")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv manifest file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing model checkpoints")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save summary metrics")
    parser.add_argument("--exp_version", type=str, default=None, help="Versioned experiment directory tag")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Number of DataLoader worker processes (default: 2 on Linux, 0 on macOS)")
    parser.add_argument("--cache_data", action="store_true", default=True,
                        help="Cache loaded slices in RAM to eliminate disk I/O bottlenecks")
    parser.add_argument("--no_cache_data", action="store_false", dest="cache_data",
                        help="Disable RAM caching of slices")
    parser.add_argument("--decoder_type", type=str, default=None, choices=["bottleneck", "multiscale"],
                        help="Decoder architecture for downstream segmentation (overrides checkpoint if provided)")
    parser.add_argument("--encoder_source", type=str, default="target", choices=["target", "context"],
                        help="Which pre-trained encoder weights to evaluate representations for (default: target)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA/MPS AMP (mixed precision)")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches for quick local smoke testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Enforce strict cuDNN determinism (disables cuDNN benchmark)")
    return parser.parse_args()


def evaluate_segmentation_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """
    Evaluates a 2D segmentation model across both slice-wise and patient-level 3D volumetric metrics.
    """
    dice_list, iou_list, prec_list, rec_list, hd95_list = [], [], [], [], []
    dice_tumor_list, iou_tumor_list, hd95_tumor_list = [], [], []
    all_logits, all_labels = [], []
    all_patients, all_slice_idxs = [], []

    # Warmup pass
    dummy_in = torch.randn(min(4, test_loader.batch_size or 4), 4, 240, 240, device=device)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
        _ = model(dummy_in)

    total_forward_time = 0.0
    eval_samples = 0
    is_cuda = (device.type == "cuda")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            eval_samples += images.shape[0]

            # Synchronize CUDA to measure pure model execution time accurately (finding M3)
            if is_cuda:
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)

            if is_cuda:
                torch.cuda.synchronize()
            total_forward_time += (time.perf_counter() - t_fwd_start)

            m = compute_segmentation_metrics(logits, labels)

            dice_list.extend(m["dice_per_sample"])
            iou_list.extend(m["iou_per_sample"])
            prec_list.extend(m["precision_per_sample"])
            rec_list.extend(m["recall_per_sample"])
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

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_patients.extend(batch["patient_id"])
            all_slice_idxs.extend(
                batch["slice_index"].tolist()
                if torch.is_tensor(batch["slice_index"])
                else batch["slice_index"]
            )

    ms_per_slice = (total_forward_time / eval_samples) * 1000.0 if eval_samples > 0 else 0.0

    # Official BraTS 3D Patient Volumetric Metric Accumulation
    vol_metrics = compute_patient_volume_metrics(
        slice_preds=all_logits,
        slice_targets=all_labels,
        patient_ids=all_patients,
        slice_indices=all_slice_idxs,
    )

    dice_tumor_mean = float(np.mean(dice_tumor_list)) if dice_tumor_list else float("nan")
    p3d_dice = vol_metrics.get("dice_3d", vol_metrics.get("patient_3d_dice_mean", float("nan")))
    p3d_hd95 = vol_metrics.get("hd95_3d", vol_metrics.get("patient_3d_hd95_mean", float("nan")))

    return {
        "test_dice_all": float(np.mean(dice_list)) if dice_list else 0.0,
        "test_dice_tumor": dice_tumor_mean,
        "test_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "hd95_2d_px": float(np.mean(hd95_list)) if hd95_list else 0.0,
        "patient_3d_dice": p3d_dice,
        "patient_3d_hd95_px": p3d_hd95,
        "ms_per_slice": ms_per_slice,
    }


def main():
    args = parse_args()
    if args.config:
        cfg = load_yaml_config(args.config)
        args = merge_config_with_args(cfg, args)

    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = args.amp and (device.type in ["cuda", "mps"])

    # Enable cuDNN benchmark for static-sized convolutions on CUDA (unless strict determinism is requested)
    if device.type == "cuda" and not args.deterministic:
        torch.backends.cudnn.benchmark = True

    # Resolve output and checkpoint directories
    root_out = Path(args.output_dir).resolve() if args.output_dir else OUTPUTS_DIR
    if args.exp_version:
        base_out = root_out / "experiments" / args.exp_version
    else:
        base_out = root_out

    ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else (base_out / "checkpoints" if (args.output_dir or args.exp_version) else CHECKPOINTS_DIR)
    metrics_dir = base_out / "metrics"
    logs_dir = base_out / "logs"
    for d in [base_out, ckpt_dir, metrics_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = get_logger("evaluate", logs_dir / "evaluate.log")
    logger.info(f"Running evaluation benchmark on device: {device}")
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) enabled for evaluation.")
    logger.info(f"Loading checkpoints from: {ckpt_dir}")
    logger.info(f"DataLoader settings: num_workers={args.num_workers}, cache_in_memory={args.cache_data}")

    metadata_path = Path(args.metadata_csv).resolve() if args.metadata_csv else get_metadata_path("brats_gli_2d")
    logger.info(f"Using test dataset metadata from: {metadata_path}")

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
    num_samples = len(test_ds)

    logger.info(f"Loaded {num_samples} test slices.")
    results = []

    # 1. Evaluate Supervised UNet Baseline
    unet_candidates = [
        ckpt_dir / "best_unet.pt",
        ckpt_dir / "unet_100pct.pt",
        CHECKPOINTS_DIR / "best_unet.pt",
        CHECKPOINTS_DIR / "unet_100pct.pt",
    ]
    unet_ckpt = next((c for c in unet_candidates if c.exists()), unet_candidates[0])

    if unet_ckpt.exists():
        logger.info(f"Evaluating standard UNet baseline from {unet_ckpt.name}...")
        unet = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
        ckpt = torch.load(unet_ckpt, map_location=device)
        unet.load_state_dict(ckpt["model_state_dict"])
        unet.eval()

        m_dict = evaluate_segmentation_model(unet, test_loader, device, use_amp=use_amp, max_batches=args.max_batches)

        unet_json = metrics_dir / "unet_train_metrics.json"
        if not unet_json.exists():
            unet_json = METRICS_DIR / "unet_train_metrics.json"
        sec_per_epoch = "N/A"
        if unet_json.exists():
            with open(unet_json, "r") as f:
                d = json.load(f)
                if "epoch_duration_sec" in d:
                    sec_per_epoch = f"{pd.Series(d['epoch_duration_sec']).mean():.2f}s"

        results.append({
            "model": "UNet Baseline",
            "test_dice": f"{m_dict['test_dice_all']:.4f}",
            "test_dice_all": f"{m_dict['test_dice_all']:.4f}",
            "test_dice_tumor": f"{m_dict['test_dice_tumor']:.4f}" if not np.isnan(m_dict['test_dice_tumor']) else "N/A",
            "test_iou": f"{m_dict['test_iou']:.4f}",
            "hd95_px": f"{m_dict['hd95_2d_px']:.2f}",
            "hd95_2d_px": f"{m_dict['hd95_2d_px']:.2f}",
            "patient_3d_dice": f"{m_dict['patient_3d_dice']:.4f}" if not np.isnan(m_dict['patient_3d_dice']) else "N/A",
            "patient_3d_hd95_px": f"{m_dict['patient_3d_hd95_px']:.2f}" if not np.isnan(m_dict['patient_3d_hd95_px']) else "N/A",
            "effective_rank": "N/A (CNN)",
            "avg_cosine_sim": "N/A",
            "infer_ms_per_slice": f"{m_dict['ms_per_slice']:.2f} ms",
            "train_sec_per_epoch": sec_per_epoch,
        })

    # 2. Evaluate Supervised SOTA 2D nnU-Net Baseline
    nnunet_candidates = [
        ckpt_dir / "best_nnunet.pt",
        ckpt_dir / "nnunet_100pct.pt",
        CHECKPOINTS_DIR / "best_nnunet.pt",
        CHECKPOINTS_DIR / "nnunet_100pct.pt",
    ]
    nnunet_ckpt = next((c for c in nnunet_candidates if c.exists()), nnunet_candidates[0])

    if nnunet_ckpt.exists():
        logger.info(f"Evaluating 2D nnU-Net baseline from {nnunet_ckpt.name}...")
        nnunet = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
        ckpt = torch.load(nnunet_ckpt, map_location=device)
        nnunet.load_state_dict(ckpt["model_state_dict"])
        nnunet.eval()

        m_dict = evaluate_segmentation_model(nnunet, test_loader, device, use_amp=use_amp, max_batches=args.max_batches)

        nnunet_json = metrics_dir / "nnunet_train_metrics.json"
        if not nnunet_json.exists():
            nnunet_json = METRICS_DIR / "nnunet_train_metrics.json"
        sec_per_epoch = "N/A"
        if nnunet_json.exists():
            with open(nnunet_json, "r") as f:
                d = json.load(f)
                if "epoch_duration_sec" in d:
                    sec_per_epoch = f"{pd.Series(d['epoch_duration_sec']).mean():.2f}s"

        results.append({
            "model": "nnU-Net (Supervised SOTA)",
            "test_dice": f"{m_dict['test_dice_all']:.4f}",
            "test_dice_all": f"{m_dict['test_dice_all']:.4f}",
            "test_dice_tumor": f"{m_dict['test_dice_tumor']:.4f}" if not np.isnan(m_dict['test_dice_tumor']) else "N/A",
            "test_iou": f"{m_dict['test_iou']:.4f}",
            "hd95_px": f"{m_dict['hd95_2d_px']:.2f}",
            "hd95_2d_px": f"{m_dict['hd95_2d_px']:.2f}",
            "patient_3d_dice": f"{m_dict['patient_3d_dice']:.4f}" if not np.isnan(m_dict['patient_3d_dice']) else "N/A",
            "patient_3d_hd95_px": f"{m_dict['patient_3d_hd95_px']:.2f}" if not np.isnan(m_dict['patient_3d_hd95_px']) else "N/A",
            "effective_rank": "N/A (CNN)",
            "avg_cosine_sim": "N/A",
            "avg_cosine_sim_centered": "N/A",
            "infer_ms_per_slice": f"{m_dict['ms_per_slice']:.2f} ms",
            "train_sec_per_epoch": sec_per_epoch,
        })

    # 3. Evaluate Self-Supervised JEPA Variants
    jepa_variants = {
        "I-JEPA": ("ijepa", IJEPA),
        "SigReg JEPA": ("sigreg_jepa", SigRegJEPA),
        "VisReg JEPA": ("visreg_jepa", VisRegJEPA),
    }

    for name, (type_name, ssl_model_cls) in jepa_variants.items():
        finetuned_candidates = [
            ckpt_dir / f"best_finetuned_{type_name}.pt",
            ckpt_dir / f"finetuned_{type_name}_100pct.pt",
            CHECKPOINTS_DIR / f"best_finetuned_{type_name}.pt",
            CHECKPOINTS_DIR / f"finetuned_{type_name}_100pct.pt",
        ]
        finetuned_ckpt = next((c for c in finetuned_candidates if c.exists()), finetuned_candidates[0])

        ssl_ckpt = ckpt_dir / f"best_{type_name}.pt"
        if not ssl_ckpt.exists():
            ssl_ckpt = CHECKPOINTS_DIR / f"best_{type_name}.pt"

        eff_rank_str, cosine_sim_str, cosine_sim_centered_str = "N/A", "N/A", "N/A"
        if ssl_ckpt.exists():
            ssl_model = ssl_model_cls(img_size=240, patch_size=16, in_channels=4, embed_dim=384).to(device)
            ckpt = torch.load(ssl_ckpt, map_location=device)
            if getattr(args, "encoder_source", "target") == "context":
                state_key = "context_encoder_state_dict" if "context_encoder_state_dict" in ckpt else "target_encoder_state_dict"
                encoder = ssl_model.context_encoder
            else:
                state_key = "target_encoder_state_dict" if "target_encoder_state_dict" in ckpt else "context_encoder_state_dict"
                encoder = getattr(ssl_model, "target_encoder", ssl_model.context_encoder)
            encoder.load_state_dict(ckpt[state_key])
            encoder.eval()

            all_rep_tokens = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if args.max_batches and batch_idx >= args.max_batches:
                        break
                    images = batch["image"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        tokens = encoder(images)
                    all_rep_tokens.append(tokens.detach().cpu())

            if all_rep_tokens:
                # Concatenate all representation tokens across the test set for true dataset-level rank
                cat_tokens = torch.cat(all_rep_tokens, dim=0).reshape(-1, 384)
                rep_metrics = compute_representation_collapse_metrics(cat_tokens)
                eff_rank_str = f"{rep_metrics['effective_rank']:.2f}"
                cosine_sim_str = f"{rep_metrics['avg_cosine_sim']:.4f}"
                cosine_sim_centered_str = f"{rep_metrics['avg_cosine_sim_centered']:.4f}"

        if finetuned_ckpt.exists():
            logger.info(f"Evaluating fine-tuned downstream segmentation for {name} from {finetuned_ckpt.name}...")
            ckpt = torch.load(finetuned_ckpt, map_location=device)
            dec_type = getattr(args, "decoder_type", None) or ckpt.get("decoder_type", "bottleneck")
            model = JEPASegmentationModel(
                img_size=240,
                patch_size=16,
                in_channels=4,
                embed_dim=384,
                out_channels=1,
                decoder_type=dec_type,
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            m_dict = evaluate_segmentation_model(model, test_loader, device, use_amp=use_amp, max_batches=args.max_batches)

            ft_json = metrics_dir / f"finetuned_{type_name}_metrics.json"
            if not ft_json.exists():
                ft_json = METRICS_DIR / f"finetuned_{type_name}_metrics.json"
            sec_per_epoch = "N/A"
            if ft_json.exists():
                with open(ft_json, "r") as f:
                    d = json.load(f)
                    if "epoch_duration_sec" in d:
                        sec_per_epoch = f"{pd.Series(d['epoch_duration_sec']).mean():.2f}s"

            results.append({
                "model": f"{name} (Fine-tuned, {dec_type})",
                "test_dice": f"{m_dict['test_dice_all']:.4f}",
                "test_dice_all": f"{m_dict['test_dice_all']:.4f}",
                "test_dice_tumor": f"{m_dict['test_dice_tumor']:.4f}" if not np.isnan(m_dict['test_dice_tumor']) else "N/A",
                "test_iou": f"{m_dict['test_iou']:.4f}",
                "hd95_px": f"{m_dict['hd95_2d_px']:.2f}",
                "hd95_2d_px": f"{m_dict['hd95_2d_px']:.2f}",
                "patient_3d_dice": f"{m_dict['patient_3d_dice']:.4f}" if not np.isnan(m_dict['patient_3d_dice']) else "N/A",
                "patient_3d_hd95_px": f"{m_dict['patient_3d_hd95_px']:.2f}" if not np.isnan(m_dict['patient_3d_hd95_px']) else "N/A",
                "effective_rank": eff_rank_str,
                "avg_cosine_sim": cosine_sim_str,
                "avg_cosine_sim_centered": cosine_sim_centered_str,
                "infer_ms_per_slice": f"{m_dict['ms_per_slice']:.2f} ms",
                "train_sec_per_epoch": sec_per_epoch,
            })

    summary_df = pd.DataFrame(results)
    print("\n" + "="*110)
    print("      RESEARCH EVALUATION BENCHMARK & RUNTIME TIMING SUMMARY (2D BraTS GLI)")
    print("="*110)
    print(summary_df.to_string(index=False))
    print("="*110 + "\n")

    out_csv = metrics_dir / "evaluation_benchmark_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved benchmark summary to: {out_csv}")


if __name__ == "__main__":
    main()

