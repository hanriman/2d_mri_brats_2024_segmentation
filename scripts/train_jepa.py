import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_WORKERS,
    LOGS_DIR,
    METRICS_DIR,
    ensure_directories,
    get_metadata_path,
    load_yaml_config,
    merge_config_with_args,
)
from brats_jepa.data import BraTS2DDataset, JEPAMaskingTransform, RandomModalityDropout
from brats_jepa.losses import IJEPALoss, SigRegLoss, VisRegLoss
from brats_jepa.models import IJEPA, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Supervised JEPA Pre-training (I-JEPA, SigReg JEPA, VisReg JEPA)")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model_type", type=str, choices=["ijepa", "sigreg_jepa", "visreg_jepa"], default="ijepa",
                        help="JEPA variant model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--p_drop", type=float, default=0.25, help="Random modality dropout probability during training")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv manifest file")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory for checkpoints and logs")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Number of DataLoader worker processes (default: 2 on Linux, 0 on macOS)")
    parser.add_argument("--cache_data", action="store_true", default=True,
                        help="Cache loaded slices in RAM to eliminate disk I/O bottlenecks")
    parser.add_argument("--no_cache_data", action="store_false", dest="cache_data",
                        help="Disable RAM caching of slices")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no_amp", action="store_false", dest="amp", help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch for quick local smoke testing")
    parser.add_argument("--swd_metric", type=str, choices=["mse", "l1"], default="mse",
                        help="Sliced-Wasserstein distance metric for VISReg shape loss (default: mse for W_2^2)")
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

    # Output directory setup
    if args.output_dir:
        base_out = Path(args.output_dir).resolve()
        ckpt_dir = base_out / "checkpoints"
        logs_dir = base_out / "logs"
        metrics_dir = base_out / "metrics"
    else:
        base_out = None
        ckpt_dir = CHECKPOINTS_DIR
        logs_dir = LOGS_DIR
        metrics_dir = METRICS_DIR
    ensure_directories(base_out)

    logger = get_logger(f"train_{args.model_type}", logs_dir / f"train_{args.model_type}.log")
    logger.info(f"Starting {args.model_type.upper()} pre-training on device: {device}")
    logger.info(f"DataLoader settings: num_workers={args.num_workers}, cache_in_memory={args.cache_data}")

    context_block_size = tuple(getattr(args, "context_block_size", (14, 14)))
    target_block_size = tuple(getattr(args, "target_block_size", (5, 5)))
    masking_transform = JEPAMaskingTransform(
        img_size=getattr(args, "img_size", 240),
        patch_size=getattr(args, "patch_size", 16),
        num_target_masks=getattr(args, "num_target_masks", 4),
        context_block_size=context_block_size,
        target_block_size=target_block_size,
        num_context_patches=getattr(args, "num_context_patches", 96),
    )

    metadata_path = Path(args.metadata_csv).resolve() if args.metadata_csv else get_metadata_path("brats_gli_2d")
    logger.info(f"Using dataset metadata from: {metadata_path}")
    
    train_ds = BraTS2DDataset(
        metadata_csv=metadata_path,
        split="train",
        jepa_masking=masking_transform,
        cache_in_memory=args.cache_data,
    )
    val_ds = BraTS2DDataset(
        metadata_csv=metadata_path,
        split="val",
        jepa_masking=masking_transform,
        cache_in_memory=args.cache_data,
    )

    use_cuda = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    logger.info(f"Loaded {len(train_ds)} training slices and {len(val_ds)} validation slices.")

    if args.model_type == "ijepa":
        model = IJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=384).to(device)
        loss_fn = IJEPALoss(loss_type="smooth_l1").to(device)
    elif args.model_type == "sigreg_jepa":
        model = SigRegJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=384, proj_dim=128).to(device)
        loss_fn = SigRegLoss(loss_type="smooth_l1", sigreg_weight=1.0, num_projections=256).to(device)
    elif args.model_type == "visreg_jepa":
        model = VisRegJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=384, proj_dim=128).to(device)
        swd_m = getattr(args, "swd_metric", "mse")
        scale_type = getattr(args, "scale_loss_type", "squared")
        loss_fn = VisRegLoss(
            loss_type="smooth_l1",
            var_weight=1.0,
            swd_weight=1.0,
            num_projections=256,
            swd_metric=swd_m,
            scale_loss_type=scale_type,
        ).to(device)

    # Filter optimizer to trainable parameters only (finding H1).
    # In I-JEPA, target encoder parameters have requires_grad=False and are updated strictly via EMA.
    # Passing them to AdamW would unnecessarily allocate momentum/variance state buffers (~40% VRAM)
    # and risk accidental gradient pollution if requires_grad is altered.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    warmup_epochs = min(5, max(1, args.epochs // 5)) if args.epochs >= 5 else 0
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Note on AMP device targeting: while inference autocast supports both CUDA and MPS,
    # training with GradScaler is strictly restricted to CUDA due to variable backend stability on MPS.
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        logger.info("CUDA Automatic Mixed Precision (AMP) enabled for accelerated training.")

    metric_tracker = MetricTracker()
    best_val_loss = float("inf")
    start_total_time = time.perf_counter()
    mod_drop = RandomModalityDropout(p_drop=args.p_drop)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images = mod_drop(batch["image"].to(device, non_blocking=True))
            ctx_idx = batch["context_indices"].to(device, non_blocking=True)
            tgt_idx_list = [t.to(device, non_blocking=True) for t in batch["target_indices"]]

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images, ctx_idx, tgt_idx_list)
                if args.model_type == "ijepa":
                    loss = loss_fn(outputs["predictions"], outputs["targets"])
                elif args.model_type in ("sigreg_jepa", "visreg_jepa"):
                    loss_dict = loss_fn(
                        outputs["predictions"],
                        outputs["targets"],
                        projected_tokens=outputs["projected_tokens"],
                    )
                    loss = loss_dict["loss"]

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if args.model_type == "ijepa":
                steps_per_epoch = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
                total_steps = args.epochs * steps_per_epoch
                current_step = (epoch - 1) * steps_per_epoch + batch_idx
                momentum = 1.0 - (1.0 - 0.996) * 0.5 * (1.0 + math.cos(math.pi * current_step / max(1, total_steps)))
                model.update_target_encoder(momentum=momentum)
            train_loss_sum += loss.item()

        scheduler.step()
        n_train_batches = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
        avg_train_loss = train_loss_sum / max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                images = batch["image"].to(device, non_blocking=True)
                ctx_idx = batch["context_indices"].to(device, non_blocking=True)
                tgt_idx_list = [t.to(device, non_blocking=True) for t in batch["target_indices"]]

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(images, ctx_idx, tgt_idx_list)
                    if args.model_type == "ijepa":
                        loss = loss_fn(outputs["predictions"], outputs["targets"])
                    elif args.model_type in ("sigreg_jepa", "visreg_jepa"):
                        loss = loss_fn(
                            outputs["predictions"],
                            outputs["targets"],
                            projected_tokens=outputs["projected_tokens"],
                        )["loss"]
                val_loss_sum += loss.item()

        n_val_batches = min(len(val_loader), args.max_batches) if args.max_batches else len(val_loader)
        avg_val_loss = val_loss_sum / max(1, n_val_batches)
        epoch_duration = time.perf_counter() - epoch_start

        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Duration: {epoch_duration:.2f}s")

        metric_tracker.update({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch_duration_sec": epoch_duration,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = ckpt_dir / f"min_loss_{args.model_type}.pt"
            min_loss_payload = {
                "epoch": epoch,
                "model_type": args.model_type,
                "context_encoder_state_dict": model.context_encoder.state_dict(),
                "val_loss": best_val_loss,
            }
            if hasattr(model, "target_encoder"):
                min_loss_payload["target_encoder_state_dict"] = model.target_encoder.state_dict()
            torch.save(min_loss_payload, ckpt_path)

        if epoch % 10 == 0 or epoch == args.epochs:
            periodic_path = ckpt_dir / f"{args.model_type}_epoch_{epoch:02d}.pt"
            periodic_payload = {
                "epoch": epoch,
                "model_type": args.model_type,
                "context_encoder_state_dict": model.context_encoder.state_dict(),
                "val_loss": avg_val_loss,
            }
            if hasattr(model, "target_encoder"):
                periodic_payload["target_encoder_state_dict"] = model.target_encoder.state_dict()
            torch.save(periodic_payload, periodic_path)

    # In self-supervised learning (I-JEPA, LeJEPA, VISReg), the final epoch representations
    # after full cosine annealing are fully structured and mature for downstream tasks.
    # This is the intended checkpoint for downstream fine-tuning (not min_loss, which
    # captures early-training representations that may exhibit variance collapse).
    final_ckpt = ckpt_dir / f"final_{args.model_type}.pt"
    best_ckpt = ckpt_dir / f"best_{args.model_type}.pt"
    final_payload = {
        "epoch": args.epochs,
        "model_type": args.model_type,
        "context_encoder_state_dict": model.context_encoder.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "checkpoint_type": "final_epoch",  # Explicit: this is the last epoch, not best val loss
    }
    if hasattr(model, "target_encoder"):
        final_payload["target_encoder_state_dict"] = model.target_encoder.state_dict()
    torch.save(final_payload, final_ckpt)
    torch.save(final_payload, best_ckpt)
    logger.info(f"===> Saved final pre-trained {args.model_type} encoder (Epoch {args.epochs}) to {best_ckpt.name} and {final_ckpt.name}")

    total_duration = time.perf_counter() - start_total_time
    metric_tracker.save_json(metrics_dir / f"{args.model_type}_pretrain_metrics.json")
    logger.info(f"Pre-training complete for {args.model_type}! Final Val Loss: {avg_val_loss:.5f} | Total Time: {total_duration:.2f}s ({total_duration/60:.2f} min)")

if __name__ == "__main__":
    main()

