import argparse
import time
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    DEFAULT_NUM_WORKERS,
    LOGS_DIR,
    METRICS_DIR,
    ensure_directories,
    get_metadata_path,
)
from brats_jepa.data import BraTS2DDataset, RandomModalityDropout
from brats_jepa.losses import CombinedDiceBCELoss
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import BraTS2DUNet
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised 2D ResUNet Segmentation Training")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--p_drop", type=float, default=0.0, help="Random modality dropout probability during training")
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
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit batches per epoch for quick local smoke testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # Enable cuDNN benchmark for static-sized convolutions on CUDA
    if device.type == "cuda":
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

    logger = get_logger("train_unet", logs_dir / "train_unet.log")
    logger.info(f"Training 2D ResUNet baseline on device: {device}")
    logger.info(f"DataLoader settings: num_workers={args.num_workers}, cache_in_memory={args.cache_data}")

    metadata_path = Path(args.metadata_csv).resolve() if args.metadata_csv else get_metadata_path("brats_gli_2d")
    logger.info(f"Using dataset metadata from: {metadata_path}")

    train_ds = BraTS2DDataset(metadata_csv=metadata_path, split="train", cache_in_memory=args.cache_data)
    val_ds = BraTS2DDataset(metadata_csv=metadata_path, split="val", cache_in_memory=args.cache_data)

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

    model = BraTS2DUNet(in_channels=4, out_channels=1).to(device)
    loss_fn = CombinedDiceBCELoss(dice_weight=1.0, bce_weight=1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        logger.info("CUDA Automatic Mixed Precision (AMP) enabled for accelerated training.")

    metric_tracker = MetricTracker()
    best_dice = 0.0
    start_total_time = time.perf_counter()
    mod_drop = RandomModalityDropout(p_drop=args.p_drop)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches and batch_idx >= args.max_batches:
                break
            images = mod_drop(batch["image"].to(device, non_blocking=True))
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

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

            train_loss += loss.item()

        scheduler.step()
        n_train_batches = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
        avg_train_loss = train_loss / max(1, n_train_batches)

        model.eval()
        val_loss = 0.0
        all_dice = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if args.max_batches and batch_idx >= args.max_batches:
                    break
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                val_loss += loss.item()
                metrics = compute_segmentation_metrics(logits, labels)
                all_dice.extend(metrics["dice_per_sample"])

        n_val_batches = min(len(val_loader), args.max_batches) if args.max_batches else len(val_loader)
        avg_val_loss = val_loss / max(1, n_val_batches)
        avg_val_dice = float(np.mean(all_dice)) if all_dice else 0.0
        epoch_duration = time.perf_counter() - epoch_start

        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val Dice: {avg_val_dice:.5f} | Duration: {epoch_duration:.2f}s")
        metric_tracker.update({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "epoch_duration_sec": epoch_duration,
        })

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            ckpt_path = ckpt_dir / "best_unet.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_dice,
            }, ckpt_path)
            logger.info(f"===> Saved best 2D UNet checkpoint (Dice: {best_dice:.5f}) to {ckpt_path.name}")

    total_duration = time.perf_counter() - start_total_time
    metric_tracker.save_json(metrics_dir / "unet_train_metrics.json")
    logger.info(f"UNet training complete! Best Val Dice: {best_dice:.5f} | Total Time: {total_duration:.2f}s ({total_duration/60:.2f} min)")

if __name__ == "__main__":
    main()

