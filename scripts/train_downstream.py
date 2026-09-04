import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brats_jepa.config import (
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    ensure_directories,
    get_metadata_path,
)
from brats_jepa.data import BraTS2DDataset
from brats_jepa.losses import CombinedDiceBCELoss
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import JEPASegmentationModel
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Downstream Segmentation Fine-Tuning for Pre-Trained JEPA Variants")
    parser.add_argument("--model_type", type=str, choices=["ijepa", "sigreg_jepa", "visreg_jepa"], default="ijepa",
                        help="JEPA variant encoder to fine-tune")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze ViT encoder weights for linear/decoder probing")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv manifest file")
    parser.add_argument("--output_dir", type=str, default=None, help="Custom output directory for checkpoints and logs")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing pre-trained SSL checkpoints")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no_amp", action="store_false", dest="amp", help="Disable automatic mixed precision")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

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

    src_ckpt_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else CHECKPOINTS_DIR

    logger = get_logger(f"train_downstream_{args.model_type}", logs_dir / f"train_downstream_{args.model_type}.log")
    logger.info(f"Starting downstream segmentation training for {args.model_type.upper()} on device: {device} (Freeze Encoder: {args.freeze_encoder})")

    metadata_path = Path(args.metadata_csv).resolve() if args.metadata_csv else get_metadata_path("brats_gli_2d")
    logger.info(f"Using dataset metadata from: {metadata_path}")

    train_ds = BraTS2DDataset(metadata_csv=metadata_path, split="train")
    val_ds = BraTS2DDataset(metadata_csv=metadata_path, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Instantiate downstream model
    model = JEPASegmentationModel(
        img_size=240,
        patch_size=16,
        in_channels=4,
        embed_dim=384,
        out_channels=1,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # Load pre-trained encoder weights if available
    pretrained_ckpt = src_ckpt_dir / f"best_{args.model_type}.pt"
    if pretrained_ckpt.exists():
        logger.info(f"Loading pre-trained {args.model_type} encoder from {pretrained_ckpt}...")
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        model.load_pretrained_encoder(ckpt["context_encoder_state_dict"])
    else:
        logger.warning(f"Pre-trained checkpoint {pretrained_ckpt} not found! Initializing with random weights.")

    loss_fn = CombinedDiceBCELoss(dice_weight=1.0, bce_weight=1.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if use_amp:
        logger.info("CUDA Automatic Mixed Precision (AMP) enabled for accelerated training.")

    metric_tracker = MetricTracker()
    best_dice = 0.0
    start_total_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_hd95 = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = loss_fn(logits, labels)

                val_loss += loss.item()
                metrics = compute_segmentation_metrics(logits, labels)
                val_dice += metrics["dice"]
                val_hd95 += metrics["hd95"]

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_hd95 = val_hd95 / len(val_loader)
        epoch_duration = time.perf_counter() - epoch_start

        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val Dice: {avg_val_dice:.4f} | Val HD95: {avg_val_hd95:.2f}px | Duration: {epoch_duration:.2f}s")
        metric_tracker.update({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_hd95": avg_val_hd95,
            "epoch_duration_sec": epoch_duration,
        })

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_name = f"best_finetuned_{args.model_type}.pt"
            ckpt_path = ckpt_dir / save_name
            torch.save({
                "epoch": epoch,
                "model_type": args.model_type,
                "model_state_dict": model.state_dict(),
                "val_dice": best_dice,
                "val_hd95": avg_val_hd95,
            }, ckpt_path)
            logger.info(f"===> Saved best fine-tuned {args.model_type} checkpoint (Dice: {best_dice:.4f}) to {save_name}")

    total_duration = time.perf_counter() - start_total_time
    metric_tracker.save_json(metrics_dir / f"finetuned_{args.model_type}_metrics.json")
    logger.info(f"Downstream fine-tuning complete for {args.model_type}! Best Val Dice: {best_dice:.4f} | Total Time: {total_duration:.2f}s")

if __name__ == "__main__":
    main()
