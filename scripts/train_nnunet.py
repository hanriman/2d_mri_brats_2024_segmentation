import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brats_jepa.config import CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, ensure_directories
from brats_jepa.data import BraTS2DDataset
from brats_jepa.losses import DeepSupervisionLoss
from brats_jepa.metrics import compute_segmentation_metrics
from brats_jepa.models import BraTS2DnnUNet
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised 2D nnU-Net Segmentation Training with Deep Supervision")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_directories()
    set_seed(args.seed)
    device = get_device(args.device)
    logger = get_logger("train_nnunet", LOGS_DIR / "train_nnunet.log")
    
    logger.info(f"Training 2D nnU-Net baseline (Deep Supervision) on device: {device}")
    
    metadata_path = Path("data/processed/2d_slices/metadata.csv").resolve()
    train_ds = BraTS2DDataset(metadata_csv=metadata_path, split="train")
    val_ds = BraTS2DDataset(metadata_csv=metadata_path, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True).to(device)
    loss_fn = DeepSupervisionLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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
            logits = model(images)
            loss = loss_fn(logits, labels)
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
                
                # eval mode returns highest resolution logits [B, 1, H, W]
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
            ckpt_path = CHECKPOINTS_DIR / "best_nnunet.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": best_dice,
                "val_hd95": avg_val_hd95,
            }, ckpt_path)
            logger.info(f"===> Saved best 2D nnU-Net checkpoint (Dice: {best_dice:.4f}) to {ckpt_path.name}")
            
    total_duration = time.perf_counter() - start_total_time
    metric_tracker.save_json(METRICS_DIR / "nnunet_train_metrics.json")
    logger.info(f"nnU-Net training complete! Best Val Dice: {best_dice:.4f} | Total Time: {total_duration:.2f}s ({total_duration/60:.2f} min)")

if __name__ == "__main__":
    main()
