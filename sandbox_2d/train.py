import argparse
import os
from pathlib import Path

import torch
from dataset import BraTS2DDataset
from model import get_2d_unet
from monai.losses import DiceLoss
from torch import nn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D ResUNet on extracted BraTS glioma slices")
    parser.add_argument("--metadata_csv", type=str, default="../data/processed/2d_slices/metadata.csv",
                        help="Path to metadata.csv manifest file")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to train on ('cuda', 'mps', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training reproducibility")
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np_seed = seed % (2**32)
    import random

    import numpy as np
    np.random.seed(np_seed)
    random.seed(seed)

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    """Compute standard Dice Similarity Coefficient (DSC) for active segmentation mask."""
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Training 2D model on device: {device}")
    
    # 2. Resolve metadata path
    script_dir = Path(__file__).resolve().parent
    metadata_path = (script_dir / args.metadata_csv).resolve()
    
    # 3. Initialize Datasets & Dataloaders
    print(f"Loading datasets from manifest: {metadata_path}")
    try:
        train_ds = BraTS2DDataset(metadata_csv=str(metadata_path), split="train")
        val_ds = BraTS2DDataset(metadata_csv=str(metadata_path), split="val")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please run `prepare_data.py` first to generate the 2D slices!")
        return
        
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Loaded {len(train_ds)} train samples and {len(val_ds)} validation samples.")
    
    # 4. Instantiate Model, Loss, Optimizer, and Scheduler
    model = get_2d_unet(in_channels=4, out_channels=1).to(device)
    
    # We use a combined loss: MONAI DiceLoss + BCEWithLogitsLoss
    dice_loss_fn = DiceLoss(sigmoid=True)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 5. Mixed Precision Setup
    use_amp = device in ["cuda", "mps"]
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    
    # Create checkpoints directory within sandbox
    ckpt_dir = script_dir / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_dice = 0.0
    print("\n--- Starting 2D UNet Training Loop ---")
    
    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)  # [B, 4, H, W]
            labels = batch["label"].to(device)  # [B, 1, H, W]
            
            optimizer.zero_grad()
            
            # Forward pass under AMP Autocast
            if use_amp:
                try:
                    with torch.amp.autocast(device_type=device, enabled=True):
                        logits = model(images)
                        d_loss = dice_loss_fn(logits, labels)
                        b_loss = bce_loss_fn(logits, labels)
                        loss = d_loss + b_loss
                except Exception:
                    # Fallback to standard precision if autocast errors
                    logits = model(images)
                    d_loss = dice_loss_fn(logits, labels)
                    b_loss = bce_loss_fn(logits, labels)
                    loss = d_loss + b_loss
                    use_amp = False
            else:
                logits = model(images)
                d_loss = dice_loss_fn(logits, labels)
                b_loss = bce_loss_fn(logits, labels)
                loss = d_loss + b_loss
                
            # Backward pass
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_dice = 0.0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(images)
                d_loss = dice_loss_fn(logits, labels)
                b_loss = bce_loss_fn(logits, labels)
                loss = d_loss + b_loss
                val_loss += loss.item()
                
                # Compute Dice metrics
                probs = torch.sigmoid(logits)
                dice = dice_coefficient(probs, labels)
                val_dice += dice
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | Val Dice: {avg_val_dice:.5f}")
        
        # Checkpointing the best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            ckpt_path = ckpt_dir / "best_2d_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_val_dice,
            }, ckpt_path)
            print(f"===> Saved new best 2D checkpoint (Dice: {best_dice:.5f}) to {ckpt_path.name}")
            
    print(f"\nTraining Complete! Best Validation Dice: {best_dice:.5f}")

if __name__ == "__main__":
    main()
