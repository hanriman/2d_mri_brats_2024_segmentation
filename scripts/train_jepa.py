import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from brats_jepa.config import CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR, ensure_directories
from brats_jepa.data import BraTS2DDataset, JEPAMaskingTransform
from brats_jepa.losses import IJEPALoss, SigRegLoss, VisRegLoss
from brats_jepa.models import IJEPA, SigRegJEPA, VisRegJEPA
from brats_jepa.utils import MetricTracker, get_device, get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Supervised JEPA Pre-training (I-JEPA, SigReg JEPA, VisReg JEPA)")
    parser.add_argument("--model_type", type=str, choices=["ijepa", "sigreg_jepa", "visreg_jepa"], default="ijepa",
                        help="JEPA variant model architecture")
    parser.add_argument("--epochs", type=int, default=50, help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_directories()
    set_seed(args.seed)
    device = get_device(args.device)
    logger = get_logger(f"train_{args.model_type}", LOGS_DIR / f"train_{args.model_type}.log")
    
    logger.info(f"Starting {args.model_type.upper()} pre-training on device: {device}")
    
    masking_transform = JEPAMaskingTransform(img_size=240, patch_size=16)
    
    metadata_path = Path("data/processed/2d_slices/metadata.csv").resolve()
    train_ds = BraTS2DDataset(metadata_csv=metadata_path, split="train", jepa_masking=masking_transform)
    val_ds = BraTS2DDataset(metadata_csv=metadata_path, split="val", jepa_masking=masking_transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    logger.info(f"Loaded {len(train_ds)} training slices and {len(val_ds)} validation slices.")
    
    if args.model_type == "ijepa":
        model = IJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=384).to(device)
        loss_fn = IJEPALoss(loss_type="l1")
    elif args.model_type == "sigreg_jepa":
        model = SigRegJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=384).to(device)
        loss_fn = SigRegLoss(var_weight=1.0, cov_weight=0.04)
    elif args.model_type == "visreg_jepa":
        model = VisRegJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=384).to(device)
        loss_fn = VisRegLoss(visreg_weight=1.0, spatial_reg_weight=0.5)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    metric_tracker = MetricTracker()
    best_val_loss = float("inf")
    start_total_time = time.perf_counter()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            ctx_idx = batch["context_indices"].to(device)
            tgt_idx_list = [t.to(device) for t in batch["target_indices"]]
            
            optimizer.zero_grad()
            outputs = model(images, ctx_idx, tgt_idx_list)
            
            if args.model_type == "ijepa":
                loss = loss_fn(outputs["predictions"], outputs["targets"])
            elif args.model_type in ["sigreg_jepa", "visreg_jepa"]:
                loss_dict = loss_fn(outputs["predictions"], outputs["targets"], outputs["context_tokens"])
                loss = loss_dict["loss"]
                
            loss.backward()
            optimizer.step()
            model.update_target_encoder()
            
            train_loss_sum += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                ctx_idx = batch["context_indices"].to(device)
                tgt_idx_list = [t.to(device) for t in batch["target_indices"]]
                
                outputs = model(images, ctx_idx, tgt_idx_list)
                if args.model_type == "ijepa":
                    loss = loss_fn(outputs["predictions"], outputs["targets"])
                else:
                    loss = loss_fn(outputs["predictions"], outputs["targets"], outputs["context_tokens"])["loss"]
                val_loss_sum += loss.item()
                
        avg_val_loss = val_loss_sum / len(val_loader)
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
            ckpt_path = CHECKPOINTS_DIR / f"best_{args.model_type}.pt"
            torch.save({
                "epoch": epoch,
                "model_type": args.model_type,
                "context_encoder_state_dict": model.context_encoder.state_dict(),
                "val_loss": best_val_loss,
            }, ckpt_path)
            logger.info(f"===> Saved best {args.model_type} encoder to {ckpt_path.name}")
            
    total_duration = time.perf_counter() - start_total_time
    metric_tracker.save_json(METRICS_DIR / f"{args.model_type}_pretrain_metrics.json")
    logger.info(f"Pre-training complete for {args.model_type}! Best Val Loss: {best_val_loss:.5f} | Total Time: {total_duration:.2f}s ({total_duration/60:.2f} min)")

if __name__ == "__main__":
    main()
