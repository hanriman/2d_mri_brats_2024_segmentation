from typing import Dict
import numpy as np
import torch
from scipy.spatial.distance import cdist

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """Computes Dice Similarity Coefficient (DSC) for binary segmentation predictions."""
    target_bin = (target > 0).float()
    pred_bin = (torch.sigmoid(pred) > threshold).float() if pred.dtype == torch.float32 and pred.max() > 1.0 else (pred > threshold).float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_hd95_single(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Computes 95th Percentile Hausdorff Distance (HD95) in pixels between binary predicted and target masks.
    Returns 0.0 for identical masks, or 50.0 max distance for completely empty non-overlapping masks.
    """
    p_mask = (pred_bin > 0).astype(np.uint8)
    t_mask = (target_bin > 0).astype(np.uint8)
    
    if np.array_equal(p_mask, t_mask):
        return 0.0
        
    pred_pts = np.argwhere(p_mask > 0)
    target_pts = np.argwhere(t_mask > 0)
    
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 50.0
        
    # Distance from pred points to nearest target point
    d_p2t = cdist(pred_pts, target_pts).min(axis=1)
    # Distance from target points to nearest pred point
    d_t2p = cdist(target_pts, pred_pts).min(axis=1)
    
    all_distances = np.concatenate([d_p2t, d_t2p])
    return float(np.percentile(all_distances, 95))

def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> Dict[str, float]:
    """Computes DSC, IoU, Precision, Recall, and HD95 (95th Percentile Hausdorff Distance)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > 0).float()
    
    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1.0 - target_bin)).sum().item()
    fn = ((1.0 - pred_bin) * target_bin).sum().item()
    
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    # Compute batch HD95
    hd95_vals = []
    p_np = pred_bin.detach().cpu().numpy()
    t_np = target_bin.detach().cpu().numpy()
    for b in range(p_np.shape[0]):
        hd95_vals.append(compute_hd95_single(p_np[b, 0], t_np[b, 0]))
        
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "hd95": float(np.mean(hd95_vals)),
    }
