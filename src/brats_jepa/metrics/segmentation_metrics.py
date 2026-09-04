from typing import Dict
import numpy as np
import torch
from scipy.spatial.distance import cdist

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5, from_logits: bool = True) -> float:
    """Computes Dice Similarity Coefficient (DSC) for binary segmentation predictions."""
    target_bin = (target > 0).float()
    if from_logits:
        pred_bin = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_hd95_single(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Computes 95th Percentile Hausdorff Distance (HD95) in pixels between binary predicted and target masks.
    Returns 0.0 for identical masks. When only one mask is empty (complete miss or hallucination),
    returns the image diagonal as the maximum possible distance penalty.
    """
    p_mask = (pred_bin > 0).astype(np.uint8)
    t_mask = (target_bin > 0).astype(np.uint8)
    
    if np.array_equal(p_mask, t_mask):
        return 0.0
        
    pred_pts = np.argwhere(p_mask > 0)
    target_pts = np.argwhere(t_mask > 0)
    
    if len(pred_pts) == 0 or len(target_pts) == 0:
        # One mask is empty while the other is not — this is a complete failure.
        # Return the image diagonal as the maximum possible distance.
        h, w = pred_bin.shape[-2], pred_bin.shape[-1]
        return float(np.sqrt(h**2 + w**2))
        
    # Distance from pred points to nearest target point
    d_p2t = cdist(pred_pts, target_pts).min(axis=1)
    # Distance from target points to nearest pred point
    d_t2p = cdist(target_pts, pred_pts).min(axis=1)
    
    all_distances = np.concatenate([d_p2t, d_t2p])
    return float(np.percentile(all_distances, 95))

def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> Dict[str, float]:
    """Computes DSC, IoU, Precision, Recall, and HD95 (95th Percentile Hausdorff Distance).
    Uses macro-averaging (per-sample then mean) following BraTS evaluation convention."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > 0).float()
    
    # Macro-averaging: compute per-sample metrics then average (BraTS convention).
    # Micro-averaging (summing TP/FP/FN across batch) lets large tumors dominate the score.
    dice_vals = []
    iou_vals = []
    precision_vals = []
    recall_vals = []
    hd95_vals = []
    
    p_np = pred_bin.detach().cpu().numpy()
    t_np = target_bin.detach().cpu().numpy()
    
    for b in range(pred_bin.shape[0]):
        p_b = pred_bin[b]
        t_b = target_bin[b]
        
        tp = (p_b * t_b).sum().item()
        fp = (p_b * (1.0 - t_b)).sum().item()
        fn = ((1.0 - p_b) * t_b).sum().item()
        
        dice_vals.append((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))
        iou_vals.append((tp + smooth) / (tp + fp + fn + smooth))
        precision_vals.append((tp + smooth) / (tp + fp + smooth))
        recall_vals.append((tp + smooth) / (tp + fn + smooth))
        hd95_vals.append(compute_hd95_single(p_np[b, 0], t_np[b, 0]))
        
    return {
        "dice": float(np.mean(dice_vals)),
        "iou": float(np.mean(iou_vals)),
        "precision": float(np.mean(precision_vals)),
        "recall": float(np.mean(recall_vals)),
        "hd95": float(np.mean(hd95_vals)),
    }
