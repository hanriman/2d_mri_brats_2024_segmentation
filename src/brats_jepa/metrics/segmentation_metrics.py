from typing import Dict
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist

def _extract_surface_points(mask_2d: np.ndarray) -> np.ndarray:
    """Extracts 2D surface boundary contour points using morphological erosion."""
    if not np.any(mask_2d):
        return np.empty((0, 2), dtype=int)
    eroded = binary_erosion(mask_2d, structure=np.ones((3, 3)))
    boundary = mask_2d ^ eroded
    pts = np.argwhere(boundary > 0)
    if len(pts) == 0:
        pts = np.argwhere(mask_2d > 0)
    return pts

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5, from_logits: bool = True) -> float:
    r"""
    Sørensen-Dice Similarity Coefficient (DSC) for Binary Segmentation.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Formulation:
           \text{DSC} = \frac{2 |P \cap T| + \epsilon}{|P| + |T| + \epsilon} = \frac{2 \cdot \text{TP} + \epsilon}{2 \cdot \text{TP} + \text{FP} + \text{FN} + \epsilon}
       Measures the volumetric overlap between predicted binary tumor mask P and ground truth T.
       Smooth term \epsilon = 10^{-5} prevents division by zero when both prediction and ground
       truth are empty (correctly classified non-tumor slices yield DSC = 1.0).

    References:
    -----------
    - Dice, L. R. (1945). "Measures of the amount of ecologic association between species."
      Ecology, 26(3), 297-302.
    - Menze, B. H., et al. (2014). "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)."
      IEEE TMI, 34(10), 1993-2024.
    """
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
    r"""
    95th Percentile Symmetric Hausdorff Distance (HD95) in Pixel Units.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Boundary Distance vs Overlap Metrics:
       While the Dice coefficient measures regional volume overlap, it is notoriously insensitive
       to fine boundary contour errors, ragged margins, or satellite lesion hallucination.
       HD95 measures spatial surface Euclidean separation:
           d_H(P, T) = \max \left\{ P_{95\%} \min_{t \in \partial T} \|p - t\|_2, \; P_{95\%} \min_{p \in \partial P} \|t - p\|_2 \right\}
       where \partial P, \partial T are morphological boundary contours. Taking the 95th percentile
       eliminates extreme distance artifacts caused by single-pixel spurious outliers.

    2. Boundary Failure Edge Cases:
       - Identical masks (both empty or exact match) -> HD95 = 0.0 px.
       - Complete miss or false alarm (one mask non-empty, other empty) -> penalized with the
         maximum spatial distance possible: the image diagonal \sqrt{H^2 + W^2} = \sqrt{240^2 + 240^2} \approx 339.41 px.

    References:
    -----------
    - Huttenlocher, D. P., Klanderman, G. A., & Rucklidge, W. J. (1993). "Comparing images using
      the Hausdorff distance." IEEE TPAMI, 15(9), 850-863.
    """
    p_mask = (pred_bin > 0).astype(bool)
    t_mask = (target_bin > 0).astype(bool)
    
    if np.array_equal(p_mask, t_mask):
        return 0.0
        
    pred_pts = _extract_surface_points(p_mask)
    target_pts = _extract_surface_points(t_mask)
    
    if len(pred_pts) == 0 or len(target_pts) == 0:
        # One mask is empty while the other is not — this is a complete failure.
        h, w = pred_bin.shape[-2], pred_bin.shape[-1]
        return float(np.sqrt(h**2 + w**2))
        
    # Distance from pred surface to nearest target surface point
    d_p2t = cdist(pred_pts, target_pts).min(axis=1)
    # Distance from target surface to nearest pred surface point
    d_t2p = cdist(target_pts, pred_pts).min(axis=1)
    
    # Standard symmetric 95th percentile Hausdorff distance
    hd95 = max(float(np.percentile(d_p2t, 95)), float(np.percentile(d_t2p, 95)))
    return hd95

def compute_segmentation_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-5) -> Dict[str, float]:
    r"""
    Full Macro-Averaged BraTS Segmentation Benchmark Suite.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Macro-Averaging Protocol:
       Computes DSC, IoU (Jaccard Index), Precision, Recall, and HD95 per individual slice,
       then computes the population mean. Micro-averaging (pooling TP, FP, FN over all slices)
       would allow large late-stage tumors (occupying ~10% of a slice) to statistically overshadow
       difficult early-stage micro-tumors (< 0.5% volume). Macro-averaging guarantees equal
       clinical weight to each diagnostic slice.

    2. Returns Per-Sample Metric Arrays:
       In addition to batch averages, returns per-sample arrays (`dice_per_sample`, etc.)
       so evaluation scripts can concatenate every single test slice across batches without
       statistical bias from uneven final batches (e.g. 1810 total test slices with batch size 8).
    """
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
        # Per-sample lists for proper global aggregation across batches
        # (avoids bias from averaging batch-means when last batch is smaller)
        "dice_per_sample": dice_vals,
        "iou_per_sample": iou_vals,
        "precision_per_sample": precision_vals,
        "recall_per_sample": recall_vals,
        "hd95_per_sample": hd95_vals,
    }
