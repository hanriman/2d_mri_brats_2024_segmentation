from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
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


def compute_dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 0.0,
    from_logits: bool = True,
) -> float:
    r"""
    Sørensen-Dice Similarity Coefficient (DSC) for Binary Segmentation with Zero-Division Guard.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Formulation:
           \text{DSC} = \begin{cases} 
               1.0, & \text{if } |P| + |T| = 0 \text{ (both prediction and target empty)} \\ 
               \frac{2 |P \cap T|}{|P| + |T|}, & \text{otherwise} 
           \end{cases}
       Guarded division prevents artificial epsilon score inflation on non-empty/empty slices.
       If smooth > 0 is explicitly provided, computes (2 * TP + smooth) / (2 * TP + FP + FN + smooth).

    References:
    -----------
    - Dice, L. R. (1945). "Measures of the amount of ecologic association between species." Ecology, 26(3), 297-302.
    - Menze, B. H., et al. (2014). "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE TMI, 34(10), 1993-2024.
    """
    target_bin = (target > 0).float()
    if from_logits:
        pred_bin = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target_bin).sum().item()
    union = pred_bin.sum().item() + target_bin.sum().item()
    if union == 0:
        return 1.0
    if smooth > 0.0:
        return float((2.0 * intersection + smooth) / (union + smooth))
    return float((2.0 * intersection) / union)


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


def compute_hd95_3d(pred_vol_3d: np.ndarray, target_vol_3d: np.ndarray) -> float:
    r"""
    95th Percentile Symmetric Hausdorff Distance (HD95) for 3D binary volumes [D, H, W] in voxel units.
    Uses binary erosion with 3D 26-connectivity structuring element and cKDTree for O(N log N) speed.
    """
    p_mask = (pred_vol_3d > 0).astype(bool)
    t_mask = (target_vol_3d > 0).astype(bool)

    if np.array_equal(p_mask, t_mask):
        return 0.0

    eroded_p = binary_erosion(p_mask, structure=np.ones((3, 3, 3)))
    pts_p = np.argwhere(p_mask ^ eroded_p)
    if len(pts_p) == 0:
        pts_p = np.argwhere(p_mask)

    eroded_t = binary_erosion(t_mask, structure=np.ones((3, 3, 3)))
    pts_t = np.argwhere(t_mask ^ eroded_t)
    if len(pts_t) == 0:
        pts_t = np.argwhere(t_mask)

    d, h, w = p_mask.shape
    diag = float(np.sqrt(d**2 + h**2 + w**2))

    if len(pts_p) == 0 or len(pts_t) == 0:
        return diag

    tree_t = cKDTree(pts_t)
    d_p2t, _ = tree_t.query(pts_p)
    tree_p = cKDTree(pts_p)
    d_t2p, _ = tree_p.query(pts_t)

    return float(max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95)))


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 0.0,
) -> dict[str, Any]:
    r"""
    Full Macro-Averaged BraTS Segmentation Benchmark Suite with Guarded Zero-Division Handling.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Zero-Division Handling (Powers, 2011):
       - Precision: Defined as TP / (TP + FP) when TP + FP > 0. When TP + FP == 0:
         returns 1.0 if FN == 0 (true negative slice), and 0.0 if FN > 0 (missed tumor).
       - Recall: Defined as TP / (TP + FN) when TP + FN > 0. When TP + FN == 0:
         returns 1.0 if FP == 0 (correct non-tumor rejection), and 0.0 if FP > 0 (false alarm).
       - Dice & IoU: Returns 1.0 when both prediction and ground truth are empty (2*TP + FP + FN == 0),
         and standard ratio when non-empty. This eliminates epsilon distortion on empty and missed slices.

    2. Macro-Averaging Protocol & Tumor-Only Stratification:
       Computes metrics per individual slice and reports macro-means. In addition, returns
       `dice_tumor_only` and `iou_tumor_only` strictly evaluated on ground-truth tumor slices
       (FN + TP > 0) to prevent the empty-slice baseline from masking model collapse.

    3. Returns Per-Sample Metric Arrays:
       Returns per-sample arrays (`dice_per_sample`, etc.) for global dataset aggregation.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    pred_bin = (torch.sigmoid(pred) > threshold).float()
    target_bin = (target > 0).float()

    dice_vals = []
    iou_vals = []
    precision_vals = []
    recall_vals = []
    hd95_vals = []
    has_tumor_vals = []

    p_np = pred_bin.detach().cpu().numpy()
    t_np = target_bin.detach().cpu().numpy()

    for b in range(pred_bin.shape[0]):
        p_b = pred_bin[b]
        t_b = target_bin[b]

        tp = (p_b * t_b).sum().item()
        fp = (p_b * (1.0 - t_b)).sum().item()
        fn = ((1.0 - p_b) * t_b).sum().item()

        has_tumor = bool(tp + fn > 0)
        has_tumor_vals.append(has_tumor)

        # Guarded Dice
        if 2.0 * tp + fp + fn > 0:
            if smooth > 0.0:
                dice_vals.append((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))
            else:
                dice_vals.append((2.0 * tp) / (2.0 * tp + fp + fn))
        else:
            dice_vals.append(1.0)

        # Guarded IoU
        if tp + fp + fn > 0:
            if smooth > 0.0:
                iou_vals.append((tp + smooth) / (tp + fp + fn + smooth))
            else:
                iou_vals.append(tp / (tp + fp + fn))
        else:
            iou_vals.append(1.0)

        # Guarded Precision: Powers (2011)
        if tp + fp > 0:
            if smooth > 0.0:
                precision_vals.append((tp + smooth) / (tp + fp + smooth))
            else:
                precision_vals.append(tp / (tp + fp))
        else:
            precision_vals.append(1.0 if fn == 0 else 0.0)

        # Guarded Recall: Powers (2011)
        if tp + fn > 0:
            if smooth > 0.0:
                recall_vals.append((tp + smooth) / (tp + fn + smooth))
            else:
                recall_vals.append(tp / (tp + fn))
        else:
            recall_vals.append(1.0 if fp == 0 else 0.0)

        hd95_vals.append(compute_hd95_single(p_np[b, 0], t_np[b, 0]))

    tumor_indices = [i for i, h in enumerate(has_tumor_vals) if h]
    dice_tumor = float(np.mean([dice_vals[i] for i in tumor_indices])) if tumor_indices else 1.0
    iou_tumor = float(np.mean([iou_vals[i] for i in tumor_indices])) if tumor_indices else 1.0
    prec_tumor = (
        float(np.mean([precision_vals[i] for i in tumor_indices])) if tumor_indices else 1.0
    )
    rec_tumor = float(np.mean([recall_vals[i] for i in tumor_indices])) if tumor_indices else 1.0
    hd95_tumor = float(np.mean([hd95_vals[i] for i in tumor_indices])) if tumor_indices else 0.0

    return {
        "dice": float(np.mean(dice_vals)),
        "iou": float(np.mean(iou_vals)),
        "precision": float(np.mean(precision_vals)),
        "recall": float(np.mean(recall_vals)),
        "hd95": float(np.mean(hd95_vals)),
        "dice_tumor_only": dice_tumor,
        "iou_tumor_only": iou_tumor,
        "precision_tumor_only": prec_tumor,
        "recall_tumor_only": rec_tumor,
        "hd95_tumor_only": hd95_tumor,
        # Per-sample lists for proper global aggregation across batches
        "dice_per_sample": dice_vals,
        "iou_per_sample": iou_vals,
        "precision_per_sample": precision_vals,
        "recall_per_sample": recall_vals,
        "hd95_per_sample": hd95_vals,
        "has_tumor_per_sample": has_tumor_vals,
    }


def compute_patient_volume_metrics(
    slice_preds: list[torch.Tensor] | torch.Tensor | list[np.ndarray] | np.ndarray,
    slice_targets: list[torch.Tensor] | torch.Tensor | list[np.ndarray] | np.ndarray,
    patient_ids: list[str],
    slice_indices: list[int] | None = None,
    threshold: float = 0.5,
    from_logits: bool = True,
) -> dict[str, Any]:
    r"""
    Official 3D BraTS Patient-Level Volumetric Evaluation Benchmark (Baid et al., 2021).

    Accumulates 2D slice predictions back into 3D patient volumes, computing
    volumetric Dice, IoU, Precision, Recall, and 3D HD95 per patient:
        \text{Dice}_{\text{3D}} = \frac{2 \sum_z \text{TP}_z}{\sum_z (2 \text{TP}_z + \text{FP}_z + \text{FN}_z)}

    This completely eliminates the slice-wise empty-slice metric distortion (where
    40% empty slices grant a 0.40 baseline to broken models).
    """

    def _to_binary_numpy(data: Any, is_target: bool = False) -> np.ndarray:
        if isinstance(data, list):
            if len(data) == 0:
                return np.empty((0, 240, 240), dtype=bool)
            if isinstance(data[0], torch.Tensor):
                t = torch.cat(data, dim=0) if data[0].ndim == 4 else torch.stack(data, dim=0)
                if is_target:
                    arr = (t > 0).detach().cpu().numpy()
                else:
                    arr = (
                        (torch.sigmoid(t) > threshold if from_logits else t > threshold)
                        .detach()
                        .cpu()
                        .numpy()
                    )
            else:
                arr = np.concatenate(data, axis=0) if data[0].ndim >= 3 else np.array(data)
                if is_target:
                    arr = arr > 0
                else:
                    if from_logits:
                        arr = (1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))) > threshold
                    else:
                        arr = arr > threshold
        elif isinstance(data, torch.Tensor):
            if is_target:
                arr = (data > 0).detach().cpu().numpy()
            else:
                arr = (
                    (torch.sigmoid(data) > threshold if from_logits else data > threshold)
                    .detach()
                    .cpu()
                    .numpy()
                )
        else:
            arr = np.asarray(data)
            if is_target:
                arr = arr > 0
            else:
                if from_logits:
                    arr = (1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))) > threshold
                else:
                    arr = arr > threshold

        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        return arr.astype(bool)

    p_bin = _to_binary_numpy(slice_preds, is_target=False)
    t_bin = _to_binary_numpy(slice_targets, is_target=True)

    N = len(patient_ids)
    if p_bin.shape[0] != N or t_bin.shape[0] != N:
        raise ValueError(
            f"Sample count mismatch: {p_bin.shape[0]} preds, {t_bin.shape[0]} targets, {N} patient_ids"
        )

    patient_slices: dict[str, list[int]] = {}
    for i, pid in enumerate(patient_ids):
        patient_slices.setdefault(pid, []).append(i)

    per_patient = {}
    p_dices, p_ious, p_precs, p_recs, p_hd95s = [], [], [], [], []

    for pid, s_idxs in patient_slices.items():
        sub_p = p_bin[s_idxs]  # [S, H, W]
        sub_t = t_bin[s_idxs]  # [S, H, W]

        tp = int(np.logical_and(sub_p, sub_t).sum())
        fp = int(np.logical_and(sub_p, ~sub_t).sum())
        fn = int(np.logical_and(~sub_p, sub_t).sum())

        # Volumetric 3D Dice
        if 2 * tp + fp + fn > 0:
            dice_3d = (2.0 * tp) / (2.0 * tp + fp + fn)
        else:
            dice_3d = 1.0  # Both volumes empty

        # Volumetric 3D IoU
        if tp + fp + fn > 0:
            iou_3d = float(tp) / (tp + fp + fn)
        else:
            iou_3d = 1.0

        # Volumetric 3D Precision (Powers, 2011)
        if tp + fp > 0:
            prec_3d = float(tp) / (tp + fp)
        else:
            prec_3d = 1.0 if fn == 0 else 0.0

        # Volumetric 3D Recall (Powers, 2011)
        if tp + fn > 0:
            rec_3d = float(tp) / (tp + fn)
        else:
            rec_3d = 1.0 if fp == 0 else 0.0

        # 3D HD95:
        if slice_indices is not None:
            p_z = [slice_indices[i] for i in s_idxs]
            max_z = max(p_z)
            h, w = sub_p.shape[1], sub_p.shape[2]
            vol_p = np.zeros((max_z + 1, h, w), dtype=bool)
            vol_t = np.zeros((max_z + 1, h, w), dtype=bool)
            for local_i, z in enumerate(p_z):
                vol_p[z] = sub_p[local_i]
                vol_t[z] = sub_t[local_i]
            hd95_3d = compute_hd95_3d(vol_p, vol_t)
        else:
            hd95_3d = compute_hd95_3d(sub_p, sub_t)

        per_patient[pid] = {
            "dice": dice_3d,
            "iou": iou_3d,
            "precision": prec_3d,
            "recall": rec_3d,
            "hd95": hd95_3d,
            "num_slices": len(s_idxs),
            "tumor_slices": int((sub_t.sum(axis=(1, 2)) > 0).sum()),
        }
        p_dices.append(dice_3d)
        p_ious.append(iou_3d)
        p_precs.append(prec_3d)
        p_recs.append(rec_3d)
        p_hd95s.append(hd95_3d)

    return {
        "dice_3d": float(np.mean(p_dices)),
        "iou_3d": float(np.mean(p_ious)),
        "precision_3d": float(np.mean(p_precs)),
        "recall_3d": float(np.mean(p_recs)),
        "hd95_3d": float(np.mean(p_hd95s)),
        "num_patients": len(patient_slices),
        "per_patient": per_patient,
    }
