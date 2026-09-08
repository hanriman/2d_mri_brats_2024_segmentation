import numpy as np
import torch

from brats_jepa.metrics.segmentation_metrics import (
    _extract_surface_points,
    compute_dice_score,
    compute_hd95_single,
    compute_segmentation_metrics,
)


def test_surface_point_extraction():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    pts = _extract_surface_points(mask)
    # The surface points should be fewer than the total interior points (100)
    assert len(pts) > 0
    assert len(pts) < 100


def test_hd95_identical_and_empty():
    m1 = np.zeros((50, 50))
    m1[10:20, 10:20] = 1
    # Identical
    assert compute_hd95_single(m1, m1) == 0.0

    # One empty -> returns diagonal sqrt(50^2 + 50^2) = 50 * sqrt(2) ~ 70.71
    m2 = np.zeros((50, 50))
    assert abs(compute_hd95_single(m1, m2) - float(np.sqrt(50**2 + 50**2))) < 1e-3


def test_segmentation_metrics():
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    metrics = compute_segmentation_metrics(pred, target)
    assert "dice" in metrics
    assert "iou" in metrics
    assert "hd95" in metrics
    assert 0.0 <= metrics["dice"] <= 1.0
    assert metrics["hd95"] >= 0.0


def test_all_zero_prediction_on_tumor_slice():
    # Slice has tumor (target non-empty), pred is all zeros (logits < -10)
    target = torch.zeros(1, 1, 64, 64)
    target[:, :, 10:20, 10:20] = 1.0
    pred = torch.full_like(target, -10.0)  # sigmoid(-10) ~ 0

    metrics = compute_segmentation_metrics(pred, target)
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["dice_tumor_only"] == 0.0
    assert metrics["has_tumor_per_sample"] == [True]


def test_all_zero_prediction_on_empty_slice():
    # Slice has NO tumor, pred is all zeros (correct rejection)
    target = torch.zeros(1, 1, 64, 64)
    pred = torch.full_like(target, -10.0)

    metrics = compute_segmentation_metrics(pred, target)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["hd95"] == 0.0
    assert metrics["has_tumor_per_sample"] == [False]


def test_false_alarm_prediction_on_empty_slice():
    # Slice has NO tumor, but model predicts tumor (FP > 0)
    target = torch.zeros(1, 1, 64, 64)
    pred = torch.full_like(target, -10.0)
    pred[:, :, 10:20, 10:20] = 10.0  # sigmoid(10) ~ 1

    metrics = compute_segmentation_metrics(pred, target)
    assert metrics["dice"] == 0.0
    assert metrics["iou"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["hd95"] > 0.0
    assert metrics["has_tumor_per_sample"] == [False]


def test_compute_dice_score_edge_cases():
    target = torch.zeros(1, 1, 32, 32)
    # Both empty
    pred_empty = torch.full_like(target, -10.0)
    assert compute_dice_score(pred_empty, target) == 1.0

    # Target empty, pred non-empty
    pred_non_empty = torch.full_like(target, 10.0)
    assert compute_dice_score(pred_non_empty, target) == 0.0

    # Target non-empty, pred empty
    target[0, 0, 5:10, 5:10] = 1.0
    assert compute_dice_score(pred_empty, target) == 0.0

    # Both identical non-empty
    assert compute_dice_score(pred_non_empty, torch.ones_like(target)) == 1.0


def test_hd95_3d():
    from brats_jepa.metrics.segmentation_metrics import compute_hd95_3d

    v1 = np.zeros((10, 40, 40), dtype=bool)
    v2 = np.zeros((10, 40, 40), dtype=bool)

    # Both empty
    assert compute_hd95_3d(v1, v2) == 0.0

    # One empty
    v1[2:6, 10:20, 10:20] = True
    diag = float(np.sqrt(10**2 + 40**2 + 40**2))
    assert abs(compute_hd95_3d(v1, v2) - diag) < 1e-3

    # Identical non-empty
    assert compute_hd95_3d(v1, v1) == 0.0

    # Offset by 2 voxels
    v2[2:6, 12:22, 10:20] = True
    hd = compute_hd95_3d(v1, v2)
    assert 1.0 <= hd <= 3.0


def test_patient_volume_metrics():
    from brats_jepa.metrics.segmentation_metrics import compute_patient_volume_metrics

    # 2 patients: patient_A has 2 tumor slices and 1 empty slice.
    # patient_B has 1 tumor slice and 2 empty slices.
    patient_ids = ["patient_A", "patient_A", "patient_A", "patient_B", "patient_B", "patient_B"]
    slice_indices = [0, 1, 2, 0, 1, 2]

    # Slice 0 (A): tumor, pred perfect
    t0 = torch.zeros(1, 1, 32, 32)
    t0[:, :, 5:15, 5:15] = 1.0
    p0 = torch.full_like(t0, -10.0)
    p0[:, :, 5:15, 5:15] = 10.0
    # Slice 1 (A): tumor, pred perfect
    t1 = torch.zeros(1, 1, 32, 32)
    t1[:, :, 10:20, 10:20] = 1.0
    p1 = torch.full_like(t1, -10.0)
    p1[:, :, 10:20, 10:20] = 10.0
    # Slice 2 (A): empty, pred empty
    t2 = torch.zeros(1, 1, 32, 32)
    p2 = torch.full_like(t2, -10.0)

    # Patient B: all zero prediction while slice 0 has tumor (collapsed model)
    t3 = torch.zeros(1, 1, 32, 32)
    t3[:, :, 8:18, 8:18] = 1.0
    p3 = torch.full_like(t3, -10.0)
    t4 = torch.zeros(1, 1, 32, 32)
    p4 = torch.full_like(t4, -10.0)
    t5 = torch.zeros(1, 1, 32, 32)
    p5 = torch.full_like(t5, -10.0)

    slice_preds = [p0, p1, p2, p3, p4, p5]
    slice_targets = [t0, t1, t2, t3, t4, t5]

    vol_metrics = compute_patient_volume_metrics(
        slice_preds, slice_targets, patient_ids, slice_indices=slice_indices
    )

    assert "dice_3d" in vol_metrics
    assert "per_patient" in vol_metrics
    assert vol_metrics["num_patients"] == 2

    # Patient A should have perfect 3D Dice = 1.0
    assert vol_metrics["per_patient"]["patient_A"]["dice"] == 1.0
    assert vol_metrics["per_patient"]["patient_A"]["precision"] == 1.0
    assert vol_metrics["per_patient"]["patient_A"]["recall"] == 1.0
    assert vol_metrics["per_patient"]["patient_A"]["hd95"] == 0.0

    # Patient B had complete background collapse on a volume containing tumor -> 3D Dice must be 0.0!
    assert vol_metrics["per_patient"]["patient_B"]["dice"] == 0.0
    assert vol_metrics["per_patient"]["patient_B"]["iou"] == 0.0
    assert vol_metrics["per_patient"]["patient_B"]["precision"] == 0.0
    assert vol_metrics["per_patient"]["patient_B"]["recall"] == 0.0
    assert vol_metrics["per_patient"]["patient_B"]["hd95"] > 0.0

    # Mean 3D Dice across the two patients should be 0.5
    assert abs(vol_metrics["dice_3d"] - 0.5) < 1e-5

    # Verify backward compatibility aliases
    assert "patient_3d_dice_mean" in vol_metrics
    assert "patient_3d_iou_mean" in vol_metrics
    assert "patient_3d_precision_mean" in vol_metrics
    assert "patient_3d_recall_mean" in vol_metrics
    assert "patient_3d_hd95_mean" in vol_metrics
    assert vol_metrics["patient_3d_dice_mean"] == vol_metrics["dice_3d"]
    assert vol_metrics["patient_3d_hd95_mean"] == vol_metrics["hd95_3d"]


def test_sparse_slices_patient_volume_metrics():
    """Verify that discontinuous sparse slices compute accurate 3D HD95 using 2D boundary extraction."""
    from brats_jepa.metrics.segmentation_metrics import compute_patient_volume_metrics

    # Patient with sparse slices at z = [10, 30, 50, 70, 90]
    patient_ids = ["patient_sparse"] * 5
    slice_indices = [10, 30, 50, 70, 90]

    # Ground truth: small tumor square in slice 30 and 50
    t_list = []
    p_list = []
    for z in slice_indices:
        t = torch.zeros(1, 1, 40, 40)
        p = torch.full((1, 1, 40, 40), -10.0)
        if z in (30, 50):
            t[:, :, 15:25, 15:25] = 1.0
            p[:, :, 15:25, 15:25] = 10.0
        t_list.append(t)
        p_list.append(p)

    # Identical predictions -> HD95 must be 0.0
    res_ident = compute_patient_volume_metrics(p_list, t_list, patient_ids, slice_indices=slice_indices)
    assert res_ident["dice_3d"] == 1.0
    assert res_ident["hd95_3d"] == 0.0

    # Perturbed prediction: shift tumor by 2 pixels along x in slice 30
    p_shift_list = []
    for z in slice_indices:
        p = torch.full((1, 1, 40, 40), -10.0)
        if z == 30:
            p[:, :, 15:25, 17:27] = 10.0  # shifted by 2 px
        elif z == 50:
            p[:, :, 15:25, 15:25] = 10.0
        p_shift_list.append(p)

    res_shift = compute_patient_volume_metrics(p_shift_list, t_list, patient_ids, slice_indices=slice_indices)
    assert 0.0 < res_shift["dice_3d"] < 1.0
    assert 1.0 <= res_shift["hd95_3d"] <= 3.0


def test_hd95_anisotropic_voxel_spacing():
    """Verify that physical voxel spacing scales 3D Euclidean distances accurately."""
    from brats_jepa.metrics.segmentation_metrics import compute_hd95_3d

    # Volume with single point shifted along z by 2 voxels
    v1 = np.zeros((10, 20, 20), dtype=bool)
    v2 = np.zeros((10, 20, 20), dtype=bool)
    v1[2, 10, 10] = True
    v2[4, 10, 10] = True  # delta_z = 2 voxels

    # Isotropic: 2.0
    hd_iso = compute_hd95_3d(v1, v2, voxel_spacing=(1.0, 1.0, 1.0))
    assert abs(hd_iso - 2.0) < 1e-4

    # Anisotropic: 5mm slice thickness -> 2 * 5.0 = 10.0 mm
    hd_aniso = compute_hd95_3d(v1, v2, voxel_spacing=(5.0, 1.0, 1.0))
    assert abs(hd_aniso - 10.0) < 1e-4


