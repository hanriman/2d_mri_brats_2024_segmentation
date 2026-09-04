import numpy as np
import torch
from brats_jepa.metrics.segmentation_metrics import (
    compute_dice_score,
    compute_hd95_single,
    compute_segmentation_metrics,
    _extract_surface_points,
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
