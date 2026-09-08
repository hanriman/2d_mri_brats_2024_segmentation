import torch

from brats_jepa.data import BraTS2DDataset, JEPAMaskingTransform, RandomModalityDropout


def test_brats_dataset_loading(dummy_dataset_dir):
    ds = BraTS2DDataset(metadata_csv=dummy_dataset_dir, split="train")
    assert len(ds) == 1
    sample = ds[0]
    assert sample["image"].shape == (4, 240, 240)
    assert sample["label"].shape == (1, 240, 240)

def test_jepa_masking_transform():
    masking = JEPAMaskingTransform(img_size=240, patch_size=16, num_target_masks=4)
    x = torch.randn(4, 240, 240)
    res = masking(x)
    assert "context_indices" in res
    assert "target_indices" in res
    assert len(res["target_indices"]) == 4
    assert res["context_indices"].dim() == 1


def test_random_modality_dropout():
    x = torch.ones(4, 4, 32, 32)

    # 1. p_drop = 0.0 -> identity
    drop_none = RandomModalityDropout(p_drop=0.0)
    out_none = drop_none(x)
    assert torch.equal(out_none, x)

    # 2. eval mode -> identity
    drop_eval = RandomModalityDropout(p_drop=0.8)
    drop_eval.eval()
    out_eval = drop_eval(x)
    assert torch.equal(out_eval, x)

    # 3. p_drop = 1.0 in train mode -> at least one modality preserved per sample
    drop_all = RandomModalityDropout(p_drop=1.0)
    drop_all.train()
    out_all = drop_all(x)
    assert out_all.shape == x.shape
    # For each sample b, at least one channel must have non-zero elements
    for b in range(4):
        active_channels = (out_all[b].sum(dim=(-1, -2)) > 0).sum()
        assert active_channels >= 1


def test_jepa_masking_spatial_uniformity():
    """Verify that stochastic context sampling avoids top-row truncation bias."""
    import numpy as np
    masking = JEPAMaskingTransform(
        img_size=240,
        patch_size=16,
        num_target_masks=4,
        num_context_patches=100,
    )
    x = torch.randn(4, 240, 240)
    _grid_h = 240 // 16  # 15
    grid_w = 240 // 16  # 15

    all_rows = []
    for _ in range(50):
        res = masking(x)
        ctx = res["context_indices"].numpy()
        rows = ctx // grid_w
        all_rows.extend(rows.tolist())
        # Verify no overlap between context and target masks
        ctx_set = set(ctx.tolist())
        for tgt in res["target_indices"]:
            tgt_set = set(tgt.numpy().tolist())
            assert ctx_set.isdisjoint(tgt_set), "Context and target indices must never overlap"

    all_rows = np.array(all_rows)
    # The grid has rows 0 to 14. Center is 7.0.
    # Without bias, mean row should be roughly centered (between 5.0 and 9.0)
    assert 5.0 <= all_rows.mean() <= 9.0, f"Mean row was {all_rows.mean():.2f}, indicating spatial bias"
    # Sampled patches must reach both top (row <= 2) and bottom (row >= 12)
    assert (all_rows <= 2).sum() > 0, "Top rows never sampled"
    assert (all_rows >= 12).sum() > 0, "Bottom rows never sampled (indicates top-row truncation bias)"


def test_select_patient_slices():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from prepare_data import select_patient_slices

    candidates = [
        {"z": i, "tumor_pixel_count": (100 if 20 <= i <= 30 else 0), "non_zero_brain": 5000}
        for i in range(50)
    ]

    # 1. Representative: 3 tumor + 2 empty
    rep = select_patient_slices(candidates, slices_per_patient=5, strategy="representative")
    assert len(rep) == 5
    tumor_count = sum(1 for s in rep if s["tumor_pixel_count"] > 0)
    empty_count = sum(1 for s in rep if s["tumor_pixel_count"] == 0)
    assert tumor_count == 3
    assert empty_count == 2

    # 2. Dense tumor: all or evenly spaced tumor slices
    dense = select_patient_slices(candidates, slices_per_patient=5, strategy="dense_tumor")
    assert len(dense) == 5
    for s in dense:
        assert s["tumor_pixel_count"] > 0

    # 3. Uniform axial: evenly spaced across entire z range
    axial = select_patient_slices(candidates, slices_per_patient=5, strategy="uniform_axial")
    assert len(axial) == 5
    assert axial[0]["z"] == 0
    assert axial[-1]["z"] == 49


def test_select_patient_slices_few_candidates():
    """Verify that select_patient_slices never duplicates slices even if candidates < needed."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from prepare_data import select_patient_slices

    # Only 2 candidates available, asking for 5
    candidates = [
        {"z": 10, "tumor_pixel_count": 50, "non_zero_brain": 2000},
        {"z": 20, "tumor_pixel_count": 0, "non_zero_brain": 2000},
    ]
    rep = select_patient_slices(candidates, slices_per_patient=5, strategy="representative")
    z_list = [s["z"] for s in rep]
    assert len(z_list) == len(set(z_list)), f"Duplicate slices detected: {z_list}"
    assert len(rep) <= 2


def test_zscore_normalize():
    """Verify foreground Z-score normalization ignores zero background."""
    from brats_jepa.data.transforms import ZScoreNormalize
    normalizer = ZScoreNormalize()
    x = torch.zeros(4, 32, 32)
    # Foreground non-zero patch
    x[:, 10:20, 10:20] = torch.randn(4, 10, 10) * 5.0 + 50.0
    out = normalizer(x)
    assert out.shape == x.shape
    # Background must stay zero
    assert (out[:, :5, :5] == 0).all()
    # Foreground mean should be ~0 and std ~1
    fg = out[:, 10:20, 10:20]
    for c in range(4):
        assert abs(fg[c].mean().item()) < 1e-4
        assert abs(fg[c].std().item() - 1.0) < 1e-2


def test_jepa_masking_contiguity():
    """Verify that context patches form a contiguous cluster with neighbor connectivity."""
    masking = JEPAMaskingTransform(img_size=240, patch_size=16, num_context_patches=96, contiguous_context=True)
    x = torch.randn(4, 240, 240)
    res = masking(x)
    ctx = res["context_indices"].numpy()
    assert len(ctx) == 96
    grid_w = 240 // 16

    # Verify context and targets are strictly disjoint
    ctx_set = set(ctx.tolist())
    for tgt in res["target_indices"]:
        tgt_set = set(tgt.numpy().tolist())
        assert ctx_set.isdisjoint(tgt_set)

    # Check that the majority of context patches share at least one 4-connected neighbor in ctx
    coords = set((idx // grid_w, idx % grid_w) for idx in ctx)
    connected_count = 0
    for r, c in coords:
        nbrs = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        if any(nbr in coords for nbr in nbrs):
            connected_count += 1
    # At least 90% of patches must be connected to an adjacent context patch
    assert connected_count / len(coords) >= 0.90


