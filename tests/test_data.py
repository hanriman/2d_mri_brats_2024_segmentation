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

