import torch

from brats_jepa.data import BraTS2DDataset, JEPAMaskingTransform


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
