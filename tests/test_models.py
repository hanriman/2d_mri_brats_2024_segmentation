import torch

from brats_jepa.models import (
    IJEPA,
    BraTS2DnnUNet,
    BraTS2DUNet,
    JEPASegmentationModel,
    SigRegJEPA,
    VisionTransformerEncoder2D,
    VisRegJEPA,
)


def test_unet_shapes(dummy_batch):
    images = dummy_batch["image"]
    model = BraTS2DUNet(in_channels=4, out_channels=1)
    logits = model(images)
    assert logits.shape == (2, 1, 240, 240)

def test_vit_encoder_shapes():
    x = torch.randn(2, 4, 240, 240)
    vit = VisionTransformerEncoder2D(img_size=240, patch_size=16, in_channels=4, embed_dim=128, depth=2, num_heads=4)
    out = vit(x)
    assert out.shape == (2, 225, 128)

def test_ijepa_forward():
    x = torch.randn(2, 4, 240, 240)
    ctx_idx = torch.arange(0, 100, dtype=torch.long)
    tgt_idx = [torch.arange(100, 120, dtype=torch.long)]
    
    ijepa = IJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=128, encoder_depth=2, predictor_depth=2, num_heads=4)
    out = ijepa(x, ctx_idx, tgt_idx)
    assert "predictions" in out
    assert "targets" in out
    assert out["predictions"][0].shape == (2, 20, 128)

def test_sigreg_and_visreg_forward():
    x = torch.randn(2, 4, 240, 240)
    ctx_idx = torch.arange(0, 100, dtype=torch.long)
    tgt_idx = [torch.arange(100, 120, dtype=torch.long)]
    
    sigreg = SigRegJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=128, encoder_depth=2, predictor_depth=2, num_heads=4)
    visreg = VisRegJEPA(img_size=240, patch_size=16, in_channels=4, embed_dim=128, encoder_depth=2, predictor_depth=2, num_heads=4)
    
    out_sig = sigreg(x, ctx_idx, tgt_idx)
    out_vis = visreg(x, ctx_idx, tgt_idx)
    assert out_sig["predictions"][0].shape == (2, 20, 128)
    assert out_vis["predictions"][0].shape == (2, 20, 128)

def test_jepa_segmentation_model(dummy_batch):
    images = dummy_batch["image"]
    seg_model = JEPASegmentationModel(img_size=240, patch_size=16, in_channels=4, embed_dim=128, encoder_depth=2, num_heads=4, out_channels=1)
    logits = seg_model(images)
    assert logits.shape == (2, 1, 240, 240)

def test_nnunet_forward(dummy_batch):
    images = dummy_batch["image"]
    nnunet = BraTS2DnnUNet(in_channels=4, out_channels=1, deep_supervision=True)
    nnunet.train()
    out_train = nnunet(images)
    # Train mode returns multi-head logits (Tensor 5D or List)
    assert isinstance(out_train, (torch.Tensor, list, tuple))
    
    nnunet.eval()
    out_eval = nnunet(images)
    assert out_eval.shape == (2, 1, 240, 240)


