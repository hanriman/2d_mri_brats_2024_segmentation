import torch

from brats_jepa.models import (
    IJEPA,
    BraTS2DnnUNet,
    BraTS2DUNet,
    JEPAPredictor,
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


def test_jepa_predictor_heterogeneous_shapes():
    B, N_ctx, N_tgt, D = 3, 50, 15, 128
    predictor = JEPAPredictor(num_patches=225, embed_dim=D, pred_embed_dim=D, depth=2, num_heads=4)
    context_tokens = torch.randn(B, N_ctx, D)

    # Case 1: 1D context, 1D target
    ctx_1d = torch.arange(0, N_ctx, dtype=torch.long)
    tgt_1d = torch.arange(N_ctx, N_ctx + N_tgt, dtype=torch.long)
    out1 = predictor(context_tokens, ctx_1d, tgt_1d)
    assert out1.shape == (B, N_tgt, D)

    # Case 2: 2D context (per-sample), 1D target (shared)
    ctx_2d = torch.randint(0, 225, (B, N_ctx), dtype=torch.long)
    out2 = predictor(context_tokens, ctx_2d, tgt_1d)
    assert out2.shape == (B, N_tgt, D)

    # Case 3: 2D context (per-sample), 2D target (per-sample)
    tgt_2d = torch.randint(0, 225, (B, N_tgt), dtype=torch.long)
    out3 = predictor(context_tokens, ctx_2d, tgt_2d)
    assert out3.shape == (B, N_tgt, D)


def test_visreg_jepa_projected_tokens():
    x = torch.randn(2, 4, 240, 240)
    ctx_idx = torch.arange(0, 80, dtype=torch.long)
    tgt_idx = [torch.arange(80, 100, dtype=torch.long)]

    visreg = VisRegJEPA(
        img_size=240,
        patch_size=16,
        in_channels=4,
        embed_dim=128,
        encoder_depth=2,
        predictor_depth=2,
        num_heads=4,
        proj_dim=64,
    )
    out = visreg(x, ctx_idx, tgt_idx)
    assert "projected_tokens" in out
    assert out["projected_tokens"].shape == (2, 80, 64)



