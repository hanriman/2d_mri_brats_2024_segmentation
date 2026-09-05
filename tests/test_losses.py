import torch

from brats_jepa.losses import CombinedDiceBCELoss, IJEPALoss, SigRegLoss, VisRegLoss


def test_dice_bce_loss():
    logits = torch.randn(2, 1, 240, 240, requires_grad=True)
    labels = torch.randint(0, 2, (2, 1, 240, 240)).float()
    loss_fn = CombinedDiceBCELoss()
    loss = loss_fn(logits, labels)
    assert loss.dim() == 0
    assert loss > 0.0
    loss.backward()
    assert logits.grad is not None

def test_ijepa_loss():
    preds = [torch.randn(2, 20, 128, requires_grad=True)]
    tgts = [torch.randn(2, 20, 128)]
    loss_fn = IJEPALoss()
    loss = loss_fn(preds, tgts)
    assert loss.dim() == 0
    assert loss >= 0.0

def test_sigreg_loss():
    preds = [torch.randn(2, 20, 128, requires_grad=True)]
    tgts = [torch.randn(2, 20, 128)]
    ctx_tokens = torch.randn(2, 100, 128, requires_grad=True)
    loss_fn = SigRegLoss(sigreg_weight=1.0)
    res = loss_fn(preds, tgts, ctx_tokens)
    assert "loss" in res
    assert "jepa_loss" in res
    assert "sigreg_loss" in res
    res["loss"].backward()
    assert preds[0].grad is not None
    assert ctx_tokens.grad is not None

def test_visreg_loss():
    preds = [torch.randn(2, 20, 128, requires_grad=True)]
    tgts = [torch.randn(2, 20, 128)]
    ctx_tokens = torch.randn(2, 100, 128, requires_grad=True)
    loss_fn = VisRegLoss(var_weight=1.0, swd_weight=1.0)
    res = loss_fn(preds, tgts, ctx_tokens)
    assert "loss" in res
    assert "jepa_loss" in res
    assert "var_loss" in res
    assert "swd_loss" in res
    res["loss"].backward()
    assert preds[0].grad is not None
    assert ctx_tokens.grad is not None

def test_epps_pulley_gaussianity():
    from brats_jepa.losses.sigreg_loss import EppsPulleyGaussianityTest
    test = EppsPulleyGaussianityTest(t_max=3.0, n_knots=17)
    
    # 1. Samples from standard normal N(0, 1)
    torch.manual_seed(42)
    gaussian_samples = torch.randn(1000, 32)
    stat_gauss = test(gaussian_samples)
    
    # 2. Fully collapsed samples (all zeros)
    collapsed_samples = torch.zeros(1000, 32)
    stat_collapsed = test(collapsed_samples)
    
    assert stat_gauss.item() >= 0.0
    assert stat_collapsed.item() > stat_gauss.item()

def test_deep_supervision_loss():
    from brats_jepa.losses import DeepSupervisionLoss
    loss_fn = DeepSupervisionLoss()
    logits = [torch.randn(2, 1, 240, 240, requires_grad=True),
              torch.randn(2, 1, 120, 120, requires_grad=True)]
    target = torch.randint(0, 2, (2, 1, 240, 240)).float()
    loss = loss_fn(logits, target)
    assert loss > 0.0
    loss.backward()
    assert logits[0].grad is not None

def test_ijepa_empty_fallback():
    loss_fn = IJEPALoss()
    tgt = torch.randn(2, 10, 128, requires_grad=True)
    loss = loss_fn([], [tgt])
    loss.backward()
    assert tgt.grad is not None

def test_sigreg_device_transfer():
    preds = [torch.randn(2, 20, 128, requires_grad=True)]
    tgts = [torch.randn(2, 20, 128)]
    ctx_tokens = torch.randn(2, 100, 128, requires_grad=True)
    loss_fn = SigRegLoss(sigreg_weight=1.0)
    res = loss_fn(preds, tgts, ctx_tokens)
    assert res["loss"] > 0.0
    res["loss"].backward()
    assert ctx_tokens.grad is not None


def test_visreg_scale_shape_decoupling():
    torch.manual_seed(42)
    loss_fn = VisRegLoss(var_weight=1.0, swd_weight=1.0, num_projections=128)
    preds = [torch.randn(4, 20, 128)]
    tgts = [torch.randn(4, 20, 128)]

    # 1. High-variance Gaussian tokens (std = 2.0):
    # Standardizing projections isolates shape so swd_loss remains near zero (< 0.10).
    # Hinge variance loss is satisfied (std >= 1.0 -> var_loss == 0.0).
    ctx_tokens_scaled = (torch.randn(4, 150, 128) * 2.0).detach().requires_grad_(True)
    res_scaled = loss_fn(preds, tgts, ctx_tokens_scaled)
    assert res_scaled["swd_loss"].item() < 0.10
    assert res_scaled["var_loss"].item() == 0.0
    res_scaled["loss"].backward()
    assert ctx_tokens_scaled.grad is not None

    # 2. Collapsed-variance Gaussian tokens (std = 0.2):
    # Shape is still normal (swd_loss < 0.10), but variance hinge triggers (1.0 - 0.2 = ~0.8 > 0.5)
    ctx_tokens_collapsed = (torch.randn(4, 150, 128) * 0.2).detach().requires_grad_(True)
    res_collapsed = loss_fn(preds, tgts, ctx_tokens_collapsed)
    assert res_collapsed["swd_loss"].item() < 0.10
    assert res_collapsed["var_loss"].item() > 0.5
    res_collapsed["loss"].backward()
    assert ctx_tokens_collapsed.grad is not None


