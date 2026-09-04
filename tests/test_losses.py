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
