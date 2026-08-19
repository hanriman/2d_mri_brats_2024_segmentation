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
    loss_fn = SigRegLoss()
    res = loss_fn(preds, tgts, ctx_tokens)
    assert "loss" in res
    assert "jepa_loss" in res
    assert "var_loss" in res
    assert "cov_loss" in res
    res["loss"].backward()

def test_visreg_loss():
    preds = [torch.randn(2, 20, 128, requires_grad=True)]
    tgts = [torch.randn(2, 20, 128)]
    ctx_tokens = torch.randn(2, 100, 128, requires_grad=True)
    loss_fn = VisRegLoss()
    res = loss_fn(preds, tgts, ctx_tokens)
    assert "loss" in res
    assert "visreg_loss" in res
    res["loss"].backward()
