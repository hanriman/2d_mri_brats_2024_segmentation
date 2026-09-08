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
    loss_fn = VisRegLoss(center_weight=1.0, scale_weight=1.0, shape_weight=1.0)
    res = loss_fn(preds, tgts, ctx_tokens)
    assert "loss" in res
    assert "jepa_loss" in res
    assert "center_loss" in res
    assert "scale_loss" in res
    assert "shape_loss" in res
    assert "var_loss" in res  # backward compatibility alias
    assert "swd_loss" in res  # backward compatibility alias
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
    assert loss.item() == 0.0
    # The returned loss should be disconnected from the target computation graph.
    # Targets come from the EMA teacher and must NEVER receive meaningful gradients.
    # Verify the loss doesn't depend on target values.
    loss.backward()
    if tgt.grad is not None:
        assert torch.all(tgt.grad == 0), "Targets should not receive non-zero gradients from the empty fallback"

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
    loss_fn = VisRegLoss(center_weight=1.0, scale_weight=1.0, shape_weight=1.0, num_projections=128)
    preds = [torch.randn(4, 20, 128)]
    tgts = [torch.randn(4, 20, 128)]

    # 1. Standard Gaussian tokens (mean=0.0, std=1.0):
    # center_loss, scale_loss, and shape_loss should all be very small.
    ctx_standard = torch.randn(4, 200, 128, requires_grad=True)
    res_std = loss_fn(preds, tgts, ctx_standard)
    assert res_std["center_loss"].item() < 0.05
    assert res_std["scale_loss"].item() < 0.05
    assert res_std["shape_loss"].item() < 0.10
    res_std["loss"].backward()
    assert ctx_standard.grad is not None

    # 2. Shifted-center tokens (mean=3.0, std=1.0):
    # center_loss triggers (3.0^2 = 9.0), while scale and shape remain low.
    ctx_shifted = (torch.randn(4, 200, 128) + 3.0).detach().requires_grad_(True)
    res_shifted = loss_fn(preds, tgts, ctx_shifted)
    assert res_shifted["center_loss"].item() > 8.0
    assert res_shifted["scale_loss"].item() < 0.05
    assert res_shifted["shape_loss"].item() < 0.10

    # 3. Scaled-variance tokens (mean=0.0, std=2.0):
    # scale_loss triggers ((1-2)^2 = 1.0), center remains near zero.
    ctx_scaled = (torch.randn(4, 200, 128) * 2.0).detach().requires_grad_(True)
    res_scaled = loss_fn(preds, tgts, ctx_scaled)
    assert res_scaled["center_loss"].item() < 0.05
    assert res_scaled["scale_loss"].item() > 0.80
    assert res_scaled["shape_loss"].item() < 0.10


def test_visreg_low_rank_collapse():
    """Verify that 1D subspace / low-rank collapse is strongly penalized by VisReg SWD."""
    torch.manual_seed(42)
    loss_fn = VisRegLoss(center_weight=1.0, scale_weight=1.0, shape_weight=1.0, num_projections=128)
    preds = [torch.randn(4, 20, 128)]
    tgts = [torch.randn(4, 20, 128)]

    # 1. Full-rank isotropic Gaussian tokens
    ctx_full_rank = torch.randn(4, 200, 128)
    res_full = loss_fn(preds, tgts, ctx_full_rank)

    # 2. Rank-1 collapsed tokens: z = u @ v.T where u is 1D scalar and v is fixed direction
    u = torch.randn(800, 1)
    v = torch.randn(1, 128)
    v = v / v.norm()
    z_collapsed = (u @ v).reshape(4, 200, 128)
    res_collapsed = loss_fn(preds, tgts, z_collapsed)

    # The rank-1 collapsed tokens must yield dramatically higher SWD shape loss than isotropic tokens
    assert res_full["shape_loss"].item() < 0.10
    assert res_collapsed["shape_loss"].item() > 0.30
    assert res_collapsed["shape_loss"].item() > 3.0 * res_full["shape_loss"].item()


def test_visreg_amp_float16_stability():
    """Verify that VisRegLoss is numerically stable under float16 without inf or nan."""
    loss_fn = VisRegLoss(center_weight=1.0, scale_weight=1.0, shape_weight=1.0, num_projections=128)
    preds = [torch.randn(2, 20, 128, dtype=torch.float16, requires_grad=True)]
    tgts = [torch.randn(2, 20, 128, dtype=torch.float16)]
    ctx_tokens = torch.randn(2, 100, 128, dtype=torch.float16, requires_grad=True)

    res = loss_fn(preds, tgts, ctx_tokens)
    for key, val in res.items():
        assert not torch.isnan(val).any(), f"NaN detected in {key}"
        assert not torch.isinf(val).any(), f"Inf detected in {key}"

    res["loss"].backward()
    assert not torch.isnan(ctx_tokens.grad).any(), "NaN in gradients"
    assert not torch.isinf(ctx_tokens.grad).any(), "Inf in gradients"


