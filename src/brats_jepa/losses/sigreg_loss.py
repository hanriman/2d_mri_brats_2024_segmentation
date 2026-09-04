
import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class EppsPulleyGaussianityTest(nn.Module):
    """
    Epps-Pulley goodness-of-fit test statistic comparing 1D empirical characteristic function
    against standard normal characteristic function phi(t) = exp(-t^2 / 2).
    Uses trapezoidal quadrature over [0, t_max] with symmetry doubling.
    Reference: LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544).
    """
    def __init__(self, t_max: float = 3.0, n_knots: int = 17):
        super().__init__()
        t = torch.linspace(0.0, t_max, n_knots, dtype=torch.float32)
        dt = t_max / (n_knots - 1)
        weights = torch.full((n_knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: [N, K] where N is number of samples, K is number of random 1D projections.
        """
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        x_t = proj.unsqueeze(-1) * t  # [N, K, Q]
        ecf_real = x_t.cos().mean(dim=0)   # [K, Q]
        ecf_imag = x_t.sin().mean(dim=0)   # [K, Q]
        err = (ecf_real - phi).square() + ecf_imag.square()  # [K, Q]
        statistic = (err @ weights) * proj.size(0)  # [K]
        return statistic.mean()


class SigRegLoss(nn.Module):
    """
    SigReg Loss: I-JEPA prediction loss + Sketched Isotropic Gaussian Regularization (SIGReg).
    Enforces that projected representations match standard isotropic Gaussian N(0, I)
    using the Cramér-Wold theorem and the Epps-Pulley test statistic.
    Reference: LeJEPA (Balestriero & LeCun, 2025, arXiv:2511.08544).
    """
    def __init__(
        self,
        loss_type: str = "l1",
        sigreg_weight: float = 1.0,
        num_projections: int = 256,
        t_max: float = 3.0,
        n_knots: int = 17,
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.sigreg_weight = sigreg_weight
        self.num_projections = num_projections
        self.ep_test = EppsPulleyGaussianityTest(t_max=t_max, n_knots=n_knots)

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)
        
        # Flatten tokens across batch and patch dimensions: [N, D]
        z = context_tokens.reshape(-1, context_tokens.shape[-1])
        N, D = z.shape
        
        # Sample M random projection directions on unit hypersphere
        A = torch.randn(D, self.num_projections, device=z.device, dtype=z.dtype)
        A = F.normalize(A, p=2, dim=0)  # [D, M]
        
        # 1D projections: [N, M]
        proj = z @ A
        
        sigreg_val = self.ep_test(proj)
        total_loss = j_loss + self.sigreg_weight * sigreg_val
        
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "sigreg_loss": sigreg_val,
        }
