
import math
import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class VisRegLoss(nn.Module):
    """
    VISReg Loss: I-JEPA prediction loss + Variance-Invariance-Sketching Regularization (VISReg).
    Decouples scale regularization (batch variance hinge) and shape regularization
    (Sliced-Wasserstein Distance to standard normal quantiles).
    Reference: VISReg (Wu, Balestriero, Levine, 2026, arXiv:2606.02572).
    """
    def __init__(
        self,
        loss_type: str = "l1",
        var_weight: float = 1.0,
        swd_weight: float = 1.0,
        num_projections: int = 256,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.var_weight = var_weight
        self.swd_weight = swd_weight
        self.num_projections = num_projections
        self.target_std = target_std

    def _batch_variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Scale regularization: forces feature variance across the batch to be >= target_std."""
        std_z = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return torch.mean(F.relu(self.target_std - std_z))

    def _sliced_wasserstein_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Shape regularization: 1D Sliced-Wasserstein distance against standard normal quantiles."""
        N, D = z.shape
        # Sample random projection vectors on unit hypersphere
        u = torch.randn(D, self.num_projections, device=z.device, dtype=z.dtype)
        u = F.normalize(u, p=2, dim=0)  # [D, M]
        
        # 1D slices: [N, M]
        proj = z @ u
        sorted_proj, _ = torch.sort(proj, dim=0)  # [N, M]
        
        # Analytical standard normal N(0, 1) quantiles: Phi^{-1}((i - 0.5) / N)
        probs = (torch.arange(1, N + 1, device=z.device, dtype=torch.float32) - 0.5) / N
        gaussian_quantiles = (torch.erfinv(2.0 * probs - 1.0) * math.sqrt(2.0)).to(dtype=z.dtype)  # [N]
        
        # L1 Wasserstein distance across all slices
        swd = F.l1_loss(sorted_proj, gaussian_quantiles.unsqueeze(-1).expand_as(sorted_proj))
        return swd

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)
        
        # Flatten across batch and patch dimensions: [N, D]
        z = context_tokens.reshape(-1, context_tokens.shape[-1])
        
        var_loss = self._batch_variance_loss(z)
        swd_loss = self._sliced_wasserstein_distance(z)
        
        total_loss = j_loss + self.var_weight * var_loss + self.swd_weight * swd_loss
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "var_loss": var_loss,
            "swd_loss": swd_loss,
        }
