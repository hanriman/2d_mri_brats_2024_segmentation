
import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class SigRegLoss(nn.Module):
    """
    SigReg Loss: I-JEPA prediction loss + Sigmoid Variance-Covariance Representation Regularization.
    Enforces feature variance > gamma and penalizes off-diagonal cross-feature covariance.
    """
    def __init__(
        self,
        loss_type: str = "l1",
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.target_std = target_std

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge variance loss to prevent feature dimension collapse."""
        # z: [B*N, D]
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(self.target_std - std_z))
        return var_loss

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Penalizes off-diagonal covariance to decorrelate feature dimensions."""
        N, D = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov_matrix = (z_centered.T @ z_centered) / (N - 1)
        
        # Zero out diagonal elements
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        off_diag_cov = cov_matrix[~diag_mask]
        cov_loss = (off_diag_cov ** 2).sum() / D
        return cov_loss

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)
        
        # Flatten context tokens across batch and patch dimensions: [B*N_ctx, D]
        z = context_tokens.reshape(-1, context_tokens.shape[-1])
        
        v_loss = self._variance_loss(z)
        c_loss = self._covariance_loss(z)
        
        total_loss = j_loss + self.var_weight * v_loss + self.cov_weight * c_loss
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "var_loss": v_loss,
            "cov_loss": c_loss,
        }
