
import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class VisRegLoss(nn.Module):
    """
    VisReg Loss: I-JEPA prediction loss + Visual Spatial Feature Regularization.
    Enforces spatial patch feature contrast and prevents spatial over-smoothing in brain MRI representations.
    """
    def __init__(
        self,
        loss_type: str = "l1",
        visreg_weight: float = 1.0,
        spatial_reg_weight: float = 0.5,
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.visreg_weight = visreg_weight
        self.spatial_reg_weight = spatial_reg_weight

    def _spatial_feature_variance(self, context_tokens: torch.Tensor) -> torch.Tensor:
        """Computes variance of representations across spatial patch dimensions within each image."""
        # context_tokens: [B, N_ctx, D]
        # Use unbiased=False to avoid NaN when N_ctx=1
        patch_std = torch.sqrt(context_tokens.var(dim=1, unbiased=False) + 1e-4)  # [B, D]
        return torch.mean(F.relu(1.0 - patch_std))

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)
        s_loss = self._spatial_feature_variance(context_tokens)
        
        total_loss = j_loss + self.visreg_weight * self.spatial_reg_weight * s_loss
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "visreg_loss": s_loss,
        }
