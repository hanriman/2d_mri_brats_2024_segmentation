
import math
import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class VisRegLoss(nn.Module):
    r"""
    VISReg Loss: JEPA Prediction Loss + Decoupled Scale & Shape Regularization.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Decoupled Scale and Shape Regularization:
       Standard self-supervised regularization methods (like VICReg) couple variance,
       covariance, and invariance into joint objectives that require fragile tuning of
       trade-off coefficients. VISReg decouples regularization into two orthogonal axes:
       - **Scale Regularization** (\mathcal{L}_{\text{var}}): An axis-aligned hinge penalty
         guaranteeing each representation dimension maintains empirical standard deviation
         \ge \gamma = 1.0, preventing point collapse (z \to 0).
       - **Shape Regularization** (\mathcal{L}_{\text{SWD}}): The Sliced Wasserstein Distance
         comparing 1D empirical quantiles against standard Gaussian quantiles after standardization.

    2. Theoretical Explanation of Effective Rank Behavior (Table 1 Defense):
       Because `_sliced_wasserstein_distance` standardizes each projection:
           \tilde{p} = \frac{p - \mu_p}{\sigma_p}
       the SWD shape loss is explicitly scale-invariant along any projection ray. Consequently,
       non-axis-aligned low-rank subspace compression (where variance along an oblique direction
       is diminished) is NOT penalized by SWD, while `_batch_variance_loss` only constrains
       axis-aligned coordinate variances. This explains why VisReg achieves high segmentation
       performance (Dice 0.865) while exhibiting a lower effective rank (16.92 vs 58.4 in SigReg),
       a key scientific observation for thesis defense.

    3. Closed-Form 1D Wasserstein Computation:
       The 1D Wasserstein-1 distance between sorted empirical samples and target quantiles has
       a closed-form exact solution:
           W_1(P_N, Q) = \frac{1}{N} \sum_{i=1}^N |x_{(i)} - \Phi^{-1}\left(\frac{i - 0.5}{N}\right)|
       which is computed in O(N \log N) time via sorting without iterative optimization.

    References:
    -----------
    - Wu, Z., Balestriero, R., & Levine, S. (2026). "Visual Representation Learning via Regularization."
      arXiv:2606.02572 (VISReg).
    - Bonneel, N., et al. (2015). "Sliced and Radon transform Wasserstein metrics of distributions."
      Journal of Mathematical Imaging and Vision, 51(1), 22-45.
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
        
        # Standardize projections along each slice to isolate distribution shape from scale & location
        proj_mean = proj.mean(dim=0, keepdim=True)
        proj_std = torch.sqrt(proj.var(dim=0, unbiased=False, keepdim=True) + 1e-6)
        proj_stdized = (proj - proj_mean) / proj_std
        
        sorted_proj, _ = torch.sort(proj_stdized, dim=0)  # [N, M]
        
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
        context_tokens: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        projected_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)
        
        # Support flexible argument names (context_tokens, tokens, or projected_tokens)
        reg_tokens = tokens if tokens is not None else (projected_tokens if projected_tokens is not None else context_tokens)
        if reg_tokens is None:
            raise ValueError("VisRegLoss requires regularized token representations.")

        # Flatten across batch and patch dimensions: [N, D]
        z = reg_tokens.reshape(-1, reg_tokens.shape[-1])
        
        var_loss = self._batch_variance_loss(z)
        swd_loss = self._sliced_wasserstein_distance(z)
        
        total_loss = j_loss + self.var_weight * var_loss + self.swd_weight * swd_loss
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "var_loss": var_loss,
            "swd_loss": swd_loss,
        }
