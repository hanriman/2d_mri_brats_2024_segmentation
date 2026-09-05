from typing import Any

import torch
from torch import nn

from .vision_transformer import JEPAPredictor, VisionTransformerEncoder2D


class VisRegJEPA(nn.Module):
    r"""
    VisReg JEPA (VISReg): Single-Encoder Predictive Architecture with Decoupled Sliced Regularization.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Heuristic-Free Predictive Learning via Optimal Transport:
       Like SigReg, VisReg JEPA removes the requirement for an asymmetric EMA teacher network.
       Instead of hypothesis testing via characteristic functions, VisReg applies 1D Sliced
       Wasserstein Distance (SWD) against standard normal quantiles, providing a geometric
       optimal-transport formulation for representation shape matching.

    2. Decoupled Variance-Sliced Projection:
       The projector MLP (384 \to 1024 \to 128) maps latent tokens into a compact 128-dimensional
       manifold for random 1D hyper-spherical projections. The separation between encoder space
       and projector space ensures the Vision Transformer preserves rich spatial features for
       downstream dense pixel prediction (segmentation).

    References:
    -----------
    - Wu, Z., Balestriero, R., & Levine, S. (2026). "Visual Representation Learning via Regularization."
      arXiv:2606.02572 (VISReg).
    - Bonneel, N., et al. (2015). "Sliced and Radon transform Wasserstein metrics of distributions."
      Journal of Mathematical Imaging and Vision, 51(1), 22-45.
    """
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        proj_dim: int = 128,
        encoder_depth: int = 8,
        predictor_depth: int = 4,
        num_heads: int = 6,
        var_weight: float = 1.0,
        swd_weight: float = 1.0,
    ):
        super().__init__()
        self.var_weight = var_weight
        self.swd_weight = swd_weight
        
        # Single Encoder (No EMA teacher required)
        self.context_encoder = VisionTransformerEncoder2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
        )
        
        # Projector MLP: maps encoder representations (D=384) to compact space (D_proj=128)
        # for Sliced-Wasserstein Gaussian sketching, isolating representation from collapse prevention.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, proj_dim),
        )
        
        # Latent Predictor
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            pred_embed_dim=embed_dim // 2,
            num_patches=self.context_encoder.num_patches,
            depth=predictor_depth,
            num_heads=num_heads,
        )

    def update_target_encoder(self, momentum: float | None = None):
        """No-op: VisReg JEPA is heuristic-free and does not use EMA teacher updating."""

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        B = images.shape[0]
        
        # 1. Forward encoder on full image without gradients to extract target representations.
        # Switch to eval mode to disable dropout/stochastic layers in targets, then restore.
        was_training = self.context_encoder.training
        self.context_encoder.eval()
        with torch.no_grad():
            full_tokens = self.context_encoder(images)  # [B, N_patches, D]
        if was_training:
            self.context_encoder.train()
        
        # 2. Forward encoder on ONLY context patches WITH gradients (no attention leakage)
        context_tokens = self.context_encoder(images, patch_indices=context_indices)
        
        # 3. Pass context tokens through projector for VISReg Sliced-Wasserstein regularization
        projected_tokens = self.projector(context_tokens)  # [B, N_ctx, proj_dim]
            
        predictions = []
        targets = []
        
        for target_indices in target_indices_list:
            if target_indices.dim() == 1:
                target_repr = full_tokens[:, target_indices, :].detach()
            else:
                target_repr = full_tokens.gather(
                    1, target_indices.unsqueeze(-1).expand(-1, -1, full_tokens.size(-1))
                ).detach()
                
            pred_repr = self.predictor(context_tokens, context_indices, target_indices)
            predictions.append(pred_repr)
            targets.append(target_repr)
            
        return {
            "predictions": predictions,
            "targets": targets,
            "context_tokens": context_tokens,
            "projected_tokens": projected_tokens,
            "target_tokens": full_tokens,
            "var_weight": self.var_weight,
            "swd_weight": self.swd_weight,
        }
