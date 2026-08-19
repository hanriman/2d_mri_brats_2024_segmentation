from typing import Any

import torch
from torch import nn

from .vision_transformer import JEPAPredictor, VisionTransformerEncoder2D


class SigRegJEPA(nn.Module):
    """
    SigReg JEPA (LeJEPA): Heuristic-free Joint-Embedding Predictive Architecture.
    Does NOT use an EMA teacher encoder. Representation collapse is prevented mathematically
    by Sketched Isotropic Gaussian / Variance-Covariance Regularization (SIGReg).
    """
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        encoder_depth: int = 8,
        predictor_depth: int = 4,
        num_heads: int = 6,
        sigreg_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
    ):
        super().__init__()
        self.sigreg_weight = sigreg_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        
        # Single Encoder (No EMA teacher required)
        self.context_encoder = VisionTransformerEncoder2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
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
        """No-op: SigReg JEPA is heuristic-free and does not use EMA teacher updating."""

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        B = images.shape[0]
        
        # 1. Forward single encoder for full image
        full_tokens = self.context_encoder(images)  # [B, N_patches, D]
        
        # 2. Extract context tokens
        if context_indices.dim() == 1:
            context_tokens = full_tokens[:, context_indices, :]
        else:
            context_tokens = torch.stack([full_tokens[b, context_indices[b], :] for b in range(B)], dim=0)
            
        predictions = []
        targets = []
        
        for target_indices in target_indices_list:
            if target_indices.dim() == 1:
                target_repr = full_tokens[:, target_indices, :].detach()
            else:
                target_repr = torch.stack([full_tokens[b, target_indices[b], :] for b in range(B)], dim=0).detach()
                
            pred_repr = self.predictor(context_tokens, context_indices, target_indices)
            predictions.append(pred_repr)
            targets.append(target_repr)
            
        return {
            "predictions": predictions,
            "targets": targets,
            "context_tokens": context_tokens,
            "target_tokens": full_tokens,
            "sigreg_weight": self.sigreg_weight,
            "var_weight": self.var_weight,
            "cov_weight": self.cov_weight,
        }
