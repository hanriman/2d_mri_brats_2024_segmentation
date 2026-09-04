from typing import Any

import torch
from torch import nn

from .vision_transformer import JEPAPredictor, VisionTransformerEncoder2D


class VisRegJEPA(nn.Module):
    """
    VisReg JEPA (VISReg): Heuristic-free Joint-Embedding Predictive Architecture.
    Does NOT use an EMA teacher encoder. Representation collapse is prevented mathematically
    by Variance-Invariance-Sketching Regularization (VISReg).
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
        visreg_weight: float = 1.0,
        spatial_reg_weight: float = 0.5,
    ):
        super().__init__()
        self.visreg_weight = visreg_weight
        self.spatial_reg_weight = spatial_reg_weight
        
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
        """No-op: VisReg JEPA is heuristic-free and does not use EMA teacher updating."""

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        B = images.shape[0]
        
        # 1. Forward encoder on full image WITHOUT gradients to extract target representations.
        #    This prevents self-attention leakage: if context tokens were extracted from a full-image
        #    forward pass, they would already contain target patch information via global attention.
        with torch.no_grad():
            full_tokens = self.context_encoder(images)  # [B, N_patches, D]
        
        # 2. Forward encoder on ONLY context patches WITH gradients (no attention leakage)
        context_tokens = self.context_encoder(images, patch_indices=context_indices)
            
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
            "visreg_weight": self.visreg_weight,
            "spatial_reg_weight": self.spatial_reg_weight,
        }
