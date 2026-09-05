import copy
from typing import Any

import torch
from torch import nn

from .vision_transformer import JEPAPredictor, VisionTransformerEncoder2D


class IJEPA(nn.Module):
    """
    Image Joint-Embedding Predictive Architecture (I-JEPA) for 2D BraTS MRI.
    Predicts target patch representations from context patches in latent space.
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
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        
        # Online Context Encoder
        self.context_encoder = VisionTransformerEncoder2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
        )
        
        # Target Encoder (Teacher - EMA updated)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # Predictor
        self.predictor = JEPAPredictor(
            embed_dim=embed_dim,
            pred_embed_dim=embed_dim // 2,
            num_patches=self.context_encoder.num_patches,
            depth=predictor_depth,
            num_heads=num_heads,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None):
        """Exponential Moving Average (EMA) update of Target Encoder weights."""
        m = momentum if momentum is not None else self.ema_decay
        for param_c, param_t in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data.mul_(m).add_((1.0 - m) * param_c.data)

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        """
        images: [B, C, H, W]
        context_indices: [B, N_ctx]
        target_indices_list: List of [B, N_tgt] target mask index tensors
        """
        B = images.shape[0]
        
        # 1. Forward online context encoder
        context_tokens = self.context_encoder(images, patch_indices=context_indices)
        
        # 2. Forward target encoder without gradients
        with torch.no_grad():
            full_target_tokens = self.target_encoder(images)  # [B, N_patches, D]
            
        predictions = []
        targets = []
        
        for target_indices in target_indices_list:
            # Extract target representations from target encoder
            if target_indices.dim() == 1:
                target_repr = full_target_tokens[:, target_indices, :]
            else:
                target_repr = full_target_tokens.gather(
                    1, target_indices.unsqueeze(-1).expand(-1, -1, full_target_tokens.size(-1))
                )
                
            # Predict target representations from context tokens
            pred_repr = self.predictor(context_tokens, context_indices, target_indices)
            
            predictions.append(pred_repr)
            targets.append(target_repr)
            
        return {
            "predictions": predictions,
            "targets": targets,
            "context_tokens": context_tokens,
            "target_tokens": full_target_tokens,
        }
