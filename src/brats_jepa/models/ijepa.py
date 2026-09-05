import copy
from typing import Any

import torch
from torch import nn

from .vision_transformer import JEPAPredictor, VisionTransformerEncoder2D


class IJEPA(nn.Module):
    r"""
    Image Joint-Embedding Predictive Architecture (I-JEPA) for 2D Multi-Modal MRI.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Latent Prediction vs Pixel Reconstruction:
       Traditional Masked Autoencoders (e.g., MAE) optimize pixel-level reconstruction:
           \min_\theta \| \hat{x}_{\text{recon}} - x_{\text{voxel}} \|_2^2
       In multi-modal brain MRI (T1, T1c, T2, FLAIR), a significant fraction of high-frequency
       pixel variance stems from scanner acquisition noise, Gibbs ringing, and microscopic tissue
       heterogeneity rather than clinical semantics. I-JEPA discards pixel reconstruction and
       instead predicts representations in latent space:
           \min_\theta \mathcal{D}(\hat{s}_y, s_y)
       forcing the model to ignore imperceptible voxel noise and encode high-level anatomical
       structure and tumor topology.

    2. EMA Momentum Teacher & Non-Collapsing Dynamics:
       To prevent the trivial collapse solution (where the encoder outputs constant vectors),
       the target encoder parameters \theta_t are updated via Exponential Moving Average (EMA):
           \theta_t \leftarrow m \theta_t + (1 - m) \theta_c, \quad m = 0.996
       The stop-gradient on the target encoder, combined with momentum temporal smoothing,
       acts as an implicit contrastive constraint that propels representation expansion
       without requiring negative sample pairs (Grill et al., 2020; Assran et al., 2023).

    3. Teacher Evaluation Mode Invariance:
       During training, `self.target_encoder.eval()` is strictly maintained to ensure that
       stochastic dropout and non-deterministic layers are deactivated, providing completely
       consistent targets for the predictor.

    References:
    -----------
    - Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., &
      Ballas, N. (2023). "Self-Supervised Learning from Images with a Joint-Embedding
      Predictive Architecture." IEEE/CVF CVPR 2023, pp. 15619-15629.
    - Grill, J. B., et al. (2020). "Bootstrap Your Own Latent - A New Approach to Self-Supervised
      Learning." NeurIPS 2020.
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

    def train(self, mode: bool = True):
        """Override to ensure EMA teacher target encoder always stays in eval mode."""
        super().train(mode)
        self.target_encoder.eval()
        return self

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
