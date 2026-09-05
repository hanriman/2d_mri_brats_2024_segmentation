import numpy as np
import torch
from torch import nn

from .vision_transformer import VisionTransformerEncoder2D


class ViTSegmentationDecoder(nn.Module):
    r"""
    Progressive 4-Stage Transpose-Convolutional Upsampling Decoder.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Spatial Token Reshaping & Progressive Upsampling:
       The Vision Transformer produces 1D patch tokens [B, N=225, D=384]. This decoder reshapes
       the sequence into a 2D spatial feature map of shape [B, D, 15, 15] and progressively
       doubles resolution across 4 strided transpose convolutions:
           15 \times 15 \xrightarrow{2\times} 30 \times 30 \xrightarrow{2\times} 60 \times 60
           \xrightarrow{2\times} 120 \times 120 \xrightarrow{2\times} 240 \times 240
       reconstructing full voxel-level resolution matching the original MRI scan.

    2. Group Normalization (Wu & He, ECCV 2018):
       Batch Normalization in small downstream fine-tuning batches (B <= 8) suffers from noisy
       mean and variance estimates, which destabilizes fine-tuning. GroupNorm divides channels
       into independent groups (e.g. 16, 8, 4), computing statistics along spatial and sub-channel
       dimensions per-sample, guaranteeing robust convergence across variable batch sizes.

    3. Controlled Capacity for Objective Representation Benchmarking:
       The decoder is intentionally designed to be lightweight (~0.6M parameters).
       A massive decoder (e.g., U-PerNet or Mask2Former) can compensate for poor encoder
       representations. Restricting decoder capacity ensures that downstream segmentation
       Dice scores directly reflect the semantic quality and linear separability of the
       underlying self-supervised representations (I-JEPA vs SigReg vs VisReg).

    References:
    -----------
    - Wu, Y., & He, K. (2018). "Group Normalization." ECCV 2018, pp. 3-19.
    """
    def __init__(self, in_dim: int = 384, out_channels: int = 1):
        super().__init__()
        self.in_dim = in_dim
        
        self.decoder = nn.Sequential(
            # Stage 1: 15x15 -> 30x30
            nn.ConvTranspose2d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
            # Stage 2: 30x30 -> 60x60
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            # Stage 3: 60x60 -> 120x120
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
            # Stage 4: 120x120 -> 240x240
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
            # Projection head to logits
            nn.Conv2d(24, out_channels, kernel_size=1)
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B, N_patches, D] -> [B, D, H_grid, W_grid]
        B, N, D = patch_tokens.shape
        H = W = int(np.round(np.sqrt(N)))
        x = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
        return self.decoder(x)

class JEPASegmentationModel(nn.Module):
    r"""
    Downstream Segmentation Architecture Coupling ViT Encoder and Convolutional Decoder.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Transfer Learning Protocol:
       Couples a pre-trained JEPA VisionTransformerEncoder2D (I-JEPA, SigReg, or VisReg)
       with the ViTSegmentationDecoder. In downstream evaluation, the model can be evaluated under
       two distinct experimental regimes:
       - **Full Fine-Tuning**: Both encoder and decoder weights are optimized on labeled 2D slices.
       - **Decoder Probing (`freeze_encoder=True`)**: Encoder weights are completely frozen. Only the
         lightweight decoder is trained. This tests whether the self-supervised representations
         linearly encode spatial tumor boundaries without task-specific representation restructuring.

    2. Strict Evaluation Mode for Frozen Encoder:
       When `freeze_encoder=True`, `self.train(mode)` enforces `self.encoder.eval()`.
       This guarantees that LayerNorm statistics and dropout layers within the pre-trained encoder
       remain strictly deterministic, preventing stochastic noise from corrupting frozen representations.
    """
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 16,
        in_channels: int = 4,
        embed_dim: int = 384,
        encoder_depth: int = 8,
        num_heads: int = 6,
        out_channels: int = 1,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = VisionTransformerEncoder2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
        )
        self.decoder = ViTSegmentationDecoder(in_dim=embed_dim, out_channels=out_channels)
        self.freeze_encoder = freeze_encoder
        
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def load_pretrained_encoder(self, encoder_state_dict: dict):
        """Loads pre-trained SSL JEPA encoder weights."""
        self.encoder.load_state_dict(encoder_state_dict)

    def train(self, mode: bool = True):
        """Override to keep frozen encoder in eval mode (disables dropout, freezes LayerNorm stats)."""
        super().train(mode)
        if self.freeze_encoder and mode:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                tokens = self.encoder(x)
        else:
            tokens = self.encoder(x)
            
        logits = self.decoder(tokens)
        return logits
