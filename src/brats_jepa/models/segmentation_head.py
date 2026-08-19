import numpy as np
import torch
from torch import nn

from .vision_transformer import VisionTransformerEncoder2D


class ViTSegmentationDecoder(nn.Module):
    """
    Lightweight 4-stage transpose-convolutional upsampling decoder.
    Transposes 2D ViT patch token grid [B, 225, 384] (15x15) to full 2D segmentation logits [B, out_channels, 240, 240].
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
    """
    Downstream segmentation model coupling a pre-trained JEPA VisionTransformerEncoder2D
    with a ViTSegmentationDecoder. Supports optional encoder freezing for linear/decoder probing.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            with torch.no_grad():
                tokens = self.encoder(x)
        else:
            tokens = self.encoder(x)
            
        logits = self.decoder(tokens)
        return logits
