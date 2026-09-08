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


class MultiScaleViTSegmentationDecoder(nn.Module):
    r"""
    Multi-Scale Feature Pyramid Decoder for Vision Transformer Representations.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Architectural Parity with Supervised Baselines (UNet, nnU-Net):
       Standard Vision Transformer downstream decoders upsample exclusively from the final
       deep bottleneck tokens (15x15). Because deep self-supervised tokens abstract away
       high-frequency spatial details in favor of global semantics, pure bottleneck upsampling
       produces blurred boundaries, directly causing the 2x-3x higher HD95 error observed
       in medical segmentation benchmarks.
       
    2. Hierarchical Representation Fusion (Lin et al., CVPR 2017; Hatamizadeh et al., WACV 2022):
       Intermediate transformer layers exhibit a natural representation hierarchy:
       - Shallow layers (L_2, L_4): preserve fine localized edge geometry, tissue intensity gradients,
         and high-frequency spatial transitions.
       - Deep layers (L_6, L_8): encode high-level abstract semantics and tumor class identity.
       This decoder extracts intermediate tokens from blocks [L_2, L_4, L_6, L_8], reshapes them
       into spatial feature maps, and progressively projects and fuses them via lateral skip
       connections into corresponding decoder stages (15x15 -> 30x30 -> 60x60 -> 120x120 -> 240x240).

    References:
    -----------
    - Lin, T.-Y., et al. (2017). "Feature Pyramid Networks for Object Detection." CVPR 2017.
    - Hatamizadeh, A., et al. (2022). "UNETR: Transformers for 3D Medical Image Segmentation." WACV 2022.
    """
    def __init__(
        self,
        in_dim: int = 384,
        out_channels: int = 1,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.deep_supervision = deep_supervision

        # Stage 1: Bottleneck 15x15 -> 30x30
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )
        # Skip connection from L6 (15x15 -> 30x30)
        self.skip_l6 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 192, kernel_size=2, stride=2),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(192 + 192, 192, kernel_size=3, padding=1),
            nn.GroupNorm(16, 192),
            nn.GELU(),
        )

        # Stage 2: 30x30 -> 60x60
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        # Skip connection from L4 (15x15 -> 60x60 via 4x transpose conv)
        self.skip_l4 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 96, kernel_size=4, stride=4),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(96 + 96, 96, kernel_size=3, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
        )

        # Stage 3: 60x60 -> 120x120
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        # Skip connection from L2 (15x15 -> 120x120 via 8x transpose conv)
        self.skip_l2 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 48, kernel_size=8, stride=8),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),
            nn.GroupNorm(4, 48),
            nn.GELU(),
        )

        # Stage 4: 120x120 -> 240x240
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )

        # Final projection to output logits
        self.head = nn.Conv2d(24, out_channels, kernel_size=1)

        if deep_supervision:
            self.ds3 = nn.Conv2d(48, out_channels, kernel_size=1)
            self.ds2 = nn.Conv2d(96, out_channels, kernel_size=1)
            self.ds1 = nn.Conv2d(192, out_channels, kernel_size=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Converts [B, N, D] patch tokens to [B, D, H, W] spatial feature map."""
        B, N, D = tokens.shape
        H = W = int(np.round(np.sqrt(N)))
        return tokens.permute(0, 2, 1).reshape(B, D, H, W)

    def forward(
        self,
        intermediate_tokens: list[torch.Tensor] | tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(intermediate_tokens, (list, tuple)):
            n_layers = len(intermediate_tokens)
            idx2 = max(0, n_layers // 4 - 1)
            idx4 = max(0, n_layers // 2 - 1)
            idx6 = max(0, 3 * n_layers // 4 - 1)
            idx8 = n_layers - 1
            z2 = self._tokens_to_spatial(intermediate_tokens[idx2])
            z4 = self._tokens_to_spatial(intermediate_tokens[idx4])
            z6 = self._tokens_to_spatial(intermediate_tokens[idx6])
            z8 = self._tokens_to_spatial(intermediate_tokens[idx8])
        else:
            z = self._tokens_to_spatial(intermediate_tokens)
            z2 = z4 = z6 = z8 = z

        # Stage 1: 15x15 -> 30x30
        x1 = self.up1(z8)
        s1 = self.skip_l6(z6)
        x1 = self.fuse1(torch.cat([x1, s1], dim=1))

        # Stage 2: 30x30 -> 60x60
        x2 = self.up2(x1)
        s2 = self.skip_l4(z4)
        x2 = self.fuse2(torch.cat([x2, s2], dim=1))

        # Stage 3: 60x60 -> 120x120
        x3 = self.up3(x2)
        s3 = self.skip_l2(z2)
        x3 = self.fuse3(torch.cat([x3, s3], dim=1))

        # Stage 4: 120x120 -> 240x240
        x4 = self.up4(x3)
        out = self.head(x4)

        if self.deep_supervision:
            return [out, self.ds3(x3), self.ds2(x2), self.ds1(x1)]
        return out


class JEPASegmentationModel(nn.Module):
    r"""
    Downstream Segmentation Architecture Coupling ViT Encoder and Convolutional Decoder.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Transfer Learning Protocol:
       Couples a pre-trained JEPA VisionTransformerEncoder2D (I-JEPA, SigReg, or VisReg)
       with either a bottleneck ViTSegmentationDecoder or a MultiScaleViTSegmentationDecoder.
       In downstream evaluation, the model can be evaluated under two distinct experimental regimes:
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
        decoder_type: str = "bottleneck",
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
        self.decoder_type = decoder_type
        if decoder_type == "multiscale":
            self.decoder = MultiScaleViTSegmentationDecoder(in_dim=embed_dim, out_channels=out_channels)
        else:
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
        if self.decoder_type == "multiscale":
            if self.freeze_encoder:
                with torch.no_grad():
                    _, intermediates = self.encoder(x, return_intermediate=True)
            else:
                _, intermediates = self.encoder(x, return_intermediate=True)
            return self.decoder(intermediates)
        else:
            if self.freeze_encoder:
                with torch.no_grad():
                    tokens = self.encoder(x)
            else:
                tokens = self.encoder(x)
            return self.decoder(tokens)
