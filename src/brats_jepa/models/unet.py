
import torch
from monai.networks.nets import UNet
from torch import nn


class BraTS2DUNet(nn.Module):
    """
    2D Residual UNet for multi-modal brain tumor segmentation and representation fine-tuning.
    Optionally exposes encoder representations for probing tasks.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.unet = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm="instance",
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning segmentation logits [B, out_channels, H, W]."""
        return self.unet(x)
