
import torch
from monai.networks.nets import UNet
from torch import nn


class BraTS2DUNet(nn.Module):
    r"""
    2D Residual U-Net Baseline for Multi-Modal Brain Tumor Segmentation.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Standard Biomedical Segmentation Benchmark:
       The classical U-Net (Ronneberger et al., MICCAI 2015) is the primary fully-supervised
       baseline in biomedical imaging. It features an encoder contracting path and a symmetrical
       decoder expanding path coupled via horizontal skip connections.

    2. Skip Connections for Spatial Precision:
       In brain tumor segmentation, deep encoders achieve high-level semantic lesion identification
       at the expense of spatial resolution (downsampling to 15x15). Skip connections copy high-resolution
       feature maps directly to corresponding decoder stages, restoring precise tumor boundary
       delineation and fine margin geometry.

    3. Residual Units & Instance Normalization:
       - **Residual Blocks** (num_res_units=2): Additive shortcut connections prevent gradient
         diminution during backpropagation across deep feature hierarchies (He et al., 2016).
       - **Instance Normalization** (`norm="instance"`): Unlike Batch Normalization, which computes
         statistics across the batch dimension and suffers when mini-batch size is small (B <= 8),
         Instance Normalization standardizes each 2D MRI slice independently. This removes inter-patient
         intensity bias and scanner-specific gain variations without batch-dependency artifacts
         (Ulyanov et al., 2016).

    References:
    -----------
    - Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for
      Biomedical Image Segmentation." MICCAI 2015, pp. 234-241.
    - Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). "Instance Normalization: The Missing
      Ingredient for Fast Stylization." arXiv:1607.08022.
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
