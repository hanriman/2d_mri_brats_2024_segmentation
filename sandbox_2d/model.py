from monai.networks.nets import UNet
from torch import nn


def get_2d_unet(in_channels: int = 4, out_channels: int = 1) -> nn.Module:
    """
    Returns an optimized 2D ResUNet model from MONAI.
    Designed to process 4-channel stacked MRI slices [T1, T1c, T2, FLAIR]
    and output a single binary segmentation channel (logits).
    
    Parameters:
    -----------
    in_channels: int (default: 4)
        Number of input modalities.
    out_channels: int (default: 1)
        Number of output segmentation classes.
    """
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512), # Feature dimensions at each layer
        strides=(2, 2, 2, 2),            # Downsampling factor at each block
        num_res_units=2,                 # Number of residual units per layer
        norm="instance",                 # Group-wise Instance normalization (standard for medical)
        dropout=0.1,                     # Dropout probability to prevent overfitting
    )
    return model
