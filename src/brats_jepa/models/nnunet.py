import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

class BraTS2DnnUNet(nn.Module):
    """
    2D nnU-Net architecture with Deep Supervision heads (Isensee et al., Nature Methods 2021).
    Features residual encoder blocks, LeakyReLU activations, Instance Normalization,
    and intermediate multi-scale decoder supervision heads.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 1, deep_supervision: bool = True):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        # 5-stage UNet structure: 240x240 -> 120x120 -> 60x60 -> 30x30 -> 15x15
        self.nnunet = DynUNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[[3, 3]] * 5,
            strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]],
            upsample_kernel_size=[[2, 2], [2, 2], [2, 2], [2, 2]],
            filters=[32, 64, 128, 256, 512],
            dropout=0.1,
            norm_name="instance",
            act_name="leakyrelu",
            deep_supervision=deep_supervision,
            deep_supr_num=3,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        If deep_supervision is True during training, returns a Tensor of shape [B, 4, out_channels, H, W]
        or list of multi-scale logit tensors.
        If deep_supervision is False / eval, returns highest resolution logits [B, out_channels, H, W].
        """
        out = self.nnunet(x)
        if not self.training:
            # DynUNet may return [B, NumHeads, C, H, W] (5D tensor) or a list/tuple of
            # multi-resolution tensors when deep_supervision=True. Extract highest resolution.
            if isinstance(out, (list, tuple)):
                return out[0]
            elif isinstance(out, torch.Tensor) and out.dim() == 5:
                return out[:, 0]
        return out
