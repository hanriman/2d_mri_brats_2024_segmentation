import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

class BraTS2DnnUNet(nn.Module):
    r"""
    2D nnU-Net Architecture with Deep Supervision (Isensee et al., Nature Methods 2021).

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Gold-Standard Clinical Segmentation Benchmark:
       nnU-Net is the undisputed competitive standard in the annual BraTS challenges (2018-2024).
       Including a rigorously implemented 2D nnU-Net baseline provides an authoritative ceiling
       against which self-supervised representation architectures (I-JEPA, SigReg, VisReg) are evaluated.

    2. Key Architectural Innovations:
       - **Residual Encoders**: Skip convolutions inside each stage mitigate vanishing gradients.
       - **LeakyReLU Activations** (\alpha = 0.01): Prevents dying neurons in the vast zero-intensity
         brain MRI background voxels.
       - **Instance Normalization**: Stabilizes training independently of mini-batch slice compositions.
       - **Multi-Scale Deep Supervision** (`deep_supr_num=3`): Intermediate segmentation heads
         supervise decoder features at resolutions 30x30, 60x60, 120x120, and 240x240, accelerating
         hierarchical representation learning.

    3. Inference Resolution Guarantee:
       During evaluation (`self.training == False`), the forward pass automatically strips away auxiliary
       deep supervision heads and outputs only the highest-resolution [B, 1, 240, 240] prediction.

    References:
    -----------
    - Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
      "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."
      Nature Methods, 18(2), 203-211.
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
            res_block=True,
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
