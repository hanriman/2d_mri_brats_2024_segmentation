import torch
from monai.losses import DiceLoss
from torch import nn


class CombinedDiceBCELoss(nn.Module):
    r"""
    Combined Soft Dice Loss + Binary Cross-Entropy with Logits.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Extreme Foreground-Background Class Imbalance:
       In 2D brain MRI slices (240 x 240 = 57,600 pixels), glioma lesions frequently occupy
       under 2% of the spatial volume, while 46.3% of slices contain no tumor at all.
       Standard Binary Cross-Entropy (BCE) gradients are heavily dominated by true negatives
       (background), driving the model toward predicting all zeros.
    
    2. Complementary Gradient Dynamics:
       - **Soft Dice Loss** directly maximizes the spatial overlap (Sørensen-Dice coefficient):
             \mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
         which is inherently invariant to background volume.
       - **Binary Cross-Entropy** provides smooth, convex, pixel-level probability calibration:
             \mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_i [g_i \log(\sigma(l_i)) + (1 - g_i) \log(1 - \sigma(l_i))]
         preventing Dice loss from encountering plateaued or erratic gradients when predictions
         diverge early in training.
       Combining both with equal weights (\lambda_{\text{Dice}} = 1.0, \lambda_{\text{BCE}} = 1.0)
       is the established gold standard in medical image segmentation benchmarks.

    References:
    -----------
    - Milletari, F., Navab, N., & Ahmadi, S. A. (2016). "V-Net: Fully Convolutional Neural
      Networks for Volumetric Medical Image Segmentation." 3DV 2016, pp. 565-571.
    - Isensee, F., et al. (2021). "nnU-Net: a self-configuring method for deep learning-based
      biomedical image segmentation." Nature Methods, 18(2), 203-211.
    """
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(sigmoid=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)  # BCEWithLogitsLoss requires float targets
        d_loss = self.dice_loss(logits, targets)
        b_loss = self.bce_loss(logits, targets)
        return self.dice_weight * d_loss + self.bce_weight * b_loss
