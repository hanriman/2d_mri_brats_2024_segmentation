import torch
from monai.losses import DiceLoss
from torch import nn


class CombinedDiceBCELoss(nn.Module):
    """Combined MONAI Dice Loss + BCEWithLogitsLoss for 2D binary tumor segmentation."""
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
