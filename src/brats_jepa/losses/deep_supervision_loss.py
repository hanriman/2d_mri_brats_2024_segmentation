import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_bce_loss import CombinedDiceBCELoss

class DeepSupervisionLoss(nn.Module):
    """
    Computes multi-scale weighted Dice + BCE loss for nnU-Net deep supervision outputs.
    Downweights lower resolution heads with factor w_s = 1 / (2^s).
    """
    def __init__(self):
        super().__init__()
        self.base_loss = CombinedDiceBCELoss(dice_weight=1.0, bce_weight=1.0)

    def forward(self, logits: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
        # Handle DynUNet 5D output tensor [B, NumHeads, C, H, W]
        if isinstance(logits, torch.Tensor) and logits.dim() == 5:
            num_heads = logits.shape[1]
            head_list = [logits[:, i] for i in range(num_heads)]
            return self._compute_multi_head_loss(head_list, target)
        elif isinstance(logits, (list, tuple)):
            return self._compute_multi_head_loss(list(logits), target)
        else:
            return self.base_loss(logits, target)

    def _compute_multi_head_loss(self, head_list: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        weights = [1.0 / (2**i) for i in range(len(head_list))]
        
        for head_logits, w in zip(head_list, weights):
            if head_logits.shape[-2:] != target.shape[-2:]:
                target_scaled = F.interpolate(target, size=head_logits.shape[-2:], mode="nearest")
            else:
                target_scaled = target
            total_loss += w * self.base_loss(head_logits, target_scaled)
            
        return total_loss
