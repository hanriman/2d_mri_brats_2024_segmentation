import torch
import torch.nn as nn
import torch.nn.functional as F
from .dice_bce_loss import CombinedDiceBCELoss

class DeepSupervisionLoss(nn.Module):
    r"""
    Multi-Scale Deep Supervision Loss for nnU-Net Architectures.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Multi-Resolution Gradient Flow:
       Standard backpropagation through deep encoder-decoder segmentation networks suffers
       from vanishing gradients in the early encoder stages and bottleneck layers. Deep supervision
       attaches auxiliary segmentation heads at intermediate decoder stages (e.g. 15x15, 30x30,
       60x60, 120x120, 240x240), injecting direct supervision gradients throughout all decoder depths.

    2. Exponential Decay Weighting:
       Following nnU-Net (Isensee et al., Nature Methods 2021, Section "Loss Function"):
           w_s = \frac{2^{-s}}{\sum_{j=0}^{S-1} 2^{-j}} \quad \text{for stage } s \in \{0, 1, \dots, S-1\}
       Lower resolution stages receive exponentially lower weights (e.g., [1.0, 0.5, 0.25] / 1.75 =
       [0.571, 0.286, 0.143]), ensuring that low-resolution coarse representations guide global
       localization without corrupting fine boundary delineation at the highest resolution.
       Normalizing weights guarantees that the overall loss magnitude remains identical to standard
       single-resolution training, avoiding learning rate recalibration.

    3. Nearest-Neighbor Target Resampling:
       Ground truth binary tumor masks are downsampled using nearest-neighbor interpolation
       (`mode="nearest"`) to preserve binary label semantics (0 or 1) without artificial continuous blurring.

    References:
    -----------
    - Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
      "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation."
      Nature Methods, 18(2), 203-211.
    - Lee, C. Y., et al. (2015). "Deeply-Supervised Nets." AISTATS 2015, pp. 562-570.
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
        total_loss = torch.tensor(0.0, device=target.device)
        raw_weights = [1.0 / (2**i) for i in range(len(head_list))]
        # Normalize weights so sum(weights) == 1.0 per nnU-Net protocol
        w_sum = sum(raw_weights)
        weights = [w / w_sum for w in raw_weights]
        
        for head_logits, w in zip(head_list, weights):
            if head_logits.shape[-2:] != target.shape[-2:]:
                target_scaled = F.interpolate(target, size=head_logits.shape[-2:], mode="nearest")
            else:
                target_scaled = target
            total_loss = total_loss + w * self.base_loss(head_logits, target_scaled)
            
        return total_loss
