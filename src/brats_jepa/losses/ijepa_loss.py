
import torch
import torch.nn.functional as F
from torch import nn


class IJEPALoss(nn.Module):
    """
    Computes Smooth L1 / L2 prediction loss between predicted target representations
    and target encoder representations across target blocks.
    """
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0
        for pred, tgt in zip(predictions, targets):
            # Normalize target representations to stabilize loss scale
            tgt_norm = F.layer_norm(tgt, (tgt.shape[-1],))
            pred_norm = F.layer_norm(pred, (pred.shape[-1],))
            
            if self.loss_type == "l1":
                block_loss = F.l1_loss(pred_norm, tgt_norm)
            else:
                block_loss = F.mse_loss(pred_norm, tgt_norm)
            loss += block_loss
            
        return loss / max(len(predictions), 1)
