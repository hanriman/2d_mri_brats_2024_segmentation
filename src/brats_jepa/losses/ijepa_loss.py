
import torch
import torch.nn.functional as F
from torch import nn


class IJEPALoss(nn.Module):
    """
    Computes prediction loss between predicted target representations
    and target encoder representations across target blocks.
    Following official I-JEPA (Assran et al., CVPR 2023), ONLY targets are LayerNormed;
    predictions are unnormalized to preserve scale/magnitude gradient flow.
    """
    def __init__(self, loss_type: str = "smooth_l1"):
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(predictions) == 0:
            if targets:
                return 0.0 * sum(t.sum() for t in targets)
            return torch.tensor(0.0, requires_grad=True)
        
        loss = torch.tensor(0.0, device=predictions[0].device)
        for pred, tgt in zip(predictions, targets):
            # Per official I-JEPA (Assran et al., 2023), only target is LayerNormed.
            # Prediction is NOT LayerNormed so the predictor learns natural scale.
            tgt_norm = F.layer_norm(tgt, (tgt.shape[-1],))
            
            if self.loss_type == "l1":
                block_loss = F.l1_loss(pred, tgt_norm)
            elif self.loss_type == "smooth_l1":
                block_loss = F.smooth_l1_loss(pred, tgt_norm)
            else:
                block_loss = F.mse_loss(pred, tgt_norm)
            loss = loss + block_loss
            
        return loss / len(predictions)
