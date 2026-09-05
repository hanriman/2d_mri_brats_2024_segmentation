
import torch
import torch.nn.functional as F
from torch import nn


class IJEPALoss(nn.Module):
    r"""
    I-JEPA Latent Representation Prediction Loss.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Asymmetric Target Normalization:
       Following official I-JEPA (Assran et al., CVPR 2023, Section 3.2):
           \mathcal{L}_y = \frac{1}{M} \sum_{i=1}^M \mathcal{D}(\hat{s}_y^{(i)}, \text{LayerNorm}(s_y^{(i)}))
       Only the teacher target representations s_y are LayerNormed. The predictor outputs
       \hat{s}_y are deliberately UNNORMALIZED. If predictions were normalized, the predictor
       could alter its output magnitude arbitrarily without gradient penalty, destabilizing
       the latent coordinate scale. Normalizing targets establishes a standardized, zero-mean,
       unit-variance coordinate target for the online network.

    2. Smooth L1 (Huber) Robustness in Medical MRI:
       Standard MSE loss quadratic penalties (\frac{1}{2} e^2) disproportionately amplify
       large outliers. In 2D multi-modal brain MRI, hyperintense necrotic cores, contrast-enhancing
       margins, or skull-stripping boundary artifacts produce heavy-tailed representation errors
       in early epochs. Smooth L1 transitions to linear penalties (|e| - 0.5) for |e| \ge 1,
       suppressing gradient shocks and providing stable convergence across heterogeneous scans.

    References:
    -----------
    - Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., &
      Ballas, N. (2023). "Self-Supervised Learning from Images with a Joint-Embedding
      Predictive Architecture." IEEE/CVF CVPR 2023, pp. 15619-15629.
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
            # Return a zero loss disconnected from targets. Targets come from
            # the EMA teacher and must never receive gradients.
            device = targets[0].device if targets else torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
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
