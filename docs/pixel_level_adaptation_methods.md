# Pixel-Level Adaptation & Fine-Tuning Strategies for Medical JEPA

## 0. Overview & Research Gap

Joint-Embedding Predictive Architectures (JEPA) pre-train Vision Transformers by predicting abstract patch-level target token representations ($15 \times 15$ grid of $16 \times 16$ pixel patches). When fine-tuning JEPA for medical segmentation tasks (e.g., Glioma or Meningioma MRI segmentation), standard decoders upsample these coarse $15 \times 15$ tokens directly to $240 \times 240$ spatial masks.

While JEPA captures strong global semantics, coarse patch quantization ($16 \times 16$) limits spatial boundary precision. This document details **4 innovative, mathematically grounded strategies** for pixel-level adaptation, including **Low-Rank Adaptation (LoRA)** for ViT encoders.

---

## 1. Strategy Summary Comparison

| Strategy | Adaptation Mechanism | Extra Parameters | Primary Scientific Advantage |
| :--- | :--- | :--- | :--- |
| **Strategy 1: Multi-Scale Layer Fusion** | Intermediate ViT layer extraction ($L_2, L_4, L_6, L_8$) | $0$ parameters | Fuses early spatial geometry ($L_2$) with deep semantic abstraction ($L_8$). |
| **Strategy 2: Pixel-Token Cross-Attention** | Lightweight CNN spatial stem queries JEPA patch tokens | $+0.12\text{M}$ parameters | Directly modulates pixel intensity gradients ($240 \times 240$) with semantic JEPA vectors. |
| **Strategy 3: Sobel Boundary Loss** | 2D spatial gradient Dice loss ($\nabla \hat{Y}$ vs $\nabla Y$) | $0$ parameters | Forces decoder optimization to prioritize sharp, high-frequency tumor boundaries. |
| **Strategy 4: Vision Transformer LoRA** | Rank-decomposition matrices ($W_0 + \frac{\alpha}{r} BA$) in Q, V projections | $+0.05\text{M}$ ($<1\%$) | Prevents catastrophic forgetting of SSL representations; ideal for low-data regimes. |

---

## 2. Strategy 1: Multi-Scale Layer Token Fusion with Deep Supervision

### Concept
Standard JEPA decoders take tokens only from the final ViT layer ($L_8$). However, self-supervised ViT representations evolve hierarchically across transformer blocks:
- **Early Layers ($L_2, L_4$)**: Retain fine-grained localized spatial coordinates, edge boundaries, and intensity gradients.
- **Deep Layers ($L_6, L_8$)**: Capture abstract tissue semantics, global context, and class identity.

### Mathematical Formulation
Let $\mathbf{Z}^{(l)} \in \mathbb{R}^{B \times 225 \times 384}$ be the output patch token grid at transformer layer $l \in \{2, 4, 6, 8\}$. We construct a fused multi-stage token tensor:
$$\mathbf{Z}_{\text{fused}} = \text{Concat}\left(\mathbf{Z}^{(2)}, \, \mathbf{Z}^{(4)}, \, \mathbf{Z}^{(6)}, \, \mathbf{Z}^{(8)}\right) \in \mathbb{R}^{B \times 225 \times (4 \times 384)}$$
A linear projection maps $\mathbf{Z}_{\text{fused}} \to \mathbb{R}^{B \times 225 \times 384}$, which is then fed into a 4-stage Feature Pyramid Decoder with auxiliary Deep Supervision heads at resolutions $120 \times 120$, $60 \times 60$, and $30 \times 30$.

---

## 3. Strategy 2: Pixel-Token Cross-Attention Module

### Concept
To achieve pixel-level spatial modulation, we introduce a dual-branch cross-attention bridge between high-resolution 2D image feature maps and abstract JEPA patch tokens.

```
Input Image (240x240x4) ---> CNN Spatial Stem (Conv2D) ---> Features F_pixel (240x240x32) [QUERIES Q]
                                                                        |
                                                                        v
JEPA Encoder Tokens ------> Linear Projections ----------> Tokens Z_JEPA (225x384)   [KEYS K / VALUES V]
                                                                        |
                                                                        v
                                                   Dense Cross-Attention Matrix (Softmax(QK^T / sqrt(d)))
                                                                        |
                                                                        v
                                                   Pixel-Adapted High-Res Output (240x240x32)
```

### Mathematical Formulation
1. **Queries ($Q$)**: High-resolution spatial features $F_{\text{pixel}} \in \mathbb{R}^{B \times 32 \times 240 \times 240}$ extracted via a 2-layer CNN stem from the input slice, projected to $W_Q F_{\text{pixel}} \in \mathbb{R}^{B \times (240 \cdot 240) \times d}$.
2. **Keys ($K$) & Values ($V$)**: JEPA patch tokens $Z_{\text{JEPA}} \in \mathbb{R}^{B \times 225 \times 384}$, projected to $W_K Z_{\text{JEPA}} \in \mathbb{R}^{B \times 225 \times d}$ and $W_V Z_{\text{JEPA}} \in \mathbb{R}^{B \times 225 \times d}$.
3. **Cross-Attention**:
   $$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) \in \mathbb{R}^{B \times 57600 \times 225}$$
   $$\hat{F}_{\text{pixel}} = A \cdot V \in \mathbb{R}^{B \times 57600 \times d} \xrightarrow{\text{reshape}} \mathbb{R}^{B \times d \times 240 \times 240}$$

---

## 4. Strategy 3: Sobel Boundary-Guided Loss Regularization

### Concept
Standard Combined Dice + BCE loss treats all pixels equally, which often causes the loss to be dominated by bulk interior tumor pixels while neglecting fine spatial boundary errors. Sobel boundary regularization isolates 2D spatial intensity gradients, forcing the decoder to sharpen spatial predictions.

### Mathematical Formulation
We define 2D Sobel gradient operators $\mathbf{S}_x$ and $\mathbf{S}_y$:
$$\mathbf{S}_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad \mathbf{S}_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

For predicted probability map $\hat{Y} = \sigma(\text{logits})$ and ground-truth mask $Y$:
$$\nabla \hat{Y} = \sqrt{(\hat{Y} * \mathbf{S}_x)^2 + (\hat{Y} * \mathbf{S}_y)^2 + \epsilon}$$
$$\nabla Y = \sqrt{(Y * \mathbf{S}_x)^2 + (Y * \mathbf{S}_y)^2 + \epsilon}$$

The boundary Dice loss is computed as:
$$\mathcal{L}_{\text{boundary}} = 1.0 - \frac{2 \sum_{i,j} (\nabla \hat{Y}_{i,j} \cdot \nabla Y_{i,j}) + \epsilon}{\sum_{i,j} (\nabla \hat{Y}_{i,j})^2 + \sum_{i,j} (\nabla Y_{i,j})^2 + \epsilon}$$

Total Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DiceBCE}} + \lambda_{\text{edge}} \mathcal{L}_{\text{boundary}}$ ($\lambda_{\text{edge}} = 0.5$).

---

## 5. Strategy 4: Low-Rank Adaptation (LoRA) for JEPA Encoder

### Is LoRA Possible for JEPA ViT?
**YES!** LoRA (Low-Rank Adaptation) is directly applicable to the Vision Transformer backbone of JEPA.

### How LoRA Works for JEPA
Instead of updating all pre-trained weights $W_0 \in \mathbb{R}^{d \times d}$ ($384 \times 384$) in the Query ($W_q$) and Value ($W_v$) projection layers of each ViT block, we freeze $W_0$ and decompose the weight update $\Delta W$ into two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $A \in \mathbb{R}^{r \times d}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$, $B \in \mathbb{R}^{d \times r}$ is initialized to zeros, rank $r \ll d$ (e.g. $r = 8$ or $r = 16$), and $\alpha$ is a scaling constant (e.g. $\alpha = 16$).

```
                      Input Vector x (384-dim)
                               |
                +--------------+--------------+
                |                             |
                v                             v
       Frozen Pre-Trained W_0            Trainable Matrix A (r x 384)
          (384 x 384)                         |
                |                             v
                |                        Trainable Matrix B (384 x r)
                |                             |
                |                             v
                |                      Scaled by (alpha / r)
                |                             |
                +--------------+--------------+
                               |
                               v
                     Output Vector y (384-dim)
```

### Why LoRA is Ideal for Medical JEPA Pixel Adaptation
1. **Preserves SSL Non-Collapse Feature Rank**: Freezing $W_0$ ensures that the full feature rank ($\mathbf{366.91}$ in SigReg JEPA) learned during 50-epoch self-supervised pre-training is never destroyed.
2. **Prevents Overfitting in Low-Data Regimes**: Reduces trainable parameters from $22\text{M}$ down to $\sim 50\text{k}$ ($<1\%$ of model size). This dramatically boosts fine-tuning performance on $1\%$ and $5\%$ label scarcity tiers.
3. **Enables Dense Boundary Refinement**: Trainable rank matrices $A$ and $B$ learn spatial attention re-weighting specifically tuned to high-resolution tumor edge gradients.

---

## 6. Implementation Code Snippets

### PyTorch Implementation of LoRA Layer for ViT Attention
```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) linear wrapper for PyTorch nn.Linear layers."""
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False  # Freeze original pre-trained weight
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        self.rank = rank
        self.scaling = alpha / rank
        
        in_features = linear.in_features
        out_features = linear.out_features
        
        # Low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard frozen projection + low-rank adapted projection
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return base_out + lora_out * self.scaling

def inject_lora_into_jepa(jepa_encoder: nn.Module, rank: int = 8, alpha: float = 16.0):
    """Injects LoRA layers into Query (q_proj) and Value (v_proj) matrices of all ViT blocks."""
    for block in jepa_encoder.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, rank=rank, alpha=alpha)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, rank=rank, alpha=alpha)
    print(f"Successfully injected LoRA (rank={rank}, alpha={alpha}) into JEPA ViT Encoder!")
```
