# Detailed Model Architecture Documentation

This document provides a comprehensive technical reference for all neural network architectures implemented in the `brats_jepa` package.

---

## Table of Contents
1. [Input Data Tensor Specifications](#1-input-data-tensor-specifications)
2. [Vision Transformer Encoder (ViT-Small)](#2-vision-transformer-encoder-vit-small)
3. [I-JEPA (Dual-Encoder with EMA Teacher Target Encoder)](#3-i-jepa-dual-encoder-with-ema-teacher-target-encoder)
4. [SigReg JEPA (Heuristic-Free Single-Encoder SIGReg)](#4-sigreg-jepa-heuristic-free-single-encoder-sigreg)
5. [VisReg JEPA (Heuristic-Free Single-Encoder VISReg)](#5-visreg-jepa-heuristic-free-single-encoder-visreg)
6. [Downstream ViT Segmentation Decoder (JEPASegmentationModel)](#6-downstream-vit-segmentation-decoder-jepasegmentationmodel)
7. [2D Residual UNet Baseline](#7-2d-residual-unet-baseline)
8. [2D nnU-Net Baseline with Deep Supervision](#8-2d-nnu-net-baseline-with-deep-supervision)
9. [Model Parameter & Runtime Summary](#9-model-parameter--runtime-summary)

---

## 1. Input Data Tensor Specifications

Every model accepts 2D multi-modal Magnetic Resonance Imaging (MRI) axial slices extracted from the BraTS GLI dataset:

$$\mathbf{X} \in \mathbb{R}^{B \times 4 \times 240 \times 240}$$

- **Batch Size ($B$)**: Configurable (default $B=8$).
- **Input Channels ($C=4$)**:
  1. $X_{\text{T1}}$: T1-weighted sequence
  2. $X_{\text{T1c}}$: Post-contrast T1-weighted sequence
  3. $X_{\text{T2}}$: T2-weighted sequence
  4. $X_{\text{FLAIR}}$: Fluid Attenuated Inversion Recovery sequence
- **Spatial Resolution**: $H = 240$ pixels, $W = 240$ pixels.
- **Normalization**: Z-score normalized independently per MRI channel over non-zero brain voxels.

---

## 2. Vision Transformer Encoder (ViT-Small)

Located in [`src/brats_jepa/models/vision_transformer.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/vision_transformer.py).

### 2.1 Mathematical Formulation
1. **Patch Embedding**: Image slices are split into non-overlapping patches of size $P \times P = 16 \times 16$:
   $$N = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right) = 15 \times 15 = 225 \text{ patches}$$
   The patch embedding layer applies 2D convolution with stride $P$:
   $$\mathbf{E}_{\text{patches}} = \text{Conv2d}_{4 \to 384, \, k=16, \, s=16}(\mathbf{X}) \in \mathbb{R}^{B \times 384 \times 15 \times 15} \xrightarrow{\text{flatten}} \mathbb{R}^{B \times 225 \times 384}$$

2. **Positional Encoding**: Learnable 1D spatial position embeddings $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{1 \times 225 \times 384}$ are added:
   $$\mathbf{Z}_0 = \mathbf{E}_{\text{patches}} + \mathbf{E}_{\text{pos}}$$

3. **Transformer Encoder Layer**:
   For each block $l \in \{1, \dots, 8\}$ using Pre-LayerNorm (`norm_first=True`):
   $$\mathbf{Z}'_l = \mathbf{Z}_{l-1} + \text{MultiHeadAttention}(\text{LayerNorm}(\mathbf{Z}_{l-1}))$$
   $$\mathbf{Z}_l = \mathbf{Z}'_l + \text{MLP}(\text{LayerNorm}(\mathbf{Z}'_l))$$

### 2.2 Layer Specification Table

| Layer Name | Type / Operation | Input Shape | Output Shape | Parameters |
| :--- | :--- | :--- | :--- | :--- |
| `patch_embed.proj` | `nn.Conv2d(4, 384, k=16, s=16)` | $[B, 4, 240, 240]$ | $[B, 384, 15, 15]$ | $24,960$ |
| `pos_embed` | `nn.Parameter` | - | $[1, 225, 384]$ | $86,400$ |
| `blocks.0` -- `blocks.7` | $8 \times$ `TransformerEncoderLayer` | $[B, N_{\text{tokens}}, 384]$ | $[B, N_{\text{tokens}}, 384]$ | $14,136,576$ |
| `norm` | `nn.LayerNorm(384)` | $[B, N_{\text{tokens}}, 384]$ | $[B, N_{\text{tokens}}, 384]$ | $768$ |

- **Embedding Dimension ($D$)**: $384$
- **Attention Heads**: $6$ ($\text{head\_dim} = 64$)
- **MLP Expansion**: $4.0 \times 384 = 1536$
- **Total Encoder Parameters**: $\mathbf{14,248,320}$ ($\sim 14.25\text{ M}$)

---

## 3. I-JEPA (Dual-Encoder with EMA Teacher Target Encoder)

Located in [`src/brats_jepa/models/ijepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/ijepa.py).

### 3.1 Architecture Overview
I-JEPA uses a **dual-encoder architecture**:
1. **Online Student Context Encoder ($E_\theta$)**: Processes only the visible context patches $x_{\text{ctx}}$ ($N_{\text{ctx}} = 196$).
2. **EMA Teacher Target Encoder ($E_{\bar{\theta}}$)**: Processes the unmasked slice $X$ to generate ground-truth target patch embeddings $y_{\text{tgt}}$.
3. **Predictor Network ($P_\phi$)**: Takes context representations + learnable target mask tokens to predict target representations in latent space.

```text
  Context Patches (196) ---> Online Context Encoder E_theta ---> Context Tokens z_ctx [B, 196, 384]
                                                                        |
                                                                        v
  Target Position Tokens ---> Predictor Network P_phi ------------> Predicted Targets y^_tgt [B, 25, 384]
                                                                        |
                                                                        v (Smooth L1 Loss)
  Full Slice X -----------> EMA Teacher Encoder E_theta_bar -------> True Target Tokens y_tgt [B, 25, 384]
```

### 3.2 Momentum Update Equation
The EMA teacher parameters $\bar{\theta}$ are updated without gradients at step $t$ via momentum $m = 0.996$:
$$\bar{\theta}_t \leftarrow m \bar{\theta}_{t-1} + (1 - m) \theta_t$$

### 3.3 Loss Function
$$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{M} \sum_{k=1}^M \left\| \text{LayerNorm}(\hat{y}_{\text{tgt}}^{(k)}) - \text{LayerNorm}(y_{\text{tgt}}^{(k)}) \right\|_1$$

---

## 4. SigReg JEPA (Heuristic-Free Single-Encoder SIGReg)

Located in [`src/brats_jepa/models/sigreg_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/sigreg_jepa.py).

### 4.1 Architecture Overview
SigReg JEPA (LeJEPA / SIGReg) is a **heuristic-free single-encoder architecture**:
- **NO EMA Teacher Encoder**: $E_{\bar{\theta}}$ is eliminated. Target representations are computed directly via the online encoder $E_\theta$ with gradients detached.
- Representation collapse is prevented mathematically using **Sketched Isotropic Gaussian Regularization** (SIGReg) on latent features $Z \in \mathbb{R}^{N \times D}$.

### 4.2 Loss Formulation
$$\mathcal{L}_{\text{SigReg}} = \mathcal{L}_{\text{I-JEPA}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}}(Z) + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}}(Z)$$

1. **Variance Hinge Loss ($\mathcal{L}_{\text{var}}$)**: Forces standard deviation of each feature dimension above $\gamma = 1.0$:
   $$\mathcal{L}_{\text{var}}(Z) = \frac{1}{D} \sum_{j=1}^D \max\left(0, \, 1.0 - \sqrt{\text{Var}(Z_{:, j}) + \epsilon}\right)$$
2. **Covariance Decorrelation Loss ($\mathcal{L}_{\text{cov}}$)**: Penalizes off-diagonal cross-feature dimension correlations:
   $$\mathbf{C} = \frac{1}{N-1} (Z - \bar{Z})^T (Z - \bar{Z}), \qquad \mathcal{L}_{\text{cov}}(Z) = \frac{1}{D} \sum_{i \neq j} \mathbf{C}_{i, j}^2$$

---

## 5. VisReg JEPA (Heuristic-Free Single-Encoder VISReg)

Located in [`src/brats_jepa/models/visreg_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/visreg_jepa.py).

### 5.1 Architecture Overview
VisReg JEPA (VISReg) is a **heuristic-free single-encoder architecture**:
- Eliminates momentum teacher updates.
- Prevents representation over-smoothing across spatial patch locations by enforcing spatial feature variance contrast.

### 5.2 Loss Formulation
$$\mathcal{L}_{\text{VisReg}} = \mathcal{L}_{\text{I-JEPA}} + \lambda_{\text{vis}} \cdot \frac{1}{B} \sum_{b=1}^B \max\left(0, \, 1.0 - \sqrt{\text{Var}_{\text{patch}}(Z_b) + \epsilon}\right)$$

---

## 6. Downstream ViT Segmentation Decoder (JEPASegmentationModel)

Located in [`src/brats_jepa/models/segmentation_head.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/segmentation_head.py).

### 6.1 Architecture Specification
Couples the pre-trained `VisionTransformerEncoder2D` with a 4-stage transpose-convolutional upsampling decoder to map $15 \times 15$ ViT patch tokens back to full $240 \times 240$ spatial logits:

```text
ViT Patch Tokens [B, 225, 384]  ---> Reshape & Permute ---> [B, 384, 15, 15]
                                                                  |
 Stage 1: ConvTranspose2d(384, 192, k=2, s=2) + GroupNorm(16) + GELU ---> [B, 192, 30, 30]
                                                                  |
 Stage 2: ConvTranspose2d(192, 96,  k=2, s=2) + GroupNorm(8)  + GELU ---> [B, 96, 60, 60]
                                                                  |
 Stage 3: ConvTranspose2d(96,  48,  k=2, s=2) + GroupNorm(4)  + GELU ---> [B, 48, 120, 120]
                                                                  |
 Stage 4: ConvTranspose2d(48,  24,  k=2, s=2) + GroupNorm(4)  + GELU ---> [B, 24, 240, 240]
                                                                  |
 Projection Head: Conv2d(24, 1, k=1)                              ---> [B, 1, 240, 240] Logits
```

- **Decoder Parameter Count**: $420,841$ ($\sim 0.42\text{ M}$)
- **Total Segmentation Model Parameters**: $14,669,161$ ($\sim 14.67\text{ M}$)

---

## 7. 2D Residual UNet Baseline

Located in [`src/brats_jepa/models/unet.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/unet.py).

### 7.1 Architecture Specification
Supervised 5-stage encoder-decoder convolutional network:
- **Encoder Channels**: $4 \to 32 \to 64 \to 128 \to 256$
- **Bottleneck**: $256 \to 512$
- **Decoder Channels**: $512 \to 256 \to 128 \to 64 \to 32 \to 1$
- **Skip Connections**: Concatenation of encoder feature maps at corresponding spatial resolutions.
- **Parameters**: $1,863,201$ ($\sim 1.86\text{ M}$)

---

## 8. 2D nnU-Net Baseline with Deep Supervision

Located in [`src/brats_jepa/models/nnunet.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/nnunet.py).

### 8.1 Architecture Specification
State-of-the-art supervised baseline (Isensee et al., *Nature Methods* 2021) wrapping MONAI `DynUNet`:
- **Encoder Blocks**: 5 residual encoder blocks with Instance Normalization and LeakyReLU activations.
- **Channels**: `(32, 64, 128, 256, 512)`
- **Strides**: `[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]`

### 8.2 Deep Supervision Multi-Scale Heads
Outputs auxiliary logits at intermediate resolution levels during training:
1. Head 0 (Main): $[B, 1, 240, 240]$ (Weight $w_0 = 1.0$)
2. Head 1: $[B, 1, 120, 120]$ (Weight $w_1 = 0.5$)
3. Head 2: $[B, 1, 60, 60]$ (Weight $w_2 = 0.25$)
4. Head 3: $[B, 1, 30, 30]$ (Weight $w_3 = 0.125$)

$$\mathcal{L}_{\text{deep\_sup}} = \sum_{s=0}^3 w_s \cdot \mathcal{L}_{\text{Dice+BCE}}(\hat{Y}_s, Y_s)$$

- **Parameters**: $9,655,908$ ($\sim 9.66\text{ M}$)

---

## 9. Model Parameter & Runtime Summary

| Model Architecture | Parameter Count | Training Speed | Inference Latency | Primary Loss Function |
| :--- | :--- | :--- | :--- | :--- |
| **UNet Baseline** | $1.86\text{ M}$ | $46.30\text{ s/epoch}$ | $175.86\text{ ms/slice}$ | Combined Dice + BCE |
| **nnU-Net Baseline (SOTA)** | $9.66\text{ M}$ | $34.68\text{ s/epoch}$ | $20.87\text{ ms/slice}$ | Deep Supervision Multi-Scale Loss |
| **I-JEPA Encoder + Predictor** | $16.06\text{ M}$ | $26.96\text{ s/epoch}$ | $20.38\text{ ms/slice}$ | Latent Smooth L1 + EMA Teacher |
| **SigReg JEPA Encoder** | $16.06\text{ M}$ | **$21.18\text{ s/epoch}$** | **$20.65\text{ ms/slice}$** | Latent Smooth L1 + SIGReg (Var/Cov) |
| **VisReg JEPA Encoder** | $16.06\text{ M}$ | **$21.56\text{ s/epoch}$** | $20.71\text{ ms/slice}$ | Latent Smooth L1 + VISReg (Spatial Var) |
| **JEPASegmentationModel** | $14.67\text{ M}$ | $21.18\text{ s/epoch}$ | $20.52\text{ ms/slice}$ | Combined Dice + BCE |
