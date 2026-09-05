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
Following official I-JEPA (Assran et al., 2023), only target representations are LayerNormed; predicted targets are unnormalized to preserve scale and magnitude gradients:
$$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{M} \sum_{k=1}^M \text{SmoothL1}\left(\hat{y}_{\text{tgt}}^{(k)}, \, \text{LayerNorm}(y_{\text{tgt}}^{(k)})\right)$$

---

## 4. SigReg JEPA (Heuristic-Free Single-Encoder SIGReg)

Located in [`src/brats_jepa/models/sigreg_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/sigreg_jepa.py).

### 4.1 Architecture Overview
SigReg JEPA (LeJEPA / SIGReg; Balestriero & LeCun, 2025) is a **heuristic-free single-encoder architecture**:
- **No EMA Teacher Encoder**: $E_{\bar{\theta}}$ is eliminated. Target representations are computed directly via the online encoder $E_\theta$ with gradients detached.
- **Projector MLP**: Encoder features $h \in \mathbb{R}^{384}$ pass through a 2-layer MLP (`Linear(384, 1024) -> LayerNorm -> GELU -> Linear(1024, 128)`) to produce projected representations $z \in \mathbb{R}^{128}$ for regularization. This decouples semantic representation learning from isotropic Gaussian collapse prevention.
- Representation collapse is prevented mathematically using **Sketched Isotropic Gaussian Regularization** (SIGReg).

### 4.2 Loss Formulation
$$\mathcal{L}_{\text{SigReg}} = \mathcal{L}_{\text{I-JEPA}} + \lambda_{\text{SIGReg}} \cdot \mathcal{T}_{\text{EP}}(g_\psi(Z))$$

1. **Cramér–Wold Slicing**: Projected tokens $z = g_\psi(h) \in \mathbb{R}^{N \times D_{\text{proj}}}$ are projected onto $M=256$ random unit vectors $u \sim \mathbb{S}^{D_{\text{proj}}-1}$.
2. **Epps–Pulley Goodness-of-Fit Test ($\mathcal{T}_{\text{EP}}$)**:
   For each 1D slice $p_m = z u_m$, the empirical characteristic function $\hat{\phi}(t) = \frac{1}{N} \sum_{n=1}^N \exp(i t p_{m, n})$ is compared against the standard normal characteristic function $\phi(t) = \exp(-t^2/2)$ via numerical quadrature over $t \in [0, 3.0]$:
   $$\mathcal{T}_{\text{EP}} = N \int_0^{t_{\max}} |\hat{\phi}(t) - \phi(t)|^2 e^{-t^2/2} \, dt$$

---

## 5. VisReg JEPA (Heuristic-Free Single-Encoder VISReg)

Located in [`src/brats_jepa/models/visreg_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/visreg_jepa.py).

### 5.1 Architecture Overview
VisReg JEPA (VISReg; Wu, Balestriero, Levine, 2026) is a **heuristic-free single-encoder architecture**:
- Eliminates momentum teacher updates.
- Decouples representation regularization into independent **Scale** and **Shape** objectives to avoid gradient vanishing during dimensional collapse.

### 5.2 Loss Formulation
$$\mathcal{L}_{\text{VisReg}} = \mathcal{L}_{\text{I-JEPA}} + \lambda_{\text{var}} \mathcal{L}_{\text{var}}(Z) + \lambda_{\text{SWD}} \mathcal{L}_{\text{SWD}}(Z)$$

1. **Scale Regularization (Batch Variance Hinge)**: Forces feature dimension variance across the batch above $\gamma = 1.0$:
   $$\mathcal{L}_{\text{var}}(Z) = \frac{1}{D} \sum_{j=1}^D \max\left(0, \, 1.0 - \sqrt{\text{Var}_B(Z_{:, j}) + \epsilon}\right)$$
2. **Shape Regularization (Sliced-Wasserstein Distance, SWD)**:
   Projects representations onto $M=256$ random unit directions $u \sim \mathbb{S}^{D-1}$, sorts the 1D projections $p_m = Z u_m$, and computes the $L_1$ Wasserstein distance against theoretical standard normal quantiles $\Phi^{-1}((i - 0.5)/N)$:
   $$\mathcal{L}_{\text{SWD}}(Z) = \frac{1}{M} \sum_{m=1}^M \frac{1}{N} \sum_{i=1}^N \left| p_{m, (i)} - \Phi^{-1}\left(\frac{i - 0.5}{N}\right) \right|$$

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
| **nnU-Net Baseline (SOTA)** | $9.66\text{ M}$ | $34.68\text{ s/epoch}$ | $20.87\text{ ms/slice}$ | Unnormalized Deep Supervision ($\sum 2^{-s} \mathcal{L}_s$) |
| **I-JEPA Encoder + Predictor** | $16.06\text{ M}$ | $26.96\text{ s/epoch}$ | $20.38\text{ ms/slice}$ | Latent Smooth L1 + Target LayerNorm + EMA Teacher |
| **SigReg JEPA Encoder + Projector** | $16.58\text{ M}$ | **$21.18\text{ s/epoch}$** | **$20.65\text{ ms/slice}$** | Latent Smooth L1 + Epps–Pulley CF Test ($\mathcal{T}_{\text{EP}}$) |
| **VisReg JEPA Encoder** | $16.06\text{ M}$ | **$21.56\text{ s/epoch}$** | $20.71\text{ ms/slice}$ | Latent Smooth L1 + Batch Var Hinge + Sliced-Wasserstein (SWD) |
| **JEPASegmentationModel** | $14.67\text{ M}$ | $21.18\text{ s/epoch}$ | $20.52\text{ ms/slice}$ | Combined Dice + BCE |
