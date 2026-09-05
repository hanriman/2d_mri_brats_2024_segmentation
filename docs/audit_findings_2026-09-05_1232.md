# Comprehensive Codebase, Theoretical & Architectural Audit Report

**Date:** 2026-09-05  
**Time:** 12:32 CEST  
**Repository:** `thesis_2d` (`2d_mri_brats_2024_segmentation`)  
**Domain:** Multi-Modal 2D MRI Brain Tumor Segmentation (BraTS 2024 Glioma & Meningioma)  
**Models Audited:** I-JEPA, SigReg JEPA (LeJEPA), VisReg JEPA, 2D Residual UNet, 2D nnU-Net with Deep Supervision  
**Auditor:** Antigravity AI Research Agent (in pair programming with Han Riman)

---

## 1. Executive Summary & Status

This document records the exhaustive audit of the mathematical theory, foundational papers, neural network architectures, data preprocessing pipelines, optimization mechanics, evaluation protocols, and LaTeX manuscript files in the `thesis_2d` codebase.

All identified theoretical discrepancies, indexing bottlenecks, optimization instabilities, missing command-line arguments, and experimental confounders have been resolved, verified with 23 passing unit tests (`uv run pytest -v`), smoke-tested end-to-end across all execution entry points, and formally incorporated into the scientific manuscript [`paper/latex/extended_main.tex`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.tex).

### Verification Scorecard

| Area | Audit Verdict | Status | Verification Evidence |
| :--- | :--- | :--- | :--- |
| **VISReg Loss Formulation** | Coupled scale & shape; caused severe rank collapse ($16.92$) | **RESOLVED** | Standardized 1D projections ($\mathcal{W}_1: 0.8059 \to 0.0267$); added Projector MLP ($384 \to 1024 \to 128$); unit test [`test_visreg_scale_shape_decoupling`](file:///Users/hanriman/Documents/master/thesis_2d/tests/test_losses.py) passing. |
| **ViT & Predictor Indexing** | Coupled 1D/2D mask dimensions; $O(B)$ Python loops | **RESOLVED** | Vectorized subset selection and positional embeddings via `torch.gather`; unit test [`test_jepa_predictor_heterogeneous_shapes`](file:///Users/hanriman/Documents/master/thesis_2d/tests/test_models.py) passing. |
| **Modality Dropout** | Unfair comparison: UNet trained without dropout while JEPA had it | **RESOLVED** | Unified [`RandomModalityDropout`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/data/transforms.py) with non-empty active fallback; added `--p_drop` across all training scripts; unit test [`test_random_modality_dropout`](file:///Users/hanriman/Documents/master/thesis_2d/tests/test_data.py) passing. |
| **Low-Data Optimization** | Gradient spikes and divergence at $1\%$ and $5\%$ labels | **RESOLVED** | Integrated `clip_grad_norm_` ($1.0$) and `CosineAnnealingLR` scheduler decay in [`scripts/evaluate_low_data.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_low_data.py). |
| **Noise Model Physics** | Ambiguous Rician terminology on standardized Z-score data | **RESOLVED** | Formally clarified high-SNR asymptotic Rician limit ($\text{SNR} \gg 1 \implies \text{Rician} \to \text{Gaussian}$) in code and paper. |
| **Notebook Compatibility** | Missing `--output_dir` in OOD scripts; code zip upload confusion | **RESOLVED** | Added `--output_dir` to [`scripts/evaluate_ood.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_ood.py) and [`scripts/evaluate_men_rt_ood.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_men_rt_ood.py); removed zip code upload in notebooks to rely solely on GitHub clone. |
| **Scientific Manuscript** | Lacked skip connection analyses, randomization details, and fixes | **RESOLVED** | Updated [`paper/latex/extended_main.tex`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.tex); successfully compiled 11-page PDF [`extended_main.pdf`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.pdf). |

---

## 2. Detailed Findings & Mathematical Remediations

### 2.1 VISReg Loss: Sliced-Wasserstein Scale-Shape Decoupling & Projector MLP

#### The Problem
In the original implementation of [`VisRegLoss`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/visreg_loss.py), the 1D Sliced-Wasserstein Distance (SWD) was computed directly between raw projected tokens $p_m = Z u_m$ and standard normal quantiles $\Phi^{-1}((i - 0.5)/N)$.
When the feature variance $\sigma^2 > 1$, empirical quantiles were scaled by $\sigma$, dominating the SWD penalty ($\mathcal{W}_1 \approx 0.8059$ for Gaussian inputs with $\sigma = 2.0$). This penalized variance expansion and forced encoder representations into a low-dimensional subspace, causing catastrophic rank collapse ($\text{Rank}_{\text{eff}} = 16.92$). Furthermore, regularizing the raw backbone tokens directly impaired spatial token semantics needed for downstream segmentation.

#### The Mathematical Fix
1. **Projection Standardization:** Before sorting projections along each 1D slice $m$, the projections are standardized:
   $$\tilde{p}_{im} = \frac{p_{im} - \mu_m}{\sigma_m + \epsilon}, \qquad \mu_m = \frac{1}{K}\sum_{i=1}^K p_{im}, \quad \sigma_m = \sqrt{\frac{1}{K}\sum_{i=1}^K (p_{im} - \mu_m)^2 + \epsilon}$$
   Standardization isolates the higher-order distribution shape ($skewness, kurtosis, tail decay$) from variance and location. Scale regularization is independently enforced by the batch variance hinge loss:
   $$\mathcal{L}_{\text{var}} = \frac{1}{d_p} \sum_{j=1}^{d_p} \max\left(0, \, 1.0 - \sqrt{\text{Var}(Z_{:, j}) + \epsilon}\right)$$
   Under this formulation, Gaussian tokens with $\sigma = 2.0$ yield $\mathcal{W}_1 = 0.0267 \ll 0.10$ and $\mathcal{L}_{\text{var}} = 0.0$, strictly decoupling scale and shape.
2. **Projector MLP:** Implemented a 2-layer Projector MLP in [`VisRegJEPA`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/visreg_jepa.py):
   $$\mathbf{Z}_{\text{proj}} = \text{Linear}_{1024 \to 128}\left(\text{GELU}\left(\text{LayerNorm}\left(\text{Linear}_{384 \to 1024}(\mathbf{Z}_{\text{ctx}})\right)\right)\right) \in \mathbb{R}^{B \times N_{\text{ctx}} \times 128}$$
   Raw encoder tokens remain spatially expressive and localized for downstream decoding, while projected tokens satisfy isotropic Gaussianity.

---

### 2.2 Vectorized ViT Indexing & JEPAPredictor Dimension Decoupling

#### The Problem
In [`src/brats_jepa/models/vision_transformer.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/vision_transformer.py):
1. `JEPAPredictor` coupled the tensor dimensionalities of `context_indices` and `target_indices`. When `context_indices` was per-sample 2D ($[B, N_{\text{ctx}}]$) and `target_indices` was 1D ($[N_{\text{tgt}}]$), a tensor dimension mismatch triggered:
   `RuntimeError: The size of tensor a (25) must match the size of tensor b (2) at non-singleton dimension 0`
2. Positional embedding indexing and subset patch extraction used an $O(B)$ Python loop with `torch.stack`, creating CPU-GPU synchronization stalls.

#### The Implementation Fix
Decoupled the dimension handling for `context_indices` and `target_indices` using vectorized `torch.gather`:
```python
# Context positional embeddings (handles 1D and 2D independently)
if context_indices.dim() == 1:
    ctx_pos = self.pos_embed[:, context_indices, :].expand(B, -1, -1)
else:
    ctx_pos = self.pos_embed.expand(B, -1, -1).gather(
        1, context_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.size(-1))
    )

# Target positional embeddings (handles 1D and 2D independently)
if target_indices.dim() == 1:
    tgt_pos = self.pos_embed[:, target_indices, :].expand(B, -1, -1)
else:
    tgt_pos = self.pos_embed.expand(B, -1, -1).gather(
        1, target_indices.unsqueeze(-1).expand(-1, -1, self.pos_embed.size(-1))
    )
```
Subset patch selection in [`VisionTransformerEncoder2D`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/vision_transformer.py) was similarly vectorized with `torch.gather`, completely eliminating the Python batch loop.

---

### 2.3 Modality Dropout Standardization & Controlled Ablations

#### The Problem
The previous manuscript claimed that UNet collapsed under zero-padded missing modalities ($0.0151$ Dice) while JEPAs survived due to self-supervised pre-training. However, this comparison was confounded: UNet was trained *without* modality dropout, whereas JEPA was trained *with* modality dropout inside its internal dataset transform.

#### The Implementation Fix
1. Extracted [`RandomModalityDropout`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/data/transforms.py) into the shared transforms library with a mathematical fallback guaranteeing that at least one modality channel remains active for every sample:
   ```python
   class RandomModalityDropout(nn.Module):
       def __init__(self, p_drop: float = 0.25):
           super().__init__()
           self.p_drop = p_drop

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           if not self.training or self.p_drop <= 0.0:
               return x
           B, C, H, W = x.shape
           mask = (torch.rand(B, C, 1, 1, device=x.device) > self.p_drop).float()
           all_zero = (mask.sum(dim=1, keepdim=True) == 0)
           random_channel = torch.randint(0, C, (B, 1, 1, 1), device=x.device)
           fallback = torch.zeros_like(mask).scatter_(1, random_channel, 1.0)
           mask = torch.where(all_zero, fallback, mask)
           return x * mask
   ```
2. Exposed `--p_drop` across all training scripts ([`train_unet.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/train_unet.py), [`train_nnunet.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/train_nnunet.py), [`train_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/train_jepa.py), [`train_downstream.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/train_downstream.py)), enabling rigorous, controlled ablations.

---

### 2.4 Low-Data Optimization Stabilization

#### The Problem
Fine-tuning full ViT backbones on $1\%$ labels ($63$ slices) with AdamW at a fixed learning rate ($10^{-4}$) exhibited loss spikes and gradient instability.

#### The Implementation Fix
Integrated two stabilization techniques across all models in [`scripts/evaluate_low_data.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_low_data.py):
- `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- `CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)`

---

### 2.5 High-SNR Asymptotic Rician Noise Formulation

#### The Problem
`apply_rician_noise` applied additive Gaussian noise $\mathcal{N}(0, \sigma^2)$ to foreground tissue. Describing this as raw Rician noise was technically inaccurate, as true magnitude Rician noise ($M = \sqrt{(S + n_1)^2 + n_2^2}$) requires non-negative raw RF magnitudes; applying square-root operations to standardized Z-scores destroys the negative domain ($x < 0$).

#### The Formulation Fix
Updated docstrings, logs, and paper text to formally specify:
In standardized Z-score space, we apply additive Gaussian noise approximating the **high-SNR asymptotic limit of Rician noise** ($\text{SNR} \gg 1 \implies \text{Rician}(\nu, \sigma) \to \mathcal{N}(\nu, \sigma^2)$) on non-zero tissue voxels ($\sigma_{\text{noise}} = 0.15$).

---

## 3. Comprehensive Architectural Micro-Designs

### 3.1 Skip Connections: Comparative Mechanics

A central architectural theme in this research is the contrast between **dense multi-scale skip connections** and **pure bottleneck semantic representations**:

```text
1. 2D Residual UNet (ResUNet):
   Encoder (32->64->128->256->512) ─── Horizontal Skips (Concat) ───> Decoder (512->256->128->64->32)
   └─ Residual Basic Blocks: x + F(x)                                   └─ Residual Basic Blocks: x + F(x)

2. 2D nnU-Net (DynUNet with Deep Supervision):
   Encoder (32->64->128->256->512) ─── Multi-Scale Skips (Concat) ──> Decoder
   └─ Residual Basic Blocks (res_block=True)                            ├─ Level 3 (30x30)  -> Deep Supervision Head 3
                                                                        ├─ Level 2 (60x60)  -> Deep Supervision Head 2
                                                                        ├─ Level 1 (120x120)-> Deep Supervision Head 1
                                                                        └─ Output  (240x240)-> Full Resolution Head 0

3. ViT / JEPA Downstream Segmentation Model:
   Encoder: ViT-S/16 (225 patches, 384-dim, 8 Pre-LN Blocks with internal residual shortcuts)
     │
     ▼ (Pure Bottleneck Token Grid [B, 384, 15, 15] - NO ENCODER SKIP CONNECTIONS)
   Decoder: 4-Stage Transposed Convolution Upsampler
     ├─ Stage 1: ConvTranspose2d (384 -> 192, k=2, s=2) + GroupNorm(16) + GELU  [30x30]
     ├─ Stage 2: ConvTranspose2d (192 -> 96,  k=2, s=2) + GroupNorm(8)  + GELU  [60x60]
     ├─ Stage 3: ConvTranspose2d (96  -> 48,  k=2, s=2) + GroupNorm(4)  + GELU  [120x120]
     ├─ Stage 4: ConvTranspose2d (48  -> 24,  k=2, s=2) + GroupNorm(4)  + GELU  [240x240]
     └─ Projection: Conv2d (24 -> 1, k=1)                                       [240x240]
```

#### Mechanistic Scientific Finding
- **Full Data ($100\%$ Labels)**: nnU-Net achieves superior boundary precision ($\text{HD95} = 3.34\,\text{px}$ vs $10.75\,\text{px}$) because early convolutional skip connections bypass the bottleneck and directly transfer high-resolution edge details to the decoder.
- **Extreme Scarcity ($1\%$ Labels / $63$ Slices)**: Early convolutional layers fail to learn semantic abstraction and overfit to high-frequency noise, causing nnU-Net to degrade to $0.3847$ Dice and UNet to collapse to $0.0421$. In contrast, JEPA pre-training forces the ViT bottleneck to encode rich global tumor semantics. Even without skip connections, SigReg JEPA achieves **$0.5933$ Dice** ($+54.2\%$ relative improvement over nnU-Net).

---

### 3.2 Stochastic Randomization Schemes

1. **JEPAMaskingTransform**:
   - Total patches: $15 \times 15 = 225$ patches from $240 \times 240$ slices.
   - Target blocks: $K = 4$ blocks of fixed spatial dimension $5 \times 5 = 25$ patches.
   - Context block: Candidate block of $14 \times 14 = 196$ patches.
   - Non-overlapping guarantee: Context patches overlapping with any target block are strictly removed.
   - Batch collation replenishment: Non-target grid patches are randomly sampled to guarantee an exact, uniform count of $N_{\text{ctx}} = 96$ context tokens across all samples in the batch.
2. **Hyperspherical Projection Sketching**:
   - SigReg & VisReg: $M = 256$ projection vectors $u_m \in \mathbb{S}^{127}$ sampled uniformly from the unit hypersphere ($u \sim \mathcal{N}(0, \mathbf{I})$, $u_m = u / \|u\|_2$).
3. **Data Augmentation Randomization**:
   - Random Horizontal Flip ($p = 0.5$).
   - Random Vertical Flip ($p = 0.5$).
   - Random Affine Rotation ($\theta \sim \mathcal{U}(-17.2^\circ, +17.2^\circ)$, bilinear for images, nearest-neighbor for masks).
   - Random Modality Dropout ($p_{\text{drop}} = 0.25$) with non-empty active fallback.
4. **Deterministic Seeding & Stratification**:
   - Global seed 42 fixed across PyTorch CPU, PyTorch MPS/CUDA, NumPy, Python `random`.
   - Patient-level volume partitioning stratified by tumor volume bins to prevent patient leakage.

---

## 4. Notebook Workflow Architecture

Both [`notebooks/colab_runner.ipynb`](file:///Users/hanriman/Documents/master/thesis_2d/notebooks/colab_runner.ipynb) and [`notebooks/kaggle_runner.ipynb`](file:///Users/hanriman/Documents/master/thesis_2d/notebooks/kaggle_runner.ipynb) were streamlined:
- **Removed Code Zip Extraction**: Removed `thesis_2d_code.zip` extraction logic. Both notebooks now exclusively and cleanly clone from GitHub:
  ```python
  REPO_URL = "https://github.com/hanriman/2d_mri_brats_2024_segmentation.git"
  work_dir = Path("/kaggle/working/thesis_2d") # or /content/thesis_2d

  if not (work_dir / "src").exists():
      !git clone {REPO_URL} {work_dir}
  else:
      !git -C {work_dir} pull

  !pip install -q -e {work_dir}
  ```
- **Consistent CLI Arguments**: Verified that all script invocations across pre-training, baseline training, downstream fine-tuning, probing, and OOD benchmarks match the updated scripts.

---

## 5. Verification & Test Evidence

### 5.1 Unit Tests (`pytest`)
All 23 unit tests pass cleanly:
```text
tests/test_data.py::test_brats_dataset_loading PASSED                    [  4%]
tests/test_data.py::test_jepa_masking_transform PASSED                   [  8%]
tests/test_data.py::test_random_modality_dropout PASSED                  [ 13%]
tests/test_losses.py::test_dice_bce_loss PASSED                          [ 17%]
tests/test_losses.py::test_ijepa_loss PASSED                             [ 21%]
tests/test_losses.py::test_sigreg_loss PASSED                            [ 26%]
tests/test_losses.py::test_visreg_loss PASSED                            [ 30%]
tests/test_losses.py::test_epps_pulley_gaussianity PASSED                [ 34%]
tests/test_losses.py::test_deep_supervision_loss PASSED                  [ 39%]
tests/test_losses.py::test_ijepa_empty_fallback PASSED                   [ 43%]
tests/test_losses.py::test_sigreg_device_transfer PASSED                 [ 47%]
tests/test_losses.py::test_visreg_scale_shape_decoupling PASSED          [ 52%]
tests/test_metrics.py::test_surface_point_extraction PASSED              [ 56%]
tests/test_metrics.py::test_hd95_identical_and_empty PASSED              [ 60%]
tests/test_metrics.py::test_segmentation_metrics PASSED                  [ 65%]
tests/test_models.py::test_unet_shapes PASSED                            [ 69%]
tests/test_models.py::test_vit_encoder_shapes PASSED                     [ 73%]
tests/test_models.py::test_ijepa_forward PASSED                          [ 78%]
tests/test_models.py::test_sigreg_and_visreg_forward PASSED              [ 82%]
tests/test_models.py::test_jepa_segmentation_model PASSED                [ 86%]
tests/test_models.py::test_nnunet_forward PASSED                         [ 91%]
tests/test_models.py::test_jepa_predictor_heterogeneous_shapes PASSED    [ 95%]
tests/test_models.py::test_visreg_jepa_projected_tokens PASSED           [100%]

======================== 23 passed, 2 warnings in 1.96s ========================
```

### 5.2 End-to-End Pipeline Smoke Tests
All six CLI pipelines executed successfully with dummy batch data:
- `scripts/train_unet.py` $\implies$ **SUCCESS**
- `scripts/train_nnunet.py` $\implies$ **SUCCESS**
- `scripts/train_jepa.py --model_type visreg_jepa` $\implies$ **SUCCESS**
- `scripts/train_jepa.py --model_type sigreg_jepa` $\implies$ **SUCCESS**
- `scripts/train_jepa.py --model_type ijepa` $\implies$ **SUCCESS**
- `scripts/train_downstream.py` $\implies$ **SUCCESS**

### 5.3 Scientific Manuscript Compilation
- [`paper/latex/extended_main.tex`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.tex) compiled with BibTeX and pdfLaTeX into an 11-page PDF document ([`paper/latex/extended_main.pdf`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.pdf)) with zero errors, zero undefined citations, and resolved references.
