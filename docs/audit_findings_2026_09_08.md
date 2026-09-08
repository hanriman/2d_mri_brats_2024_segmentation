# Comprehensive Codebase, Mathematical Theory, and Implementation Audit Report

**Date:** September 8, 2026  
**Time:** 23:36 CEST (UTC+2)  
**Repository:** `thesis_2d`  
**Auditor:** AI Research Assistant (Antigravity)  
**Focus:** Self-Supervised Joint-Embedding Predictive Architectures (I-JEPA, SigReg JEPA / LeJEPA, VisReg JEPA) and Supervised Baselines (2D UNet, 2D nnU-Net with Deep Supervision) on 2D Multi-Modal BraTS 2024 Glioma MRI (`BraTS GLI`) and Meningioma OOD (`BraTS-MEN-RT`).

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Mathematical Theory & Reference Validation](#2-mathematical-theory--reference-validation)
   - [2.1 I-JEPA (Assran et al., CVPR 2023)](#21-i-jepa-assran-et-al-cvpr-2023)
   - [2.2 SigReg JEPA / LeJEPA (Balestriero & LeCun, 2025)](#22-sigreg-jepa--lejepa-balestriero--lecun-2025)
   - [2.3 VisReg JEPA (Wu, Balestriero, Levine, 2026)](#23-visreg-jepa-wu-balestriero-levine-2026)
   - [2.4 Supervised Baselines (2D UNet and 2D nnU-Net)](#24-supervised-baselines-2d-unet-and-2d-nnu-net)
3. [Bugs and Misimplementations](#3-bugs-and-misimplementations)
   - [3.1 Critical Bug: Volumetric Metric Key Mismatch (Silent NaN / N/A)](#31-critical-bug-volumetric-metric-key-mismatch-silent-nan--na)
   - [3.2 Mathematical Discrepancy: VISReg Sliced-Wasserstein Distance (W1 vs W2^2)](#32-mathematical-discrepancy-visreg-sliced-wasserstein-distance-w1-vs-w22)
   - [3.3 Manuscript Inconsistency: ViT Encoder Layer Count](#33-manuscript-inconsistency-vit-encoder-layer-count)
   - [3.4 CLI Argument Inconsistency in Master Evaluation Script](#34-cli-argument-inconsistency-in-master-evaluation-script)
4. [Medical Imaging & Data Pipeline Verification](#4-medical-imaging--data-pipeline-verification)
   - [4.1 Patient-Level Isolation vs. Data Leakage](#41-patient-level-isolation-vs-data-leakage)
   - [4.2 Multi-Modal Z-Score Normalization](#42-multi-modal-z-score-normalization)
   - [4.3 Random Modality Dropout with Guaranteed Non-Empty Fallback](#43-random-modality-dropout-with-guaranteed-non-empty-fallback)
   - [4.4 2D-to-3D Reconstruction on Sparse Slices](#44-2d-to-3d-reconstruction-on-sparse-slices)
5. [Code Quality, Dead Code & Numerical Stability](#5-code-quality-dead-code--numerical-stability)
6. [Actionable Remediation Plan](#6-actionable-remediation-plan)

---

## 1. Executive Summary

A comprehensive scientific, theoretical, and architectural audit was performed on the repository. The objective was to evaluate whether the implementation faithfully captures the mathematical formulations in published papers, identify bugs, detect dead or obsolete code, and assess medical imaging data validity.

### Overall Assessment
- **Mathematical Foundations:** Strong alignment with published literature. Asymmetric LayerNorm target scaling in I-JEPA, analytical characteristic function quadrature in LeJEPA / SIGReg, and decoupled scale-shape regularization in VISReg are correctly formalized.
- **Data Engineering:** Zero data leakage. Slices are strictly partitioned by patient identity (`patient_id`). Multi-modal handling adheres to clinical standards.
- **Identified Deficiencies:** A dictionary key mismatch in metric reporting silently produced `NaN` / `N/A` for all 3D patient volume evaluations across evaluation scripts. Additionally, VISReg's SWD loss uses $W_1$ (L1) rather than the official paper's $W_2^2$ (MSE).
- **Dead Code:** An obsolete directory [`sandbox_2d/`](file:///Users/hanriman/Documents/master/thesis_2d/sandbox_2d) and orphaned build logs (`texput.log`) were found.

---

## 2. Mathematical Theory & Reference Validation

### 2.1 I-JEPA (Assran et al., CVPR 2023)
*Reference: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023.*

$$\mathcal{L}_{\text{I-JEPA}} = \frac{1}{M} \sum_{k=1}^M \mathcal{D}\left(\hat{\mathbf{s}}_y^{(k)}, \, \text{LayerNorm}(\mathbf{s}_y^{(k)})\right)$$

1. **Target Normalization:** Implemented in [`src/brats_jepa/losses/ijepa_loss.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/ijepa_loss.py#L52-L64). Targets $\mathbf{s}_y$ from the EMA teacher are LayerNormed (`F.layer_norm`), while online predictions $\hat{\mathbf{s}}_y$ remain unnormalized. This preserves coordinate scale gradients in the predictor.
2. **Exponential Moving Average (EMA) Momentum Schedule:** Implemented in [`scripts/train_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/train_jepa.py#L187-L188):
   $$m_t = 1 - (1 - m_0) \cdot \frac{1}{2} \left(1 + \cos\left(\frac{\pi t}{T}\right)\right), \quad m_0 = 0.996$$
3. **Teacher Invariance:** Target encoder is maintained in `eval()` mode (`self.target_encoder.eval()`) during training, deactivating dropout and ensuring deterministic targets.
4. **Patch Masking:** [`src/brats_jepa/data/transforms.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/data/transforms.py#L103-L179) samples 4 target blocks of $5 \times 5$ patches and guarantees a constant $N_{\text{ctx}} = 96$ context patches per slice. This ensures regular tensor shapes for batch collation without ragged padding.

---

### 2.2 SigReg JEPA / LeJEPA (Balestriero & LeCun, 2025)
*Reference: Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics", arXiv:2511.08544, 2025.*

$$\mathcal{T}_{\text{EP}} = N \int_0^{t_{\max}} \left| \hat{\phi}_N(t) - \phi_0(t) \right|^2 w(t) \, dt$$

1. **Cramér–Wold Device:** High-dimensional features are projected onto $M = 256$ random unit directions sampled uniformly on $\mathbb{S}^{D-1}$ via `F.normalize(torch.randn(...), p=2, dim=0)`.
2. **Numerical Quadrature:** In [`src/brats_jepa/losses/sigreg_loss.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/sigreg_loss.py#L50-L77), trapezoidal quadrature over $t \in [0, 3.0]$ with $K = 17$ knots correctly assigns interior weights $2\,dt$ and boundary weights $dt$, weighted by the Gaussian characteristic function $\phi_0(t) = e^{-t^2/2}$.
3. **Gradient Scaling:** Multiplying the statistic by $N$ cancels the $\frac{1}{N}$ factor in $\frac{\partial \hat{\phi}}{\partial z}$, producing $\mathcal{O}(1)$ per-sample gradients.
4. **Architectural Trade-off (Projector MLP):** [`src/brats_jepa/models/sigreg_jepa.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/sigreg_jepa.py#L69-L74) passes context tokens through a 2-layer MLP (`384 -> 1024 -> 128`) before computing $\mathcal{T}_{\text{EP}}$.
   - *Analysis:* While LeJEPA was conceptually conceived to operate without projection heads, introducing an MLP projector acts as an information buffer. It prevents destructive Gaussian distortion on the ViT backbone ($D=384$) while preventing collapse in the latent space ($D=128$). This explains why backbone tokens retain higher average cosine similarity ($0.7531$).

---

### 2.3 VisReg JEPA (Wu, Balestriero, Levine, 2026)
*Reference: Wu, Balestriero, & Levine, "VISReg: Visual Representation Learning via Regularization", arXiv:2606.02572, 2026.*

$$\mathcal{L}_{\text{VisReg}} = \mathcal{L}_{\text{JEPA}} + \lambda_{\text{center}} \mathcal{L}_{\text{center}} + \lambda_{\text{scale}} \mathcal{L}_{\text{scale}} + \lambda_{\text{shape}} \mathcal{L}_{\text{shape}}$$

1. **Center Regularization:** $\mathcal{L}_{\text{center}} = \frac{1}{D} \|\boldsymbol{\mu}_z\|_2^2$ centers representations at the origin.
2. **Scale Regularization:** $\mathcal{L}_{\text{scale}} = \frac{1}{D} \sum_{d=1}^D (1 - \sigma_d)^2$ enforces unit coordinate variance.
3. **Decoupled Shape Regularization:** Features are centered and normalized with stop-gradient on standard deviation ($\tilde{z} = (z - \mu) / (\text{sg}(\sigma) + \epsilon)$), decoupling shape from scale dynamics.
4. **Mathematical Discrepancy:** In [`src/brats_jepa/losses/visreg_loss.py:L110`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/visreg_loss.py#L110), the implementation uses $W_1$ (L1 loss) rather than the official paper's $W_2^2$ (MSE loss). See [Section 3.2](#32-mathematical-discrepancy-visreg-sliced-wasserstein-distance-w1-vs-w22).

---

### 2.4 Supervised Baselines (2D UNet and 2D nnU-Net)
1. **2D ResUNet ([`src/brats_jepa/models/unet.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/unet.py)):** 5-stage architecture with Instance Normalization and residual units. Skip connections preserve spatial tumor margin details.
2. **2D nnU-Net ([`src/brats_jepa/models/nnunet.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/nnunet.py)):** MONAI `DynUNet` with residual blocks, LeakyReLU ($\alpha = 0.01$), Instance Normalization, and 3 auxiliary deep supervision heads.
3. **Normalized Deep Supervision:** [`src/brats_jepa/losses/deep_supervision_loss.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/deep_supervision_loss.py) applies normalized exponential decay weights ($w_s = \frac{2^{-s}}{\sum 2^{-j}}$) with nearest-neighbor target interpolation, maintaining consistent gradient scaling across multi-scale outputs.

---

## 3. Bugs and Misimplementations

### 3.1 Critical Bug: Volumetric Metric Key Mismatch (Silent NaN / N/A)
* **Severity:** High (Functional & Reporting)
* **Affected Files:**
  - [`scripts/evaluate.py:L129-L130`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate.py#L129-L130)
  - [`scripts/evaluate_low_data.py:L119-L120`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_low_data.py#L119-L120)
  - [`scripts/evaluate_ood.py:L225-L226`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_ood.py#L225-L226)
  - [`scripts/evaluate_men_rt_ood.py:L254-L255`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_men_rt_ood.py#L254-L255)
* **Description:**
  [`compute_patient_volume_metrics`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/metrics/segmentation_metrics.py#L410-L418) returns dictionary keys `"dice_3d"` and `"hd95_3d"`. However, all evaluation scripts query `"patient_3d_dice_mean"` and `"patient_3d_hd95_mean"`.
* **Consequence:**
  The `.get()` calls return `float("nan")`. As a result, [`evaluate.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate.py) outputs `"N/A"` in benchmark tables, and [`evaluate_low_data.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_low_data.py) saves `NaN` values across all low-data CSV summaries.
* **Fix:** Provide alias keys in [`compute_patient_volume_metrics`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/metrics/segmentation_metrics.py):
  ```python
  "dice_3d": float(np.mean(p_dices)),
  "hd95_3d": float(np.mean(p_hd95s)),
  "patient_3d_dice_mean": float(np.mean(p_dices)),
  "patient_3d_hd95_mean": float(np.mean(p_hd95s)),
  ```

---

### 3.2 Mathematical Discrepancy: VISReg Sliced-Wasserstein Distance ($W_1$ vs $W_2^2$)
* **Severity:** Medium (Theoretical Conformance)
* **Affected File:** [`src/brats_jepa/losses/visreg_loss.py:L110`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/visreg_loss.py#L110)
* **Description:**
  [`VisRegLoss._sliced_wasserstein_distance`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/visreg_loss.py#L87-L111) computes L1 loss ($W_1$):
  ```python
  swd = F.l1_loss(sorted_proj, gaussian_quantiles.unsqueeze(-1).expand_as(sorted_proj))
  ```
  The official VISReg paper (*Wu, Balestriero, Levine, 2026*) defines the shape penalty via squared differences ($W_2^2$):
  ```python
  swd = F.mse_loss(sorted_proj, gaussian_quantiles.unsqueeze(-1).expand_as(sorted_proj))
  ```
* **Consequence:** $W_1$ delivers constant magnitude gradients $(\pm 1)$ regardless of how far samples are in the distribution tails. Squared $W_2^2$ applies quadratic penalties to extreme tail deviations, promoting tighter Gaussian fit.
* **Fix:** Update default shape distance to `F.mse_loss`, with an optional parameter for `l1`.

---

### 3.3 Manuscript Inconsistency: ViT Encoder Layer Count
* **Severity:** Low (Documentation & Paper)
* **Affected File:** [`paper/latex/extended_main.tex:L380`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.tex#L380)
* **Description:** Line 380 states: *"the patch token grid from the final ViT layer $\mathbf{Z}^{(12)} \in \mathbb{R}^{B \times 225 \times 384}$"*. However, Line 110 and the codebase [`vision_transformer.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/models/vision_transformer.py#L70) specify encoder depth as $L=8$.
* **Fix:** Update $\mathbf{Z}^{(12)}$ to $\mathbf{Z}^{(8)}$ in the LaTeX manuscript.

---

### 3.4 CLI Argument Inconsistency in Master Evaluation Script
* **Severity:** Low (Usability)
* **Affected File:** [`scripts/evaluate.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate.py)
* **Description:** Unlike [`train_downstream.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/train_downstream.py) and [`evaluate_low_data.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate_low_data.py), [`scripts/evaluate.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate.py) does not expose `--decoder_type` and `--encoder_source` CLI flags, defaulting to bottleneck evaluation.
* **Fix:** Add argument parsing support for `--decoder_type` and `--encoder_source`.

---

## 4. Medical Imaging & Data Pipeline Verification

### 4.1 Patient-Level Isolation vs. Data Leakage
- **Validation:** [`scripts/prepare_data.py:L224-L260`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/prepare_data.py#L224-L260) partitions the dataset using stratified splitting over unique `patient_id`s.
- **Verdict:** **Zero Data Leakage.** Slices from the same 3D patient volume never cross train/val/test partitions.

### 4.2 Multi-Modal Z-Score Normalization
- **Validation:** [`prepare_data.py:L33-L42`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/prepare_data.py#L33-L42) computes mean and standard deviation strictly over non-zero brain voxels (`volume > 0`). Background skull-stripped air voxels (0) are excluded, preventing background volume from biasing tissue intensity statistics.

### 4.3 Random Modality Dropout with Guaranteed Non-Empty Fallback
- **Validation:** [`src/brats_jepa/data/transforms.py:L28-L74`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/data/transforms.py#L28-L74) drops modalities independently ($p_{\text{drop}} = 0.25$). If all 4 channels are zeroed, a random channel is activated:
  ```python
  fallback = torch.zeros_like(mask).scatter_(1, random_channel, 1.0)
  mask = torch.where(all_zero, fallback, mask)
  ```
  This prevents models from training on degenerate all-zero inputs.

### 4.4 2D-to-3D Reconstruction on Sparse Slices
- **Observation:** In [`compute_patient_volume_metrics`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/metrics/segmentation_metrics.py#L382-L393), 3D volume reconstruction stacks the 5 extracted slices into a 3D grid:
  ```python
  vol_p = np.zeros((max_z + 1, h, w), dtype=bool)
  ```
  Because only 5 discontinuous axial slices are sampled with large physical gaps ($\Delta z \ge 10$), applying a 3D $3 \times 3 \times 3$ morphological structuring element classifies all non-zero slice pixels as surface points (due to empty $z-1$ and $z+1$ planes).
- **Recommendation:** Extract 2D boundary contours slice-by-slice before running the 3D KDTree query.

---

## 5. Code Quality, Dead Code & Numerical Stability

1. **Obsolete Directory (`sandbox_2d/`):**
   The [`sandbox_2d/`](file:///Users/hanriman/Documents/master/thesis_2d/sandbox_2d) directory contains early prototype code (`dataset.py`, `model.py`, `prepare_data.py`, `train.py`). It is not referenced anywhere in the core package and can be safely deleted.
2. **Build Artifacts (`texput.log`):**
   An orphaned LaTeX error log exists in the repository root. It should be removed and added to `.gitignore`.
3. **AMP Numerical Stability in Loss Functions:**
   In [`sigreg_loss.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/sigreg_loss.py) and [`visreg_loss.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/visreg_loss.py), trigonometric functions (`cos`, `sin`) and inverse error functions (`erfinv`) are evaluated. While `gaussian_quantiles` are computed in `float32`, casting projections to `float32` prior to empirical characteristic function evaluation prevents potential underflow under CUDA `float16`.

---

## 6. Actionable Remediation Plan

| Priority | Component | Issue | Action |
| :--- | :--- | :--- | :--- |
| **P0** | [`segmentation_metrics.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/metrics/segmentation_metrics.py#L410-L418) | Metric key mismatch causing `NaN`/`N/A` | Add `patient_3d_dice_mean` and `patient_3d_hd95_mean` alias keys to `compute_patient_volume_metrics`. |
| **P1** | [`visreg_loss.py`](file:///Users/hanriman/Documents/master/thesis_2d/src/brats_jepa/losses/visreg_loss.py#L110) | SWD uses L1 ($W_1$) instead of MSE ($W_2^2$) | Change default SWD loss to `F.mse_loss` per Wu et al. (2026). |
| **P1** | [`extended_main.tex`](file:///Users/hanriman/Documents/master/thesis_2d/paper/latex/extended_main.tex#L380) | Typo: $\mathbf{Z}^{(12)}$ instead of $\mathbf{Z}^{(8)}$ | Correct equation text to reflect 8 Transformer layers. |
| **P2** | [`scripts/evaluate.py`](file:///Users/hanriman/Documents/master/thesis_2d/scripts/evaluate.py) | Missing CLI arguments | Add `--decoder_type` and `--encoder_source` arguments. |
| **P2** | [`sandbox_2d/`](file:///Users/hanriman/Documents/master/thesis_2d/sandbox_2d), `texput.log` | Dead code and build artifact | Remove directory and log file; update `.gitignore`. |
