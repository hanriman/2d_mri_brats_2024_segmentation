# Research Proposal: Representation Regularization in Joint-Embedding Predictive Architectures for Multi-Modal Brain Glioma MRI

## 1. Executive Summary & One-Sentence Summary

> **We investigate whether feature variance and spatial visual regularizations (SigReg & VisReg) prevent representation collapse in Joint-Embedding Predictive Architectures (I-JEPA) trained on 2D multi-modal BraTS glioma MRI, thereby improving downstream segmentation accuracy compared to standard supervised UNet baselines.**

---

## 2. Research Problem & Context

Automated brain tumor segmentation from multi-modal Magnetic Resonance Imaging (MRI) — comprising T1-weighted (T1), post-contrast T1-weighted (T1c), T2-weighted (T2), and Fluid Attenuated Inversion Recovery (FLAIR) modalities — is crucial for neuro-oncology diagnosis, radiation treatment planning, and monitoring glioma progression.

Supervised deep neural networks such as UNet achieve high accuracy when full pixel-wise annotations are available. However, manual annotation of 3D brain tumors is labor-intensive, expertise-demanding, and subject to inter-observer variability. Self-Supervised Learning (SSL) offers a compelling strategy to pre-train representation encoders on unannotated MRI volumes.

---

## 3. Existing Knowledge & Research Gap

### Existing Knowledge
Joint-Embedding Predictive Architectures (I-JEPA) pre-train vision transformers in latent space by predicting target patch representations from context patches without pixel-level reconstruction. This avoids high-frequency pixel noise and focuses representations on high-level semantic features.

### Research Gap
Standard I-JEPA relies on Exponential Moving Average (EMA) teacher updating to prevent representation collapse (where all patch embeddings collapse to a constant vector). In complex multi-modal medical imaging (4 channels per slice), EMA alone may not guarantee high effective feature rank or optimal spatial feature dispersion. It remains uncertain whether representation regularizations—specifically:
1. **SigReg JEPA** (Sigmoid/Variance-Covariance feature regularizer), and
2. **VisReg JEPA** (Visual spatial contrast regularizer)

can systematically enhance feature rank, prevent latent collapse, and translate to superior downstream glioma segmentation over standard I-JEPA and baseline supervised UNet.

---

## 4. Research Questions & Hypotheses

* **RQ1**: How do I-JEPA, SigReg JEPA, and VisReg JEPA compare in terms of latent representation richness (effective rank, feature variance, pairwise cosine similarity) when pre-trained on 2D multi-modal BraTS glioma slices?
* **RQ2**: Does SigReg or VisReg regularization mitigate latent feature collapse more effectively than standard EMA-only I-JEPA?
* **RQ3**: Does self-supervised pre-training with JEPA variants improve downstream segmentation Dice score compared to randomly initialized supervised UNet?

### Hypotheses
* **H1**: SigReg JEPA will maintain significantly higher effective feature rank and lower average pairwise cosine similarity compared to standard I-JEPA.
* **H2**: Pre-trained representations with higher effective rank will yield improved downstream fine-tuning Dice scores on glioma segmentation.

---

## 5. Methodology & Models Under Study

### Target Dataset
* **BraTS 2D Slices**: Extracted 4-channel axial slices [T1, T1c, T2, FLAIR] with dimension $240 \times 240$ from the BraTS GLI dataset, preprocessed with non-zero Z-score intensity normalization.

### Model Architectures
1. **UNet Baseline**: 2D Residual UNet (MONAI) trained fully supervised using combined Dice + BCE loss.
2. **I-JEPA**: Standard Image Joint-Embedding Predictive Architecture with Vision Transformer encoder, predictor, and Exponential Moving Average (EMA) teacher target encoder.
3. **SigReg JEPA (LeJEPA / SIGReg)**: **Heuristic-free single-encoder architecture** without an EMA teacher encoder. Representation collapse is prevented analytically via Sketched Isotropic Gaussian / Variance-Covariance feature regularization.
4. **VisReg JEPA (VISReg)**: **Heuristic-free single-encoder architecture** without an EMA teacher encoder. Representation collapse is prevented via Variance-Invariance-Sketching / spatial patch visual contrast regularization.

---

## 6. Evaluation Matrix

| Model | SSL Pre-training Loss | Effective Rank | Avg Cosine Sim | Downstream Val Dice | Test Dice |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UNet (Baseline)** | N/A (Supervised) | N/A | N/A | TBD | TBD |
| **I-JEPA** | Predictor $L_1$ | TBD | TBD | TBD | TBD |
| **SigReg JEPA** | $L_1$ + SigReg (Var+Cov) | TBD | TBD | TBD | TBD |
| **VisReg JEPA** | $L_1$ + VisReg (Spatial Var)| TBD | TBD | TBD | TBD |

---

## 7. Execution Plan & Artifacts

1. **Phase 1: Environment & Pipeline Preparation** (`uv`, `src/brats_jepa/`)
2. **Phase 2: Self-Supervised Pre-training Sweeps** (`scripts/train_jepa.py`)
3. **Phase 3: Supervised & Fine-tuning Benchmark** (`scripts/train_unet.py`)
4. **Phase 4: Probing & Representation Analysis** (`scripts/evaluate.py`)
5. **Phase 5: Publication Figures & Manuscript** (`scripts/generate_figures.py`, `paper/latex/`)
