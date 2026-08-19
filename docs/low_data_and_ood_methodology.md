# Low-Data Label Efficiency & Out-of-Distribution (OOD) Methodology Guide

This document provides a comprehensive technical specification of the experimental design, mathematical formulation, CLI commands, data flow, and evaluation metrics for evaluating **Low-Data Label Efficiency**, **Random Modality Dropout Training**, and **Out-of-Distribution (OOD) Scanner & Cross-Pathology Generalization** on pre-trained Joint-Embedding Predictive Architectures (JEPA) vs. supervised baselines (UNet and nnU-Net).

---

## 1. Clinical Motivation & Scientific Hypotheses

### 1.1 The Label Scarcity Bottleneck in Medical Imaging
In clinical neuro-oncology, acquiring pixel-wise 3D ground-truth tumor segmentations requires specialized neuroradiologist expertise and hours of manual annotation per volume. While fully supervised models (e.g., nnU-Net) perform exceptionally well when thousands of labeled slices are available, their performance degrades sharply when trained on small annotated cohorts.

* **Hypothesis 1 (Low-Data Transferability)**: Self-supervised **SigReg JEPA** pre-training learns domain-general anatomical representations without labels. When fine-tuned on extremely low-data regimes ($1\%$ or $5\%$ labeled data, i.e., $11$ or $56$ slices), SigReg JEPA will retain high feature quality and significantly outperform fully supervised baselines (UNet and nnU-Net), which overfit severely.

### 1.2 Missing Modality Resilience via Modality Dropout Training
Real-world MRI acquisitions frequently suffer from missing sequences (e.g. emergency acquisitions omitting T2/FLAIR or providing only post-contrast T1c).

* **Hypothesis 2 (Modality Dropout Generalization)**: Injecting **Random Modality Dropout** ($p_{\text{drop}} = 0.25$) during training randomly masks 1, 2, or 3 modality channels per slice. This prevents neural networks from co-depending on all 4 modalities simultaneously, forcing encoders to extract self-contained anatomical representations per sequence and boosting zero-shot cross-modality transfer.

### 1.3 Scanner Hardware & Protocol Shifts
MRI data acquired across different hospital sites, field strengths (e.g., 1.5T vs. 3.0T), or scanner vendors (Siemens, GE, Philips) exhibit variations in Signal-to-Noise Ratio (SNR) and magnetic field inhomogeneities.

* **Hypothesis 3 (Scanner Shift Resilience)**: Supervised convolutional feature maps overfit to high-frequency acquisition artifacts. Self-supervised JEPA latent embeddings will exhibit superior robustness to synthetic scanner noise and intensity bias fields.

### 1.4 Real-World Cross-Pathology & Missing-Modality Generalization
* **Hypothesis 4 (Cross-Pathology Transfer)**: Pre-trained JEPA encoders fine-tuned on Intra-axial Glioma (`BraTS GLI`) with Random Modality Dropout can transfer zero-shot to Extra-axial Meningioma (`BraTS-MEN-RT`) scans providing only single T1-contrast sequences.

---

## 2. Low-Data Label Efficiency & Modality Dropout Protocol

### 2.1 Subsampled Dataset Splits
From the full training dataset ($N_{\text{total}} = 1,134$ slices), we extract deterministic stratified subsets representing 6 label availability tiers:

$$\mathcal{D}_{\text{train}}^{(\text{frac})} \subset \mathcal{D}_{\text{train}}^{(1.0)}, \qquad \text{frac} \in \{0.01, 0.05, 0.10, 0.25, 0.50, 1.00\}$$

| Label Fraction ($\text{frac}$) | Sample Count ($N$) | Saved Checkpoint Tag | Clinical Scan Equivalent |
| :--- | :--- | :--- | :--- |
| **1%** | **11 slices** | `*_1pct.pt` | $\sim 1$ patient volume |
| **5%** | **56 slices** | `*_5pct.pt` | $\sim 5$ patient volumes |
| **10%** | **113 slices** | `*_10pct.pt` | $\sim 10$ patient volumes |
| **25%** | **283 slices** | `*_25pct.pt` | $\sim 25$ patient volumes |
| **50%** | **567 slices** | `*_50pct.pt` | $\sim 50$ patient volumes |
| **100%** | **1,134 slices** | `*_100pct.pt` | Full dataset |

### 2.2 Random Modality Dropout Implementation
During training steps, input slices $X \in \mathbb{R}^{B \times 4 \times 240 \times 240}$ pass through a Random Modality Dropout layer:

$$\mathbf{M}_{b, c} \sim \text{Bernoulli}(1 - p_{\text{drop}}), \qquad \hat{X}_{b, c, :, :} = X_{b, c, :, :} \odot \mathbf{M}_{b, c}$$

where $p_{\text{drop}} = 0.25$, and a fallback constraint guarantees that at least one modality channel remains active per slice.

---

## 3. Out-of-Distribution (OOD) Scanner Shift Protocol

We evaluate model robustness under two synthetic physical perturbations simulating scanner hardware variations:

### 3.1 Rician Noise Shift (Low SNR / 1.5T Scanner Simulation)
$$I_{\text{noisy}}(x, y) = \sqrt{\left(I(x, y) + \eta_1\right)^2 + \eta_2^2}, \qquad \eta_1, \eta_2 \sim \mathcal{N}(0, \sigma^2)$$
where $\sigma = 0.15$ introduces realistic 1.5T scanner noise.

### 3.2 $B_1$ Intensity Bias Field Shift (Coil Sensitivity Shift)
$$I_{\text{bias}}(x, y) = I(x, y) \cdot \left(1.0 + \alpha \cdot (x^2 + y^2)\right)$$
where $\alpha = 0.35$ induces smooth radial intensity decay from the image center.

---

## 4. Real-World Cross-Pathology Protocol (`BraTS-MEN-RT`)

### 4.1 Dataset Characteristics
- **Dataset**: `BraTS-MEN-RT` (Brain Tumor Meningioma Radiotherapy Challenge).
- **Cohort**: 571 patients ($90,723$ axial 2D slices).
- **Target**: Gross Tumor Volume (`gtv.nii.gz`).
- **Modality**: T1-post contrast sequence (`t1c.nii.gz`).

### 4.2 4-Channel Model Input Adaptation Strategies
1. **Strategy 1: Channel Replication**:
   $$X = [X_{\text{T1c}}, X_{\text{T1c}}, X_{\text{T1c}}, X_{\text{T1c}}]$$
2. **Strategy 2: Zero-Padding Missing Modalities**:
   $$X = [\mathbf{0}, X_{\text{T1c}}, \mathbf{0}, \mathbf{0}]$$

---

## 5. Execution & CLI Command Guide

```bash
# Step 1: Run Low-Data Label Efficiency Benchmark with Checkpoint Archiving
uv run python scripts/evaluate_low_data.py --epochs 30 --exp_version v2_low_data_efficiency

# Step 2: Run Synthetic OOD Scanner Generalization Benchmark
uv run python scripts/evaluate_ood.py --exp_version v3_ood_generalization

# Step 3: Run BraTS-MEN-RT Cross-Pathology OOD Benchmark
uv run python scripts/evaluate_men_rt_ood.py --exp_version v4_men_rt_ood

# Step 4: Generate Publication Figures & Compile LaTeX Paper
uv run python scripts/generate_figures.py
cd paper/latex && pdflatex main.tex && bibtex main && pdflatex main.tex
```

---

## 6. Versioned Experiments Map

| Version Tag | Experiment Description | Saved Checkpoints | Output Location |
| :--- | :--- | :--- | :--- |
| **`v1_full_data_100pct`** | 100% Full-Data Baseline & SSL | `best_*.pt` | `outputs/experiments/v1_full_data_100pct/` |
| **`v2_low_data_efficiency`** | Low-Data Efficiency ($1\%$ to $100\%$) | `*_1pct.pt`, `*_5pct.pt`, ... | `outputs/experiments/v2_low_data_efficiency/` |
| **`v3_ood_generalization`** | Synthetic OOD Scanner Shift | Evaluates pre-trained checkpoints | `outputs/experiments/v3_ood_generalization/` |
| **`v4_men_rt_ood`** | Real-world Meningioma OOD | Evaluates pre-trained checkpoints | `outputs/experiments/v4_men_rt_ood/` |
