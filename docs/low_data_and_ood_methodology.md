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

---

## 5. Evaluation Metric Integrity & Multi-Level Aggregation Protocol

A central finding of this thesis is that standard 2D slice-wise evaluation metrics can create catastrophic statistical artifacts if healthy slices and zero-division guards are not rigorously managed.

### 5.1 Zero-Division Guard Formulation
For any binary predicted segmentation mask $P$ and ground-truth mask $Y$:

$$\text{Dice}(P, Y) = \frac{2 |P \cap Y| + \epsilon}{|P| + |Y| + \epsilon}, \qquad \text{IoU}(P, Y) = \frac{|P \cap Y| + \epsilon}{|P \cup Y| + \epsilon}$$

- **True Negative Concordance**: When both $P = \emptyset$ and $Y = \emptyset$, $\frac{\epsilon}{\epsilon} = 1.0$ (reflecting perfect identification of healthy tissue).
- **False Concordance Guard**: When either $|P| > 0$ and $|Y| = 0$, or $|P| = 0$ and $|Y| > 0$, $|P \cap Y| = 0$, returning $0.0$.

### 5.2 The Slice-Wise Stratification Mandate ($\text{Dice}_{\text{all}}$ vs. $\text{Dice}_{\text{tumor}}$)
Axial MRI volumes contain extensive healthy slices at the cranial base and vertex where no tumor pathology is present.
In the BraTS GLI evaluation set ($N=633$ axial slices):
- **$294$ slices** are completely empty of tumor ($Y = \emptyset$).
- **$339$ slices** contain active tumor pathology ($Y \ne \emptyset$).

Macro-averaging over all slices conflates non-tumor identification with active lesion segmentation:
$$\text{Dice}_{\text{all}} = \frac{1}{N_{\text{all}}} \sum_{i=1}^{N_{\text{all}}} \text{Dice}(P_i, Y_i)$$

To isolate true tumor delineation quality, we report stratified tumor-positive slice metrics:
$$\text{Dice}_{\text{tumor}} = \frac{1}{N_{\text{tumor}}} \sum_{i \in \{j \mid |Y_j| > 0\}} \text{Dice}(P_i, Y_i)$$

### 5.3 Diagnostic Case Study: The $\text{Dice} = \text{IoU} = 0.4647577$ Mathematical Proof
In preliminary low-data benchmarks, the 10% labeled UNet baseline reported:
$$\text{Dice}_{\text{all}} = 0.4647577, \qquad \text{IoU}_{\text{all}} = 0.4647577$$

**Mathematical Proof of Collapse**:
1. By set theory, $\text{Dice} = \frac{2 \text{IoU}}{1 + \text{IoU}}$. For any non-trivial overlapping sets with $0 < \text{IoU} < 1$, $\text{Dice} > \text{IoU}$ strictly holds.
2. Equality $\text{Dice} = \text{IoU}$ is strictly impossible unless every single evaluated slice yields binary concordance: $\text{Dice}_i, \text{IoU}_i \in \{0.0, 1.0\}$.
3. If a model completely collapses and outputs all zeros ($\hat{Y} \equiv 0$):
   - On the $294$ healthy slices ($Y = \emptyset$): $P = \emptyset \implies \text{Dice} = 1.0, \text{IoU} = 1.0$.
   - On the $339$ tumor slices ($Y \ne \emptyset$): $P = \emptyset \implies \text{Dice} = 0.0, \text{IoU} = 0.0$.
   - Macro-average over $633$ slices:
     $$\text{Dice}_{\text{all}} = \frac{294 \times 1.0 + 339 \times 0.0}{633} = \frac{294}{633} \approx 0.4647709 \dots$$
4. Unstratified slice-wise metrics thus masked complete optimization failure as a seemingly respectable score ($\sim 0.465$). Under tumor-stratified evaluation, this collapse is immediately exposed:
   $$\text{Dice}_{\text{tumor}} = \mathbf{0.0000}$$

### 5.4 3D Patient-Level Volumetric Aggregation ($\text{Dice}_{\text{3D}}$ & $\text{HD95}_{\text{3D}}$)
To align with official clinical benchmark protocols (BraTS Challenge, MICCAI), 2D slice predictions are re-assembled into contiguous 3D patient volumes:
$$\mathbf{V}_{\text{pred}}^{(p)} \in \{0, 1\}^{D_p \times 240 \times 240}, \qquad \mathbf{V}_{\text{gt}}^{(p)} \in \{0, 1\}^{D_p \times 240 \times 240}$$

Global 3D intersection and union are computed over all voxels in the patient volume before division:
$$\text{Dice}_{\text{3D}}^{(p)} = \frac{2 \sum_{v} \mathbf{V}_{\text{pred}}^{(p)}(v) \mathbf{V}_{\text{gt}}^{(p)}(v)}{\sum_v \mathbf{V}_{\text{pred}}^{(p)}(v) + \sum_v \mathbf{V}_{\text{gt}}^{(p)}(v)}$$

The 95th-percentile Hausdorff Distance ($\text{HD95}_{\text{3D}}$) is evaluated on the 3D surface boundary point sets $\partial V_{\text{pred}}$ and $\partial V_{\text{gt}}$:
$$\text{HD95}_{\text{3D}} = \max\left(P_{95\%} \min_{y \in \partial V_{\text{gt}}} \|x - y\|_2, \; P_{95\%} \min_{x \in \partial V_{\text{pred}}} \|x - y\|_2\right)$$

---

## 6. Execution & CLI Command Guide

```bash
# Step 1: Pre-train JEPA architectures (I-JEPA, SigReg, or VisReg)
uv run python scripts/train_jepa.py --model_type visreg_jepa --epochs 50 --batch_size 32 --amp

# Step 2: Downstream Fine-Tuning with Hierarchical Multi-Scale FPN Decoder
uv run python scripts/train_downstream.py --model_type visreg_jepa \
    --pretrained_ckpt outputs/checkpoints/best_visreg_jepa.pt \
    --decoder_type multiscale --epochs 30 --batch_size 16 --amp

# Step 3: Run Low-Data Label Efficiency Benchmark with Tumor Stratification & 3D Metrics
uv run python scripts/evaluate_low_data.py --epochs 30 --decoder_type multiscale \
    --evaluate_3d --exp_version v2_low_data_efficiency

# Step 4: Run Synthetic OOD Scanner Generalization Benchmark
uv run python scripts/evaluate_ood.py --decoder_type multiscale --exp_version v3_ood_generalization

# Step 5: Run BraTS-MEN-RT Cross-Pathology OOD Benchmark
uv run python scripts/evaluate_men_rt_ood.py --exp_version v4_men_rt_ood

# Step 6: Generate Publication Figures & Compile LaTeX Paper
uv run python scripts/generate_figures.py
cd paper/latex && pdflatex extended_main.tex && bibtex extended_main && pdflatex extended_main.tex
```

---

## 7. Versioned Experiments Map

| Version Tag | Experiment Description | Key Metric Additions | Output Location |
| :--- | :--- | :--- | :--- |
| **`v1_full_data_100pct`** | 100% Full-Data Baseline & SSL | $\text{Dice}_{\text{all}}$, $\text{Dice}_{\text{tumor}}$, $\text{Dice}_{\text{3D}}$, $\text{HD95}_{\text{3D}}$ | `outputs/experiments/v1_full_data_100pct/` |
| **`v2_low_data_efficiency`** | Low-Data Efficiency ($1\%$ to $100\%$) | Tumor-stratified + 3D patient volume metrics | `outputs/experiments/v2_low_data_efficiency/` |
| **`v3_ood_generalization`** | Synthetic OOD Scanner Shift | Rician noise ($\sigma=0.15$), $B_1$ bias field ($\alpha=0.35$) | `outputs/experiments/v3_ood_generalization/` |
| **`v4_men_rt_ood`** | Real-world Meningioma OOD | Zero-shot single T1c cross-pathology evaluation | `outputs/experiments/v4_men_rt_ood/` |
