# Self-Supervised Representation Learning on 2D BraTS Glioma MRI: Comparing I-JEPA, SigReg JEPA, VisReg JEPA, UNet, and nnU-Net

A production-grade, reproducible research repository for self-supervised representation learning, Low-Data Label Efficiency ($1\%$ to $100\%$ labels), Out-of-Distribution (OOD) Scanner Domain Generalization, and brain tumor segmentation on 2D multi-modal Glioma MRI (T1, T1c, T2, FLAIR).

---

## 1. Project Overview

This repository investigates Joint-Embedding Predictive Architectures (JEPA) for multi-modal brain MRI slice representation learning, comparing:
1. **I-JEPA**: Dual-encoder Image Joint-Embedding Predictive Architecture predicting target patch representations in latent space with Exponential Moving Average (EMA) teacher updates.
2. **SigReg JEPA** (LeJEPA / SIGReg): Heuristic-free single-encoder architecture regularized via variance hinge and covariance decorrelation penalties, eliminating momentum teacher updates.
3. **VisReg JEPA** (VISReg): Heuristic-free single-encoder architecture regularized via spatial patch feature variance contrast penalties.
4. **2D UNet Baseline**: Supervised 5-stage Residual UNet baseline for multi-modal tumor segmentation.
5. **2D nnU-Net Baseline**: State-of-the-art supervised baseline with residual encoder blocks, Instance Normalization, LeakyReLU, and multi-scale **Deep Supervision** heads.

---

## 2. Converged Benchmark Summary Table (Version 1: Full-Data 100% Labels)

| Model Architecture | Downstream Test Dice $\uparrow$ | Downstream Test IoU $\uparrow$ | 95th Percentile HD95 $\downarrow$ | Effective Feature Rank $\uparrow$ | Avg Cosine Sim $\downarrow$ | Inference Latency | Training Speed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **UNet Baseline (Supervised)** | 0.8691 | 0.7701 | 5.22 px | N/A (CNN) | N/A | **19.18 ms/slice** | **17.61 s/epoch** |
| **nnU-Net Baseline (Supervised SOTA)** | **0.9073** | **0.8319** | **3.34 px** | N/A (CNN) | N/A | 20.59 ms/slice | 33.65 s/epoch |
| **I-JEPA (Fine-tuned)** | 0.8437 | 0.7332 | 11.06 px | 124.98 | 0.9556 | 18.40 ms/slice | 20.32 s/epoch |
| **SigReg JEPA (Fine-tuned)** | **0.8530** | **0.7464** | **10.75 px** | **366.91** (Max) | **0.0309** | **18.38 ms/slice** | **20.16 s/epoch** |
| **VisReg JEPA (Fine-tuned)** | 0.8358 | 0.7215 | 12.19 px | 16.92 | **0.0132** | 18.21 ms/slice | 20.23 s/epoch |

---

## 3. Key Experimental Regimes

1. **Full-Data Benchmark (`v1_full_data_100pct`)**: 50 epochs SSL pre-training + 30 epochs downstream fine-tuning on 100% labeled slices ($1,134$ training images).
2. **Low-Data Label Efficiency (`v2_low_data_efficiency`)**: Evaluates transferability when clinical ground-truth annotations are extremely scarce ($1\%, 5\%, 10\%, 25\%, 50\%, 100\%$ labels).
3. **Out-of-Distribution (OOD) Scanner Generalization (`v3_ood_generalization`)**: Evaluates feature robustness under simulated scanner hardware shifts (Rician Noise SNR degradation & $B_1$ Field Inhomogeneity coil shifts).

---

## 4. Directory Layout

```text
thesis_2d/
├── pyproject.toml             # Environment dependencies & tool configs (uv, pytest, ruff)
├── uv.lock                    # Dependency lockfile
├── .python-version            # Python 3.12 specification
├── .gitignore                 # Binary/output exclusion rules
├── dir_structure.md           # Project directory specification document
├── agents.md                  # Scientific reasoning & AI agent writing guidelines
├── README.md                  # Project overview and CLI guide
│
├── configs/                   # Experiment & model configurations
│   ├── base.yaml
│   ├── dataset/brats2d.yaml
│   ├── model/                 # unet.yaml, nnunet.yaml, ijepa.yaml, sigreg_jepa.yaml, visreg_jepa.yaml
│   └── experiment/            # Pre-training and fine-tuning configs
│
├── src/                       # Core python library package (`brats_jepa`)
│   └── brats_jepa/
│       ├── config.py          # Centralized path resolution
│       ├── data/              # BraTS2DDataset & JEPAMaskingTransform
│       ├── models/            # BraTS2DUNet, BraTS2DnnUNet, VisionTransformerEncoder2D, JEPASegmentationModel
│       ├── losses/            # CombinedDiceBCELoss, DeepSupervisionLoss, IJEPALoss, SigRegLoss, VisRegLoss
│       ├── metrics/           # compute_segmentation_metrics (Dice, IoU, HD95), compute_effective_rank
│       └── utils/             # Seed, device, logging & metric tracking
│
├── scripts/                   # CLI runners
│   ├── prepare_data.py        # 2D slice extraction & dataset manifest generation
│   ├── train_jepa.py          # Self-supervised JEPA pre-training runner
│   ├── train_unet.py          # Supervised UNet baseline runner
│   ├── train_nnunet.py        # Supervised 2D nnU-Net baseline runner (Deep Supervision)
│   ├── train_downstream.py    # Downstream segmentation fine-tuning runner for pre-trained JEPAs
│   ├── evaluate.py            # Master evaluation benchmark (Dice, IoU, HD95, Latency, Rank)
│   ├── evaluate_low_data.py   # Low-Data Label Efficiency benchmark (1% to 100% labels)
│   ├── evaluate_ood.py        # Out-of-Distribution (OOD) Scanner Domain Generalization runner
│   ├── generate_figures.py    # Publication plot generator
│   └── run_full_pipeline.py   # Master automation pipeline runner with versioning support
│
├── tests/                     # Automated pytest suite (12 tests passing)
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_losses.py
│
├── docs/                      # Technical documentation & reference specifications
│   └── model_architectures.md # Complete model architecture specifications and tensor dimensions
│
├── paper/                     # LaTeX manuscript source (NeurIPS 2026 style)
│   └── latex/
│       ├── main.pdf           # Compiled NeurIPS 2026 PDF Paper
│       ├── main.tex           # LaTeX document source
│       ├── references.bib     # BibTeX references (I-JEPA, SIGReg, VISReg, UNet, nnU-Net, BraTS 2024)
│       └── neurips_2026.sty   # NeurIPS 2026 LaTeX style package
│
├── data/                      # Dataset root
│   └── processed/2d_slices/   # Preprocessed 4-channel .npz slices & metadata.csv
│
└── outputs/                   # Active execution outputs
    ├── checkpoints/           # Trained PyTorch model checkpoints (.pt)
    ├── figures/               # High-resolution benchmark plot PNGs
    ├── metrics/               # Evaluation CSV summary & training JSON logs
    ├── logs/                  # Detailed execution text logs
    │
    └── experiments/           # Versioned Experiment Archive (Preserved Forever)
        ├── v1_full_data_100pct/           <-- Version 1: Full-Data Benchmark (100% Labels)
        ├── v2_low_data_efficiency/        <-- Version 2: Low-Data Label Efficiency (1% to 100%)
        └── v3_ood_generalization/         <-- Version 3: Out-of-Distribution Scanner Shift
```

---

## 5. Quickstart Guide

### Step 1: Environment Setup with `uv`
```bash
uv sync
```

### Step 2: Data Preprocessing
```bash
uv run python scripts/prepare_data.py
```

### Step 3: Run Automated Pipeline
Execute experiment sweeps using versioned modes:
```bash
# Run Low-Data Label Efficiency Benchmark
uv run python scripts/run_full_pipeline.py --mode low_data

# Run OOD Scanner Domain Generalization Benchmark
uv run python scripts/run_full_pipeline.py --mode ood

# Run All Experiments, Generate Plots, and Compile Paper PDF
uv run python scripts/run_full_pipeline.py --mode all
```

---

## 6. Individual Execution Guide

### Pre-training JEPAs (SSL - 50 Epochs)
```bash
uv run python scripts/train_jepa.py --model_type ijepa --epochs 50
uv run python scripts/train_jepa.py --model_type sigreg_jepa --epochs 50
uv run python scripts/train_jepa.py --model_type visreg_jepa --epochs 50
```

### Supervised Baselines (30 Epochs)
```bash
# Standard 2D UNet Baseline
uv run python scripts/train_unet.py --epochs 30

# 2D nnU-Net Baseline (Deep Supervision SOTA)
uv run python scripts/train_nnunet.py --epochs 30
```

### Low-Data & OOD Generalization Benchmarks
```bash
# Low-Data Label Efficiency (1%, 5%, 10%, 25%, 50%, 100% labels)
uv run python scripts/evaluate_low_data.py --epochs 30 --exp_version v2_low_data_efficiency

# OOD Scanner Generalization (Rician Noise & Bias Field Shifts)
uv run python scripts/evaluate_ood.py --exp_version v3_ood_generalization
```

### Plot Generation & LaTeX Compilation
```bash
# Generate publication plots
uv run python scripts/generate_figures.py

# Compile NeurIPS 2026 LaTeX paper
cd paper/latex && pdflatex main.tex && bibtex main && pdflatex main.tex
```

---

## 7. Running Automated Unit Tests
```bash
uv run pytest
```
