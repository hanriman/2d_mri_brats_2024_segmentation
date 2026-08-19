# Setup & Execution Guide: BraTS 2D JEPA Research Repository

## 1. Prerequisites & Environment Setup

This project uses `uv` for dependency management.

```bash
# 1. Verify uv installation
uv --version

# 2. Sync dependencies into virtualenv (.venv)
uv sync
```

---

## 2. Running Data Preparation

```bash
# Verify or extract 2D axial slices into data/processed/2d_slices/
uv run python scripts/prepare_data.py
```

---

## 3. Running Pre-training Sweeps

### A. Baseline I-JEPA Pre-training
```bash
uv run python scripts/train_jepa.py --model_type ijepa --epochs 50 --batch_size 8
```

### B. SigReg JEPA Pre-training
```bash
uv run python scripts/train_jepa.py --model_type sigreg_jepa --epochs 50 --batch_size 8
```

### C. VisReg JEPA Pre-training
```bash
uv run python scripts/train_jepa.py --model_type visreg_jepa --epochs 50 --batch_size 8
```

---

## 4. Running Supervised Baseline & Evaluation

```bash
# 1. Train Supervised 2D ResUNet Baseline
uv run python scripts/train_unet.py --epochs 30 --batch_size 8

# 2. Run Comprehensive Downstream & Representation Benchmark
uv run python scripts/evaluate.py

# 3. Generate Publication Figures
uv run python scripts/generate_figures.py
```

---

## 5. Running Automated Unit Tests

```bash
uv run pytest
```
