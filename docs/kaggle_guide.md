# 🚀 Complete Guide: Training & Benchmarking on Kaggle GPU

This guide provides an end-to-end walkthrough for running the entire **BraTS 2D JEPA Representation Learning and Segmentation Benchmark** on Kaggle's free GPU resources (NVIDIA Tesla T4).

---

## 1. Why Kaggle? Hardware & Performance Benefits

| Feature | Kaggle Free Tier | Impact on This Project |
| :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA Tesla T4 (16 GB VRAM) or 2x T4 | Ample memory for ViT-B/16 and 4-channel MRI tensors |
| **Mixed Precision (AMP)** | Tensor Cores supported via FP16 | **~3x training speedup** with `--amp` |
| **Runtime Limits** | Up to 12 hours per session / 30 hours per week | Sufficient to run full SSL pre-training + downstream tasks |
| **Pre-installed Stack** | PyTorch, CUDA, TorchVision, Pandas, SciPy | Zero environment friction; only `monai` needed |

---

## 2. Architecture of the Kaggle Pipeline

```text
[Local Machine]
  scripts/package_for_kaggle.py --data_only
      │
      └── dist_kaggle/brats_2d_datasets.zip (1.6 GB)
              │
              ▼  (Upload to Kaggle Datasets)
[Kaggle Cloud Environment]
  GitHub (https://github.com/hanriman/2d_mri_brats_2024_segmentation.git)
      │
      ▼  (!git clone directly into notebook)
  /kaggle/working/
      ├── thesis_2d/ (editable codebase)
      └── outputs/
            ├── checkpoints/ (*.pt models)
            ├── metrics/     (*.csv benchmark tables)
            ├── figures/     (*.png publication plots)
            └── logs/        (*.log training logs)
              │
              ▼  (1-Click Download)
  /kaggle/working/outputs.zip

  /kaggle/input/
      └── brats-2d-datasets/brats_gli_2d/
                            brats_men_rt_2d/
```

---

## 3. Step-by-Step Instructions

### Step 1: Package Local Dataset

Run the automated packaging script for data:

```bash
uv run python scripts/package_for_kaggle.py --data_only
```

This creates the upload-ready dataset archive:
- **`dist_kaggle/brats_2d_datasets.zip`** (~1.6 GB): Slices and metadata for `brats_gli_2d` (9,046 slices) and `brats_men_rt_2d` (2,856 slices).

*(Note: You do NOT need to upload code, because the Kaggle runner clones directly from your GitHub repository!)*

---

### Step 2: Upload Dataset to Kaggle

1. Go to [kaggle.com](https://www.kaggle.com/) and log in.
2. Click **Create** (left sidebar) -> **New Dataset**.
3. In the upload popup:
   - Enter Title: `brats-2d-datasets`
   - Drag and drop `dist_kaggle/brats_2d_datasets.zip` (1.6 GB).
   - Click **Create** and wait ~1–2 minutes for Kaggle to finish processing.

Once finished, your dataset is live at `/kaggle/input/brats-2d-datasets/`!

---

### Step 3: Create and Configure the Kaggle Notebook

1. In Kaggle, click **Create** -> **New Notebook**.
2. Go to **File** -> **Import Notebook** -> Select [`notebooks/kaggle_runner.ipynb`](file:///Users/hanriman/Documents/master/thesis_2d/notebooks/kaggle_runner.ipynb).
3. In the right-hand **Notebook Settings** panel:
   - **Accelerator**: Choose **GPU T4 x1** (or **GPU T4 x2**).
   - **Internet**: Toggle to **On** (required for `pip install monai`).
   - **Persistence**: "Variables and files" (optional).
4. In the top-right corner of the notebook, click **+ Add Input**:
   - Search for `brats-2d-datasets` and click **Add**.

---

### Step 4: Run the Notebook Phases

The notebook is divided into clear, modular sections:

#### Section 1 to 4: Environment, Dependencies & Setup
- **Cell 1**: `!nvidia-smi` and checks CUDA availability.
- **Cell 2**: `!pip install -q monai` (installs in ~10 seconds).
- **Cell 3**: Clones the GitHub repository directly to `/kaggle/working/thesis_2d` and runs `pip install -e .`.
- **Cell 4**: Verifies dataset paths and runs a sanity tensor check on `BraTS2DDataset`.

#### Section 5: Self-Supervised JEPA Pre-training (50 Epochs, AMP)
```bash
# Full Pre-training (50 Epochs, ~48 min on T4 GPU)
!python scripts/train_jepa.py --model_type ijepa --epochs 50 --batch_size 32 --num_workers 2 --amp
!python scripts/train_jepa.py --model_type sigreg_jepa --epochs 50 --batch_size 32 --num_workers 2 --amp
!python scripts/train_jepa.py --model_type visreg_jepa --epochs 50 --batch_size 32 --num_workers 2 --amp

# Tip: For a fast pipeline dry-run (smoke test), use --epochs 5 (~4.5 min per model):
# !python scripts/train_jepa.py --model_type ijepa --epochs 5 --batch_size 32 --num_workers 2 --amp
```

#### Section 6: Supervised Baselines (30 Epochs, AMP)
```bash
!python scripts/train_unet.py --epochs 30 --batch_size 32 --num_workers 2 --amp
!python scripts/train_nnunet.py --epochs 30 --batch_size 32 --num_workers 2 --amp
```

#### Section 7: Downstream Segmentation Fine-Tuning (30 Epochs, AMP)
```bash
!python scripts/train_downstream.py --model_type ijepa --epochs 30 --batch_size 32 --num_workers 2 --amp
!python scripts/train_downstream.py --model_type sigreg_jepa --epochs 30 --batch_size 32 --num_workers 2 --amp
!python scripts/train_downstream.py --model_type visreg_jepa --epochs 30 --batch_size 32 --num_workers 2 --amp
```

#### Section 8: Test Evaluation & Probing
```bash
!python scripts/evaluate.py --batch_size 32 --num_workers 2
```
Computes Test Dice, IoU, HD95, Linear Probing Accuracy, Effective Rank, and Cosine Similarity across all models.

#### Section 9: Low-Data Efficiency Benchmark (1% - 100% Labels)
```bash
!python scripts/evaluate_low_data.py --epochs 30 --batch_size 32 --num_workers 2 --amp --exp_version kaggle_low_data
# Tip: For dry-run smoke tests, use --epochs 2 to finish within 5 minutes.
```

#### Section 10: Out-of-Distribution (OOD) Benchmarks
```bash
!python scripts/evaluate_ood.py --exp_version kaggle_ood
!python scripts/evaluate_men_rt_ood.py --max_samples 5000 --exp_version kaggle_men_rt_ood
```

#### Section 11 & 12: Figure Generation & 1-Click Download
- Automatically generates all comparison bar plots and label efficiency curves.
- Displays figures inline in the notebook.
- Archives `/kaggle/working/outputs` into `/kaggle/working/outputs.zip`.

---

### Step 5: On-the-Fly Code Updates (No Kernel Restart Required)

Since every execution cell executes `!python scripts/...` via a subprocess, updates to code files take effect immediately. You never need to restart the Kaggle kernel or reinstall dependencies when pulling repository changes:

```python
# Pull latest updates from GitHub in 2 seconds
!git -C /kaggle/working/thesis_2d pull origin main
```

---

### Step 5: Download & Synchronize Results Locally

1. In the right-hand panel under **Output** (`/kaggle/working`), locate `outputs.zip`.
2. Click the three dots `...` next to `outputs.zip` and select **Download**.
3. Move `outputs.zip` to your project root on your local computer and extract it:
   ```bash
   unzip -o outputs.zip -d .
   ```
4. Re-compile the LaTeX manuscript with the new empirical results:
   ```bash
   cd paper/latex
   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
   ```

---

## 4. Background Execution ("Commit & Run")

If you want the entire benchmark to run unattended without keeping your browser window open:

1. Click the blue **Save Version** button (top right of Kaggle interface).
2. Version Type: **Save & Run All (Commit)**.
3. Click **Save**.
4. Kaggle will spin up a dedicated headless VM, execute all cells sequentially, produce the outputs, and shut down automatically when finished.
5. You can view the logs or download the final output files anytime from the notebook's **Versions** tab.

---

## 5. Troubleshooting & Tips

- **CUDA Out of Memory (OOM):** If using batch size 16 exceeds memory during deep supervision with nnU-Net, reduce batch size to 8:
  ```bash
  --batch_size 8
  ```
- **Session Disconnects:** Interactive sessions disconnect after 40 minutes of browser inactivity. For long multi-hour runs, always use **Save Version -> Save & Run All (Commit)**.
- **Quota Management:** Check your remaining weekly GPU quota under your Kaggle profile -> Account Settings (30 hours/week, reset weekly).
