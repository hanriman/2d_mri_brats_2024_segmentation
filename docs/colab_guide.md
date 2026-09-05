# 🚀 Complete Guide: Training & Benchmarking on Google Colab

This guide explains how to run the entire **BraTS 2D JEPA Representation Learning and Segmentation Benchmark** on **Google Colab** (Free or Pro tier) with GPU acceleration (Tesla T4, A100, or V100).

---

## 1. Google Colab vs. Kaggle: Key Comparisons

| Factor | Kaggle | Google Colab |
| :--- | :--- | :--- |
| **GPU Accelerator** | NVIDIA Tesla T4 (16 GB) / 2x T4 | Free: NVIDIA T4 (15 GB)<br>Pro: A100 (40GB) / L4 (24GB) / V100 |
| **Persistence** | Session ephemeral; output zipped manually | **Google Drive Mount**: permanent checkpoint & metric storage |
| **Idle Timeout** | 40 minutes interactive (unlimited in background Commit) | ~15–30 minutes idle disconnect on free tier |
| **Max Session Duration** | 12 hours max | Free: 4–12 hours before preemption<br>Pro: up to 24 hours |
| **Dataset Storage** | Instant Kaggle Input mounts (`/kaggle/input`) | Upload `brats_2d_datasets.zip` to Google Drive or `/content/` |

---

## 2. High-Speed Architecture on Google Colab

```text
GitHub (https://github.com/hanriman/2d_mri_brats_2024_segmentation.git)
  │
  ▼ (!git clone directly into Colab)
Colab High-Speed Local Disk (/content/)
  ├── thesis_2d/ (cloned codebase, pip install -e .)
  └── brats_gli_2d/ (unzipped 9,046 slice .npy files)
        │
        ▼ (Model Training & Evaluation with CUDA AMP)
Google Drive (/content/drive/MyDrive/)
  ├── brats_2d_datasets.zip (1.6 GB upload for data)
  └── thesis_2d_outputs/  ◄── (Permanent Checkpoints & Results)
            ├── checkpoints/
            ├── metrics/
            └── figures/
```

> [!IMPORTANT]
> **Why unzip data to `/content/` instead of reading directly from Drive?**
> Google Drive uses a FUSE network filesystem. Reading thousands of small `.npy` files across network FUSE causes high latency. Unzipping to `/content/` gives **~10x faster** DataLoader batching!
> However, **all outputs (checkpoints, CSV summaries, figures) should be saved to `/content/drive/MyDrive/thesis_2d_outputs`** so you never lose results if Colab disconnects.

---

## 3. Step-by-Step Walkthrough

### Step 1: Push Latest Changes & Prepare Dataset

1. **Push your code to GitHub:**
   Make sure your local changes are committed and pushed to your repo:
   ```bash
   git add .
   git commit -m "feat: Kaggle & Colab GPU support with AMP, dynamic paths, and smoke tests"
   git push origin main
   ```
2. **Package & Upload Dataset:**
   Package the datasets:
   ```bash
   uv run python scripts/package_for_kaggle.py --data_only
   ```
   Upload `dist_kaggle/brats_2d_datasets.zip` (~1.6 GB) to your **Google Drive** (`MyDrive/`).
   *(You do NOT need to upload code to Google Drive, since Colab will git clone directly from GitHub!)*

---

### Step 2: Open the Notebook in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** -> Select [`notebooks/colab_runner.ipynb`](file:///Users/hanriman/Documents/master/thesis_2d/notebooks/colab_runner.ipynb).
3. Set the hardware accelerator:
   - Go to **Runtime** -> **Change runtime type**.
   - Select **T4 GPU** (or **A100 / L4** if you have Colab Pro).
   - Click **Save**.

---

### Step 3: Run the Setup Cells

1. **Mount Google Drive & Check GPU**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !nvidia-smi
   ```
2. **Install Dependencies**:
   ```python
   !pip install -q monai
   ```
3. **Clone Code & Install**:
   The notebook automatically clones your repository from GitHub directly into `/content/thesis_2d` and installs it in editable mode (`pip install -e .`).
4. **Fast Local Dataset Extraction**:
   Unzips `brats_2d_datasets.zip` onto `/content/` in ~15 seconds.

---

### Step 4: Run Training with Mixed Precision (`--amp`)

All training commands in the notebook use `--amp` and specify `--output_dir /content/drive/MyDrive/thesis_2d_outputs`:

- **Phase 1: Self-Supervised JEPA Pre-training** (I-JEPA, SigReg, VisReg)
- **Phase 2: Supervised Baselines** (ResUNet, nnU-Net)
- **Phase 3: Downstream Segmentation Fine-Tuning**
- **Phase 4: Probing & Evaluation**
- **Phase 5: Low-Data Efficiency Benchmark** (1% to 100% annotations)
- **Phase 6: OOD Benchmarks** (Scanner shifts & Meningioma cross-pathology)
- **Phase 7: Publication Figure Generation & Inline Display**

Because `--output_dir` points to your Google Drive, every checkpoint (`.pt`), metric table (`.csv`), and plot (`.png`) is instantly preserved in your Drive as it finishes.

---

## 4. Tips to Prevent Colab Disconnects

1. **Keep Browser Tab Active:** The free tier disconnects if the tab is left in the background for a prolonged period.
2. **Background Execution (Colab Pro):** If using Colab Pro, you can close the browser and let it run in the background.
3. **Resuming Training:** Since checkpoints are stored in Google Drive, if a disconnect occurs mid-run, you can simply restart the runtime, re-mount Drive, and resume downstream fine-tuning without losing pre-trained weights.
