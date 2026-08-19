# AI Research Directory Template & Best Practices Guide

A production-grade, reproducible directory structure designed for AI, Machine Learning, and Computational Biology research projects with **Automatic Experiment Versioning**.

---

## 1. Project Directory Layout

```text
ai-research-project/
├── README.md                  # Project overview, key findings, setup instructions, & CLI quickstart
├── pyproject.toml             # Environment dependencies, build system, & tool configs (ruff, pytest)
├── uv.lock / environment.yml  # Exact pinned dependency lockfile
├── .python-version            # Python version specification (e.g., 3.12)
├── .gitignore                 # Exclude heavy binaries, checkpoints, local caches, and secret keys
├── .pre-commit-config.yaml    # Pre-commit hooks for formatting, linting, and secret detection
├── dir_structure.md           # This directory structure specification document
│
├── configs/                   # Experiment & model configurations (YAML / Hydra / Pydantic)
│   ├── base.yaml              # Default hyperparameter defaults
│   ├── model/                 # Model architecture configs (unet.yaml, nnunet.yaml, ijepa.yaml, etc.)
│   ├── dataset/               # Dataset & preprocessing configs
│   └── experiment/            # Named experiment specs (e.g., baseline.yaml, ablation_loss.yaml)
│
├── src/                       # Core python library package (reusable modules)
│   └── my_project/
│       ├── __init__.py
│       ├── config.py          # Centralized path & env variable resolution
│       ├── models/            # Model architectures & neural net blocks
│       ├── data/              # Data loaders, dataset classes, & preprocessing pipelines
│       ├── losses/            # Custom loss functions & regularizers (SIGReg, VISReg, Dice+BCE, DeepSup)
│       ├── metrics/           # Custom evaluation probes & metrics (Dice, IoU, HD95, Effective Rank)
│       └── utils/             # Logging, seeding, device handlers, & metric tracking
│
├── scripts/                   # Executable CLI runners & benchmark entry points
│   ├── prepare_data.py        # 2D slice extraction & dataset manifest generation
│   ├── train_jepa.py          # Self-supervised JEPA pre-training runner
│   ├── train_unet.py          # Supervised UNet baseline runner
│   ├── train_nnunet.py        # Supervised 2D nnU-Net baseline runner (Deep Supervision)
│   ├── train_downstream.py    # Downstream segmentation fine-tuning runner for pre-trained JEPAs
│   ├── evaluate.py            # Master evaluation benchmark (Dice, IoU, HD95, Latency, Rank)
│   ├── evaluate_low_data.py   # Low-Data Label Efficiency benchmark (1% to 100% labels)
│   ├── evaluate_ood.py        # Out-of-Distribution (OOD) Scanner Domain Generalization runner
│   ├── generate_figures.py    # Generates publication-ready figures & plots
│   └── run_full_pipeline.py   # Master automation pipeline runner with versioning support
│
├── notebooks/                 # Exploratory data analysis & interactive visualizations
│   ├── 01_eda.ipynb
│   └── 02_latent_analysis.ipynb
│
├── tests/                     # Automated test suite (pytest)
│   ├── conftest.py            # Shared fixtures (dummy tensors, mini datasets)
│   ├── test_models.py         # Shape & gradient sanity checks
│   ├── test_losses.py         # Loss invariance & boundary tests
│   └── test_data.py           # Data loading & transformation tests
│
├── docs/                      # Research documentation & design specifications
│   ├── model_architectures.md # Detailed technical specs & tensor dimensions for all models
│   ├── research_proposal.md   # Research question, hypotheses, & execution roadmap
│   ├── data_dictionary.md     # Feature definitions & data lineage details
│   └── setup_guide.md         # Environment setup & compute hardware instructions
│
├── paper/                     # Manuscript source code & submission artifacts
│   ├── latex/                 # Tracked LaTeX source files (.tex, .bib, style files)
│   │   ├── main.tex           # NeurIPS 2026 Camera-Ready LaTeX source
│   │   ├── references.bib     # BibTeX citations
│   │   └── neurips_2026.sty   # NeurIPS 2026 LaTeX style package
│   └── submission_plan.md    # Target journals/conferences & review checklists
│
├── data/                      # Local data storage (GIT IGNORED)
│   ├── raw/                   # Immutable raw datasets (read-only)
│   ├── interim/               # Intermediate transformed states
│   └── processed/             # Ready-to-train tensors or processed feature files
│
└── outputs/                   # Generated artifacts & versioned run logs (GIT IGNORED)
    ├── checkpoints/           # Latest active model weight checkpoints (.pt)
    ├── figures/               # Latest active output plots (.pdf, .png, .svg)
    ├── metrics/               # Latest active evaluation JSONs, CSV summaries
    ├── logs/                  # Console output logs & error tracebacks
    │
    └── experiments/           # Versioned Experiment Archive (PRESERVED FOREVER)
        ├── v1_full_data_100pct/           <-- Version 1: Full-Data Benchmark (100% Labels)
        │   ├── checkpoints/               (best_ijepa.pt, best_sigreg_jepa.pt, best_nnunet.pt)
        │   ├── metrics/                   (evaluation_benchmark_summary.csv, *_metrics.json)
        │   ├── figures/                   (segmentation_performance_benchmark.png, etc.)
        │   └── main_v1_paper.pdf          (Compiled NeurIPS Paper PDF for Version 1)
        │
        ├── v2_low_data_efficiency/        <-- Version 2: Low-Data Label Efficiency (1% to 100%)
        │   ├── metrics/                   (low_data_benchmark_summary.csv)
        │   └── figures/                   (low_data_label_efficiency.png)
        │
        └── v3_ood_generalization/         <-- Version 3: Out-of-Distribution Scanner Shift
            ├── metrics/                   (ood_benchmark_summary.csv)
            └── figures/                   (ood_domain_generalization.png)
```

---

## 2. Key Architectural Principles

### A. Strict Decoupling of Library vs. Execution
- **`src/` is a Python Library**: Holds purely functional, tested, reusable code. It should **never** execute training loops or hardcode file paths upon import.
- **`scripts/` are Entry Points**: CLI entry points that import from `src/`, parse arguments (e.g. via `argparse`), load configs, and run workflows.

### B. Immutable Data Rule
- **`data/raw/` is Read-Only**: Original raw datasets (e.g. `.nii.gz`, `.csv`, `.npz`) must never be modified in-place by any script.
- All transformations write new files to `data/interim/` or `data/processed/`.

### C. Standardized Output & Automatic Experiment Versioning
- **Never Overwrite Previous Completed Runs**: When launching a new experiment regime (e.g., Low-Data efficiency or OOD Scanner shift), previous completed runs must be archived into `outputs/experiments/v{version}_{description}/`.
- **Version Isolation**: Each version folder maintains its own isolated `checkpoints/`, `metrics/`, `figures/`, `logs/`, and compiled PDF paper so results remain 100% reproducible and comparable across experiment iterations.

### D. Versioning LaTeX Source Code
- Include LaTeX source files under `paper/latex/` inside git tracking.
- Exclude compiled LaTeX intermediate build artifacts (`*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`, `*.toc`) in `.gitignore`.

---

## 3. Recommended `.gitignore` Snippet for AI Research

```gitignore
# Environment & Dependencies
.venv/
env/
venv/
*.egg-info/
dist/
build/

# Local Data & Outputs (Heavy Binaries)
data/
outputs/
runs/
wandb/
.hydra/

# Python & Jupyter Caches
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/

# Operating System & IDE
.DS_Store
Thumbs.db
.vscode/
.idea/

# LaTeX Build Artifacts (Keep .tex, .bib, .cls, .pdf figures tracked)
paper/latex/*.aux
paper/latex/*.bbl
paper/latex/*.blg
paper/latex/*.fdb_latexmk
paper/latex/*.fls
paper/latex/*.log
paper/latex/*.out
paper/latex/*.synctex.gz
paper/latex/*.toc
```

---

## 4. Modern Tooling Recommendations

| Category | Tool | Purpose |
| :--- | :--- | :--- |
| **Package Manager** | `uv` | Ultra-fast dependency resolution and lockfile management |
| **Config Management** | `YAML / argparse` | Hierarchical configuration composition with CLI overrides |
| **Linting & Formatting**| `ruff` | Fast, unified linter and code formatter |
| **Testing** | `pytest` | Unit testing for neural net shapes, loss bounds, and data pipelines |
| **Experiment Tracking** | `MetricTracker` / `JSON` / `WandB` | Dynamic logging of metrics, hyperparameters, and artifacts |
| **Hardware Acceleration**| `PyTorch` | Portable acceleration across CUDA, MPS (Apple Silicon), and CPU |

---

## 5. Quickstart: Using This Template

```bash
# 1. Initialize repository structure
mkdir -p configs src/brats_jepa scripts notebooks tests docs paper/latex data/{raw,processed} outputs/{checkpoints,figures,metrics,experiments}

# 2. Initialize uv Python environment
uv sync

# 3. Run master automation pipeline with versioning
uv run python scripts/run_full_pipeline.py --mode all
```
