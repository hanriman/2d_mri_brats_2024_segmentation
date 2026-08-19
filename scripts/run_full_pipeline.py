"""
Master Automation Pipeline Runner with Versioned Experiments.
Supports v1_full_data_100pct, v2_low_data_efficiency, v3_ood_generalization, and v4_men_rt_ood.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

def run_cmd(cmd: list[str], cwd: Path | None = None):
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=cwd)
    duration = time.perf_counter() - start
    print(f"[COMPLETED in {duration:.2f}s]: {' '.join(cmd)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner")
    parser.add_argument("--mode", type=str, choices=["all", "low_data", "ood", "full_data"], default="all",
                        help="Experiment mode to run")
    return parser.parse_args()

def main():
    args = parse_args()
    start_all = time.perf_counter()
    
    if args.mode in ["all", "full_data"]:
        print("\n" + "="*80)
        print("PHASE 1: FULL-DATA BENCHMARK (100% Labels, 50-Epoch SSL / 30-Epoch Baseline)")
        print("="*80)
        run_cmd(["uv", "run", "python", "scripts/train_jepa.py", "--model_type", "ijepa", "--epochs", "50", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_jepa.py", "--model_type", "sigreg_jepa", "--epochs", "50", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_jepa.py", "--model_type", "visreg_jepa", "--epochs", "50", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_unet.py", "--epochs", "30", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_nnunet.py", "--epochs", "30", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_downstream.py", "--model_type", "ijepa", "--epochs", "30", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_downstream.py", "--model_type", "sigreg_jepa", "--epochs", "30", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/train_downstream.py", "--model_type", "visreg_jepa", "--epochs", "30", "--batch_size", "8"])
        run_cmd(["uv", "run", "python", "scripts/evaluate.py"])

    if args.mode in ["all", "low_data"]:
        print("\n" + "="*80)
        print("PHASE 2: LOW-DATA LABEL EFFICIENCY BENCHMARK (1%, 5%, 10%, 25%, 50%, 100% Labels)")
        print("="*80)
        run_cmd(["uv", "run", "python", "scripts/evaluate_low_data.py", "--epochs", "30", "--exp_version", "v2_low_data_efficiency"])

    if args.mode in ["all", "ood"]:
        print("\n" + "="*80)
        print("PHASE 3: OUT-OF-DISTRIBUTION (OOD) SCANNER & CROSS-PATHOLOGY BENCHMARKS")
        print("="*80)
        run_cmd(["uv", "run", "python", "scripts/evaluate_ood.py", "--exp_version", "v3_ood_generalization"])
        run_cmd(["uv", "run", "python", "scripts/evaluate_men_rt_ood.py", "--max_samples", "5000", "--exp_version", "v4_men_rt_ood"])

    # Generate All Publication Figures & Compile LaTeX Paper
    run_cmd(["uv", "run", "python", "scripts/generate_figures.py"])
    paper_dir = Path("paper/latex").resolve()
    run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir)
    run_cmd(["bibtex", "main"], cwd=paper_dir)
    run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir)
    run_cmd(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=paper_dir)
    
    total = time.perf_counter() - start_all
    print(f"\n" + "="*80)
    print(f"PIPELINE SWEEP COMPLETED SUCCESSFULLY IN {total/60:.2f} MINUTES")
    print(f"="*80 + "\n")

if __name__ == "__main__":
    main()
