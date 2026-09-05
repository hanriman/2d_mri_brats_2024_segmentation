"""
Master Automation Pipeline Runner with Custom Version Tagging and Wall-Clock Timestamp Logging.
Supports --mode (all, low_data, ood, full_data) and custom --tag (e.g. --tag v2_post_audit).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None):
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=cwd)
    duration = time.perf_counter() - start
    print(f"[COMPLETED in {duration:.2f}s ({duration/60:.2f} min)]: {' '.join(cmd)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline Runner")
    parser.add_argument("--mode", type=str, choices=["all", "low_data", "ood", "full_data"], default="all",
                        help="Experiment mode to run")
    parser.add_argument("--tag", type=str, default="v2_post_audit",
                        help="Custom experiment version tag prefix (e.g. v2_post_audit)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA AMP (mixed precision)")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to metadata.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output root directory")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Pre-trained checkpoint directory")
    parser.add_argument("--p_drop", type=float, default=0.25, help="Random modality dropout probability during training")
    parser.add_argument("--skip_latex", action="store_true", help="Skip LaTeX compilation")
    return parser.parse_args()

def main():
    args = parse_args()
    start_wall_time = datetime.now()
    start_all = time.perf_counter()
    tag = args.tag
    
    low_data_version = f"{tag}_low_data"
    ood_version = f"{tag}_ood"
    men_rt_version = f"{tag}_men_rt_ood"
    
    print("\n" + "="*85)
    print(f"  MASTER PIPELINE EXECUTION STARTED AT: {start_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  MODE: {args.mode.upper()} | VERSION TAG: {tag}")
    print("="*85 + "\n")
    
    base_flags = []
    if args.amp:
        base_flags.append("--amp")
    if args.metadata_csv:
        base_flags.extend(["--metadata_csv", args.metadata_csv])
    if args.output_dir:
        base_flags.extend(["--output_dir", args.output_dir])
        
    train_flags = list(base_flags) + ["--p_drop", str(args.p_drop)]
    ckpt_flags = []
    if args.checkpoint_dir:
        ckpt_flags.extend(["--checkpoint_dir", args.checkpoint_dir])

    if args.mode in ["all", "full_data"]:
        print("\n" + "="*80)
        print("PHASE 1: FULL-DATA BENCHMARK (100% Labels, 50-Epoch SSL / 30-Epoch Baseline)")
        print("="*80)
        run_cmd([sys.executable, "scripts/train_jepa.py", "--model_type", "ijepa", "--epochs", "50", "--batch_size", "8"] + train_flags)
        run_cmd([sys.executable, "scripts/train_jepa.py", "--model_type", "sigreg_jepa", "--epochs", "50", "--batch_size", "8"] + train_flags)
        run_cmd([sys.executable, "scripts/train_jepa.py", "--model_type", "visreg_jepa", "--epochs", "50", "--batch_size", "8"] + train_flags)
        run_cmd([sys.executable, "scripts/train_unet.py", "--epochs", "30", "--batch_size", "8"] + train_flags)
        run_cmd([sys.executable, "scripts/train_nnunet.py", "--epochs", "30", "--batch_size", "8"] + train_flags)
        run_cmd([sys.executable, "scripts/train_downstream.py", "--model_type", "ijepa", "--epochs", "30", "--batch_size", "8"] + train_flags + ckpt_flags)
        run_cmd([sys.executable, "scripts/train_downstream.py", "--model_type", "sigreg_jepa", "--epochs", "30", "--batch_size", "8"] + train_flags + ckpt_flags)
        run_cmd([sys.executable, "scripts/train_downstream.py", "--model_type", "visreg_jepa", "--epochs", "30", "--batch_size", "8"] + train_flags + ckpt_flags)
        run_cmd([sys.executable, "scripts/evaluate.py"] + ([f for f in base_flags if f != "--amp"]) + ckpt_flags)

    if args.mode in ["all", "low_data"]:
        print("\n" + "="*80)
        print(f"PHASE 2: LOW-DATA LABEL EFFICIENCY BENCHMARK ({low_data_version})")
        print("="*80)
        run_cmd([sys.executable, "scripts/evaluate_low_data.py", "--epochs", "30", "--exp_version", low_data_version] + base_flags + ckpt_flags)

    if args.mode in ["all", "ood"]:
        print("\n" + "="*80)
        print(f"PHASE 3: OUT-OF-DISTRIBUTION (OOD) SCANNER & CROSS-PATHOLOGY BENCHMARKS ({ood_version})")
        print("="*80)
        run_cmd([sys.executable, "scripts/evaluate_ood.py", "--exp_version", ood_version] + ([f for f in base_flags if f != "--amp"]) + ckpt_flags)
        run_cmd([sys.executable, "scripts/evaluate_men_rt_ood.py", "--max_samples", "5000", "--exp_version", men_rt_version] + ([f for f in base_flags if f != "--amp"]) + ckpt_flags)

    # Generate All Publication Figures & Compile LaTeX Paper
    run_cmd([sys.executable, "scripts/generate_figures.py"])
    if not args.skip_latex:
        paper_dir = Path("paper/latex").resolve()
        tex_target = "extended_main" if (paper_dir / "extended_main.tex").exists() else "main"
        try:
            run_cmd(["pdflatex", "-interaction=nonstopmode", f"{tex_target}.tex"], cwd=paper_dir)
            run_cmd(["bibtex", tex_target], cwd=paper_dir)
            run_cmd(["pdflatex", "-interaction=nonstopmode", f"{tex_target}.tex"], cwd=paper_dir)
            run_cmd(["pdflatex", "-interaction=nonstopmode", f"{tex_target}.tex"], cwd=paper_dir)
        except Exception as e:
            print(f"LaTeX compilation skipped or failed: {e}")
    else:
        print("Skipping LaTeX compilation (--skip_latex requested).")
    
    end_wall_time = datetime.now()
    total_sec = time.perf_counter() - start_all
    
    print("\n" + "="*85)
    print(f"  MASTER PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  START TIME: {start_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  END TIME:   {end_wall_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  TOTAL ELAPSED TIME: {total_sec/60:.2f} MINUTES ({total_sec:.2f} SECONDS)")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
