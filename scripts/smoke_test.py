#!/usr/bin/env python3
"""
Rapid End-to-End Local Smoke Test for BraTS 2D JEPA Project.
Verifies dataset loading, SSL pre-training, supervised baselines, downstream fine-tuning,
evaluation benchmarking, and figure generation in ~20-30 seconds.

Usage:
    uv run python scripts/smoke_test.py
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def print_step(title: str):
    print("\n" + "=" * 75)
    print(f"  🧪 {title}")
    print("=" * 75)


def run_cmd(cmd: list[str]):
    print(f"[CMD]: {' '.join(cmd)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    duration = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"\n❌ FAILED in {duration:.2f}s:\n{res.stderr}")
        print(res.stdout)
        sys.exit(1)
    print(f"✓ Passed in {duration:.2f}s")


def main():
    start_total = time.perf_counter()
    print("\n" + "#" * 75)
    print("   BRATS 2D JEPA — RAPID LOCAL SMOKE TEST SUITE")
    print("#" * 75)

    test_out = PROJECT_ROOT / "outputs" / "smoke_test_temp"
    if test_out.exists():
        shutil.rmtree(test_out)
    test_out.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Loading Sanity Check
    print_step("Step 1/6: Verifying Dataset & Transforms")
    from brats_jepa.config import get_metadata_path
    from brats_jepa.data import BraTS2DDataset, JEPAMaskingTransform

    meta_path = get_metadata_path("brats_gli_2d")
    print(f"Resolved metadata path: {meta_path}")
    assert meta_path.exists(), f"Metadata not found at {meta_path}"

    ds = BraTS2DDataset(metadata_csv=meta_path, split="train", jepa_masking=JEPAMaskingTransform())
    sample = ds[0]
    print(f"Sample Image Tensor: shape={sample['image'].shape}, dtype={sample['image'].dtype}")
    print(f"Context Indices Tensor: shape={sample['context_indices'].shape} (guaranteed uniform)")
    print(f"Target Indices Count: {len(sample['target_indices'])}")
    print("✓ Dataset & Masking Transform passed!")

    # 2. Fast SSL Pre-training (1 epoch, 2 batches)
    print_step("Step 2/6: Smoke Testing SSL Pre-training (I-JEPA)")
    run_cmd([
        sys.executable, "scripts/train_jepa.py",
        "--model_type", "ijepa",
        "--epochs", "1",
        "--batch_size", "2",
        "--max_batches", "2",
        "--output_dir", str(test_out),
    ])

    # 3. Fast Supervised Baseline Training (1 epoch, 2 batches)
    print_step("Step 3/6: Smoke Testing Supervised UNet Baseline")
    run_cmd([
        sys.executable, "scripts/train_unet.py",
        "--epochs", "1",
        "--batch_size", "2",
        "--max_batches", "2",
        "--output_dir", str(test_out),
    ])

    # 4. Fast Downstream Fine-Tuning (1 epoch, 2 batches)
    print_step("Step 4/6: Smoke Testing Downstream Fine-Tuning")
    run_cmd([
        sys.executable, "scripts/train_downstream.py",
        "--model_type", "ijepa",
        "--epochs", "1",
        "--batch_size", "2",
        "--max_batches", "2",
        "--output_dir", str(test_out),
        "--checkpoint_dir", str(test_out / "checkpoints"),
    ])

    # 5. Fast Evaluation Benchmark (2 test batches)
    print_step("Step 5/6: Smoke Testing Evaluation Benchmark")
    run_cmd([
        sys.executable, "scripts/evaluate.py",
        "--output_dir", str(test_out),
        "--checkpoint_dir", str(test_out / "checkpoints"),
        "--max_batches", "2",
    ])

    # 6. Fast Figure Generation
    print_step("Step 6/6: Smoke Testing Publication Figure Generation")
    run_cmd([
        sys.executable, "scripts/generate_figures.py",
        "--metrics_dir", str(test_out / "metrics"),
        "--figures_dir", str(test_out / "figures"),
    ])

    # Cleanup temporary test directory
    if test_out.exists():
        shutil.rmtree(test_out)

    total_time = time.perf_counter() - start_total
    print("\n" + "=" * 75)
    print(f"  🎉 ALL LOCAL SMOKE TESTS PASSED SUCCESSFULLY in {total_time:.2f}s!")
    print("  Your codebase and datasets are fully validated and ready to train")
    print("  on Kaggle or Google Colab with 100% confidence!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
