#!/usr/bin/env python3
"""
Automated Kaggle Packaging Script for BraTS JEPA 2D Project.
Packages datasets and codebase into clean, upload-ready zip archives for Kaggle.

Usage:
    uv run python scripts/package_for_kaggle.py
    uv run python scripts/package_for_kaggle.py --code_only
    uv run python scripts/package_for_kaggle.py --data_only
    uv run python scripts/package_for_kaggle.py --separate_datasets
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

# Project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_PATTERNS = {
    "__pycache__",
    ".pyc",
    ".pyo",
    ".git",
    ".gitignore",
    ".gitattributes",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "outputs",
    "dist_kaggle",
    ".system_generated",
    ".antigravity",
    "paper",
    "scratch",
}

CODE_DIRS = ["src", "scripts", "configs", "tests"]
CODE_FILES = ["pyproject.toml", "README.md", "LICENSE"]


def should_exclude(rel_path: Path) -> bool:
    """Returns True if any part of the path matches an exclusion pattern."""
    for part in rel_path.parts:
        if part in EXCLUDE_PATTERNS or part.endswith(".egg-info"):
            return True
        if any(part.endswith(pat) for pat in [".pyc", ".pyo", ".swp"]):
            return True
    return False


def format_size(num_bytes: int) -> str:
    """Formats byte count into human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def package_code(output_zip: Path):
    """Packages code, configs, tests, and pyproject.toml into a zip file."""
    print(f"\n📦 Packaging Codebase -> {output_zip} ...")
    t0 = time.perf_counter()
    file_count = 0
    total_uncompressed = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add directories
        for folder_name in CODE_DIRS:
            folder_path = PROJECT_ROOT / folder_name
            if not folder_path.is_dir():
                continue
            for root, dirs, files in os.walk(folder_path):
                # Filter out excluded directories in-place
                dirs[:] = [d for d in dirs if d not in EXCLUDE_PATTERNS]
                for file in sorted(files):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(PROJECT_ROOT)
                    if should_exclude(rel_path):
                        continue
                    zf.write(file_path, arcname=str(rel_path))
                    file_count += 1
                    total_uncompressed += file_path.stat().st_size

        # 2. Add root config/readme files
        for fname in CODE_FILES:
            file_path = PROJECT_ROOT / fname
            if file_path.is_file():
                zf.write(file_path, arcname=fname)
                file_count += 1
                total_uncompressed += file_path.stat().st_size

        # 3. Generate and embed requirements.txt for convenience on Kaggle
        requirements_content = (
            "monai>=1.3.0\n"
            "tqdm>=4.66.0\n"
            "pyyaml>=6.0\n"
            "seaborn>=0.13.0\n"
            "scikit-learn>=1.4.0\n"
            "pandas>=2.0.0\n"
            "matplotlib>=3.8.0\n"
            "scipy>=1.11.0\n"
            "nibabel>=5.0.0\n"
        )
        zf.writestr("requirements.txt", requirements_content)
        file_count += 1

    elapsed = time.perf_counter() - t0
    zip_size = output_zip.stat().st_size
    ratio = (1.0 - (zip_size / total_uncompressed)) * 100 if total_uncompressed > 0 else 0
    print(f"   ✓ Added {file_count} files ({format_size(total_uncompressed)} uncompressed)")
    print(f"   ✓ Code archive size: {format_size(zip_size)} (saved {ratio:.1f}%) in {elapsed:.2f}s")


def package_dataset(
    dataset_name: str,
    dataset_dir: Path,
    output_zip: Path,
    arc_prefix: str = ""
):
    """Packages a single dataset directory into a zip file."""
    if not dataset_dir.exists():
        print(f"⚠️ Dataset directory not found: {dataset_dir}. Skipping.")
        return 0, 0

    print(f"\n📦 Packaging Dataset: {dataset_name} ({dataset_dir}) -> {output_zip} ...")
    t0 = time.perf_counter()
    file_count = 0
    total_uncompressed = 0

    # Collect all files to show accurate progress
    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            p = Path(root) / f
            if not should_exclude(p.relative_to(PROJECT_ROOT)):
                all_files.append(p)

    total_files = len(all_files)
    print(f"   Scanning found {total_files:,} files...")

    # Write to zip
    mode = "a" if output_zip.exists() else "w"
    with zipfile.ZipFile(output_zip, mode, zipfile.ZIP_DEFLATED) as zf:
        for idx, file_path in enumerate(all_files, start=1):
            rel_path = file_path.relative_to(dataset_dir)
            arc_name = f"{arc_prefix}/{rel_path}" if arc_prefix else f"{dataset_name}/{rel_path}"
            zf.write(file_path, arcname=arc_name)
            file_count += 1
            total_uncompressed += file_path.stat().st_size
            if idx % 2000 == 0 or idx == total_files:
                print(f"   Progress: {idx:,} / {total_files:,} files ({idx/total_files*100:.1f}%)")

    elapsed = time.perf_counter() - t0
    zip_size = output_zip.stat().st_size
    print(f"   ✓ Archived {file_count:,} files from {dataset_name}")
    print(f"   ✓ Completed in {elapsed:.1f}s | Uncompressed: {format_size(total_uncompressed)} | Archive: {format_size(zip_size)}")
    return file_count, total_uncompressed


def parse_args():
    parser = argparse.ArgumentParser(description="Package thesis code and datasets for Kaggle.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dist_kaggle",
        help="Target output directory for zip packages (default: dist_kaggle)",
    )
    parser.add_argument("--code_only", action="store_true", help="Only package codebase")
    parser.add_argument("--data_only", action="store_true", help="Only package datasets")
    parser.add_argument(
        "--separate_datasets",
        action="store_true",
        help="Package datasets into separate zips (brats_gli_2d.zip and brats_men_rt_2d.zip) instead of bundled",
    )
    parser.add_argument(
        "--exclude_men_rt",
        action="store_true",
        help="Exclude BraTS-MEN-RT (Meningioma OOD) dataset from packaging",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      KAGGLE EXPORT & PACKAGING PIPELINE")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {out_dir}")

    # 1. Package Codebase
    if not args.data_only:
        code_zip = out_dir / "thesis_2d_code.zip"
        package_code(code_zip)

    # 2. Package Datasets
    if not args.code_only:
        gli_dir = PROJECT_ROOT / "data" / "processed" / "brats_gli_2d"
        men_dir = PROJECT_ROOT / "data" / "processed" / "brats_men_rt_2d"

        if args.separate_datasets:
            if gli_dir.exists():
                gli_zip = out_dir / "brats_gli_2d.zip"
                if gli_zip.exists():
                    gli_zip.unlink()
                package_dataset("brats_gli_2d", gli_dir, gli_zip, arc_prefix="brats_gli_2d")

            if not args.exclude_men_rt and men_dir.exists():
                men_zip = out_dir / "brats_men_rt_2d.zip"
                if men_zip.exists():
                    men_zip.unlink()
                package_dataset("brats_men_rt_2d", men_dir, men_zip, arc_prefix="brats_men_rt_2d")
        else:
            bundle_zip = out_dir / "brats_2d_datasets.zip"
            if bundle_zip.exists():
                bundle_zip.unlink()
            if gli_dir.exists():
                package_dataset("brats_gli_2d", gli_dir, bundle_zip, arc_prefix="brats_gli_2d")
            if not args.exclude_men_rt and men_dir.exists():
                package_dataset("brats_men_rt_2d", men_dir, bundle_zip, arc_prefix="brats_men_rt_2d")

    print("\n" + "=" * 80)
    print("  PACKAGE CREATION COMPLETE!")
    print(f"  Files created in: {out_dir}")
    for item in sorted(out_dir.iterdir()):
        if item.is_file():
            print(f"    - {item.name:30s} ({format_size(item.stat().st_size)})")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
