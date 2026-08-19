import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from brats_jepa.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 5 2D Axial Slices per 3D BraTS GLI volume")
    parser.add_argument("--data_dir", type=str, default=str(RAW_DATA_DIR / "BraTS-GLI"),
                        help="Input raw data directory containing GLI patient folders")
    parser.add_argument("--output_dir", type=str, default=str(PROCESSED_DATA_DIR),
                        help="Output directory to save 2D slices")
    parser.add_argument("--slices_per_patient", type=int, default=5,
                        help="Number of 2D axial slices to sample per 3D volume")
    parser.add_argument("--min_brain_pixels", type=int, default=1000,
                        help="Minimum non-zero brain pixels per slice")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of patient scans to preprocess (for debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splitting reproducibility")
    return parser.parse_args()


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize volume intensities using Z-score calculated across non-zero brain region."""
    mask = volume > 0
    if not np.any(mask):
        return volume.astype(np.float32)
    mean = volume[mask].mean()
    std = volume[mask].std()
    normalized = (volume - mean) / std if std > 0 else volume - mean
    normalized[~mask] = 0.0
    return normalized.astype(np.float32)


def select_patient_slices(candidates: list, slices_per_patient: int = 5) -> list:
    """Selects up to `slices_per_patient` slices per volume:
    - Up to 3 top slices with largest tumor pixel count
    - 2 context non-tumor slices (lower & upper brain elevation)
    """
    if len(candidates) <= slices_per_patient:
        return sorted(candidates, key=lambda c: c["z"])

    tumor_cands = sorted([c for c in candidates if c["tumor_pixel_count"] > 0],
                         key=lambda c: c["tumor_pixel_count"], reverse=True)
    non_tumor_cands = sorted([c for c in candidates if c["tumor_pixel_count"] == 0],
                             key=lambda c: c["z"])

    target_tumor_count = min(3, len(tumor_cands))
    selected = tumor_cands[:target_tumor_count]
    selected_zs = {c["z"] for c in selected}

    remaining_pool = [c for c in candidates if c["z"] not in selected_zs]
    remaining_pool = sorted(remaining_pool, key=lambda c: c["z"])

    needed = slices_per_patient - len(selected)

    if len(non_tumor_cands) >= needed:
        pool_to_sample = non_tumor_cands
    else:
        pool_to_sample = remaining_pool

    if needed > 0 and len(pool_to_sample) > 0:
        idxs = np.linspace(0, len(pool_to_sample) - 1, num=needed, dtype=int)
        for idx in idxs:
            selected.append(pool_to_sample[idx])

    if len(selected) < slices_per_patient and len(remaining_pool) > 0:
        for c in remaining_pool:
            if c["z"] not in {s["z"] for s in selected}:
                selected.append(c)
                if len(selected) == slices_per_patient:
                    break

    return sorted(selected, key=lambda c: c["z"])


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print(f"Reading raw BraTS-GLI data from: {data_dir}")
    print(f"Saving processed 2D slices to: {output_dir}")
    print(f"Sampling target: {args.slices_per_patient} slices per 3D volume")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t1n_files = sorted(list(data_dir.rglob("*-t1n.nii.gz")))
    if args.limit:
        t1n_files = t1n_files[:args.limit]
        print(f"Limiting preprocessing to first {args.limit} cases.")

    if len(t1n_files) == 0:
        print(f"Error: No raw NIfTI files (*-t1n.nii.gz) found in {data_dir}.")
        return

    print(f"Found {len(t1n_files)} patient volumes. Analyzing tumor volumes and sampling slices...")
    patient_records = []
    
    for t1n_path in tqdm(t1n_files, desc="Analyzing BraTS-GLI volumes"):
        p_dir = t1n_path.parent
        patient_id = p_dir.name
        
        seg_paths = list(p_dir.glob("*-seg.nii.gz"))
        t1c_paths = list(p_dir.glob("*-t1c.nii.gz"))
        t2w_paths = list(p_dir.glob("*-t2w.nii.gz"))
        t2f_paths = list(p_dir.glob("*-t2f.nii.gz"))
        
        if not (t1c_paths and t2w_paths and t2f_paths):
            continue
            
        try:
            t1n_3d = nib.load(str(t1n_path)).get_fdata().astype(np.float32)
            t1c_3d = nib.load(str(t1c_paths[0])).get_fdata().astype(np.float32)
            t2w_3d = nib.load(str(t2w_paths[0])).get_fdata().astype(np.float32)
            t2f_3d = nib.load(str(t2f_paths[0])).get_fdata().astype(np.float32)
            
            if seg_paths:
                seg_3d = nib.load(str(seg_paths[0])).get_fdata().astype(np.float32)
            else:
                seg_3d = np.zeros_like(t1n_3d)
        except Exception as e:
            print(f"Error loading {patient_id}: {e}")
            continue

        tumor_volume = float(np.sum(seg_3d > 0))
        t1n_norm = zscore_normalize(t1n_3d)
        t1c_norm = zscore_normalize(t1c_3d)
        t2w_norm = zscore_normalize(t2w_3d)
        t2f_norm = zscore_normalize(t2f_3d)

        num_slices = t1n_3d.shape[2]
        patient_candidates = []

        for z in range(num_slices):
            slice_t1n = t1n_norm[:, :, z]
            slice_seg = seg_3d[:, :, z]

            non_zero_brain = np.sum(slice_t1n != 0)
            if non_zero_brain < args.min_brain_pixels:
                continue

            tumor_pixel_count = int(np.sum(slice_seg > 0))
            patient_candidates.append({
                "z": z,
                "slice_t1n": slice_t1n,
                "slice_t1c": t1c_norm[:, :, z],
                "slice_t2w": t2w_norm[:, :, z],
                "slice_t2f": t2f_norm[:, :, z],
                "slice_seg": slice_seg,
                "non_zero_brain": non_zero_brain,
                "tumor_pixel_count": tumor_pixel_count,
            })

        if not patient_candidates:
            continue

        selected_slices = select_patient_slices(patient_candidates, slices_per_patient=args.slices_per_patient)

        for item in selected_slices:
            z = item["z"]
            slice_seg = item["slice_seg"]
            tumor_pixel_count = item["tumor_pixel_count"]
            has_tumor = tumor_pixel_count > 0

            stacked_slice = np.stack([
                item["slice_t1n"],
                item["slice_t1c"],
                item["slice_t2w"],
                item["slice_t2f"]
            ], axis=0)  # Shape: [4, H, W]

            patient_records.append({
                "patient_id": patient_id,
                "slice_index": z,
                "tumor_volume": tumor_volume,
                "tumor_pixel_count": tumor_pixel_count,
                "has_tumor": has_tumor,
                "dir_path": p_dir,
                "stacked_slice": stacked_slice,
                "mask_slice": (slice_seg > 0).astype(np.float32)
            })

    if not patient_records:
        print("Error: No valid patient slices extracted!")
        return

    df = pd.DataFrame(patient_records)
    
    # Patient-level binning for stratified split
    patient_summary = df.groupby("patient_id")["tumor_volume"].first().reset_index()
    active_mask = patient_summary["tumor_volume"] > 0
    if active_mask.sum() >= 5:
        patient_summary.loc[active_mask, "bin"] = pd.qcut(
            patient_summary.loc[active_mask, "tumor_volume"], q=4, labels=[1, 2, 3, 4]
        ).astype(int)
        patient_summary.loc[~active_mask, "bin"] = 0
    else:
        patient_summary["bin"] = 0

    # 70/10/20 train/val/test split by patient ID
    try:
        train_ids, temp_ids = train_test_split(
            patient_summary["patient_id"].values, test_size=0.3, stratify=patient_summary["bin"].values, random_state=args.seed
        )
        temp_df = patient_summary[patient_summary["patient_id"].isin(temp_ids)]
        val_ids, test_ids = train_test_split(
            temp_df["patient_id"].values, test_size=2/3, stratify=temp_df["bin"].values, random_state=args.seed
        )
    except ValueError:
        all_ids = patient_summary["patient_id"].values
        n_train = max(1, int(len(all_ids) * 0.7))
        train_ids = all_ids[:n_train]
        remaining = all_ids[n_train:]
        val_ids = remaining[:1] if len(remaining) > 0 else train_ids
        test_ids = remaining[1:] if len(remaining) > 1 else val_ids

    split_map = {}
    for p_id in train_ids:
        split_map[p_id] = "train"
    for p_id in val_ids:
        split_map[p_id] = "val"
    for p_id in test_ids:
        split_map[p_id] = "test"

    df["split"] = df["patient_id"].map(split_map)

    final_records = []
    print("\nSaving 2D slices to disk...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Writing 2D slices"):
        patient_id = row["patient_id"]
        z = row["slice_index"]
        npz_filename = f"{patient_id}_z{z:03d}_slice.npz"
        npz_filepath = output_dir / npz_filename
        
        np.savez_compressed(
            str(npz_filepath),
            image=row["stacked_slice"],  # [4, H, W]
            mask=row["mask_slice"]       # [H, W]
        )
        
        final_records.append({
            "slice_id": f"{patient_id}_z{z:03d}",
            "patient_id": patient_id,
            "slice_index": z,
            "tumor_volume": row["tumor_volume"],
            "tumor_pixel_count": row["tumor_pixel_count"],
            "has_tumor": row["has_tumor"],
            "split": row["split"],
            "file_path": f"2d_slices/{npz_filename}"
        })

    metadata_df = pd.DataFrame(final_records)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nBraTS-GLI 5-slice preprocessing complete! Saved {len(metadata_df)} total 2D slices.")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
