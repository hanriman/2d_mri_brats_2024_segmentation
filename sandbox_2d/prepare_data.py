import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 2D Axial Slices with largest tumor area from 3D BraTS GLI dataset")
    parser.add_argument("--data_dir", type=str, default="../data/raw/BraTS_2024/BraTS-GLI/training_data1_v2",
                        help="Input raw data directory containing patient folders")
    parser.add_argument("--output_dir", type=str, default="../data/processed/2d_slices",
                        help="Output directory to save 2D slices")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of patient scans to preprocess (for debugging)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for data splitting reproducibility")
    return parser.parse_args()

def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize volume intensities using Z-score calculated across non-zero brain region."""
    mask = volume > 0
    if not np.any(mask):
        return volume
    mean = volume[mask].mean()
    std = volume[mask].std()
    if std > 0:
        normalized = (volume - mean) / std
    else:
        normalized = volume - mean
    # Zero out background voxels
    normalized[~mask] = 0.0
    return normalized

def main():
    args = parse_args()
    
    # Resolve paths relative to the scripts location
    script_dir = Path(__file__).resolve().parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    print(f"Reading raw data from: {data_dir}")
    print(f"Saving processed slices to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Find all patient folders
    patient_dirs = []
    for p in data_dir.iterdir():
        if p.is_dir() and not p.name.startswith('.'):
            # Check for modalities
            t1n_files = list(p.glob("*-t1n.nii.gz"))
            if len(t1n_files) > 0:
                patient_dirs.append(p)
                
    patient_dirs = sorted(patient_dirs)
    if args.limit:
        patient_dirs = patient_dirs[:args.limit]
        print(f"Limiting preprocessing to first {args.limit} cases.")
        
    if len(patient_dirs) == 0:
        print(f"Error: No patient directories found in {data_dir}")
        return
        
    print(f"Found {len(patient_dirs)} patient cases. Phase 1: Analyzing tumor volumes...")
    
    # 2. Gather patient cases and calculate tumor volumes for stratification
    patient_records = []
    for p_dir in tqdm(patient_dirs, desc="Analyzing tumor volumes"):
        patient_id = p_dir.name
        
        # Locate files
        seg_paths = list(p_dir.glob("*-seg.nii.gz"))
        if len(seg_paths) == 0:
            # Skip if no segmentation file (pre-training cases might lack ground truth, but we need seg for 2D slicing)
            continue
            
        seg_path = seg_paths[0]
        
        # Load seg to calculate total tumor volume
        seg_img = nib.load(str(seg_path)).get_fdata()
        tumor_volume = np.sum(seg_img > 0)
        
        patient_records.append({
            "patient_id": patient_id,
            "dir_path": p_dir,
            "tumor_volume": float(tumor_volume)
        })
        
    if len(patient_records) == 0:
        print("Error: No segmentation masks found! 2D slice extraction requires segmentation files.")
        return
        
    # 3. Discretize tumor sizes into bins for stratified splitting
    df = pd.DataFrame(patient_records)
    
    # Define bins (0 = no tumor, and then 4 quantiles for active tumors)
    active_tumor_mask = df["tumor_volume"] > 0
    if active_tumor_mask.sum() >= 5:
        # Quantile binning for active tumors
        df.loc[active_tumor_mask, "bin"] = pd.qcut(df.loc[active_tumor_mask, "tumor_volume"], q=4, labels=[1, 2, 3, 4]).astype(int)
        df.loc[~active_tumor_mask, "bin"] = 0
    else:
        df["bin"] = 0
        
    df["bin"] = df["bin"].astype(int)
    
    # 4. Split Dataset (70/10/20)
    print("Performing splitting based on tumor volume...")
    try:
        # First split: train (70%) and temp (30%)
        train_ids, temp_ids = train_test_split(
            df["patient_id"].values,
            test_size=0.3,
            stratify=df["bin"].values,
            random_state=args.seed
        )
        
        # Filter temp df to stratify again
        temp_df = df[df["patient_id"].isin(temp_ids)]
        
        # Second split: val (10% total -> 1/3 of temp) and test (20% total -> 2/3 of temp)
        val_ids, test_ids = train_test_split(
            temp_df["patient_id"].values,
            test_size=2/3,
            stratify=temp_df["bin"].values,
            random_state=args.seed
        )
    except ValueError:
        print(f"Split allocation failed (likely due to too few samples: {len(df)}). Manually partitioning...")
        all_ids = df["patient_id"].values
        n_train = max(1, int(len(df) * 0.7))
        train_ids = all_ids[:n_train]
        remaining = all_ids[n_train:]
        if len(remaining) > 1:
            val_ids = remaining[:1]
            test_ids = remaining[1:]
        else:
            val_ids = remaining
            test_ids = remaining
    
    # Map splits back to DataFrame
    df.loc[df["patient_id"].isin(train_ids), "split"] = "train"
    df.loc[df["patient_id"].isin(val_ids), "split"] = "val"
    df.loc[df["patient_id"].isin(test_ids), "split"] = "test"
    
    # 5. Extract 2D slices and serialize
    print("Phase 2: Extracting 2D slices and writing .npz files...")
    
    final_records = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
        patient_id = row["patient_id"]
        p_dir = row["dir_path"]
        
        # Load files
        t1n_path = next(p_dir.glob("*-t1n.nii.gz"))
        t1c_path = next(p_dir.glob("*-t1c.nii.gz"))
        t2w_path = next(p_dir.glob("*-t2w.nii.gz"))
        t2f_path = next(p_dir.glob("*-t2f.nii.gz"))
        seg_path = next(p_dir.glob("*-seg.nii.gz"))
        
        # Load 3D volumes
        t1n_3d = nib.load(str(t1n_path)).get_fdata().astype(np.float32)
        t1c_3d = nib.load(str(t1c_path)).get_fdata().astype(np.float32)
        t2w_3d = nib.load(str(t2w_path)).get_fdata().astype(np.float32)
        t2f_3d = nib.load(str(t2f_path)).get_fdata().astype(np.float32)
        seg_3d = nib.load(str(seg_path)).get_fdata().astype(np.float32)
        
        # Find axial slice index z_max with largest tumor area
        slice_sums = np.sum(seg_3d > 0, axis=(0, 1))
        z_max = int(np.argmax(slice_sums))
        
        # Fallback to middle slice if mask is completely empty
        if slice_sums[z_max] == 0:
            z_max = seg_3d.shape[2] // 2
            
        # Z-score normalize 3D volumes first to capture full context distribution
        t1n_norm = zscore_normalize(t1n_3d)
        t1c_norm = zscore_normalize(t1c_3d)
        t2w_norm = zscore_normalize(t2w_3d)
        t2f_norm = zscore_normalize(t2f_3d)
        
        # Extract 2D slices at z_max
        slice_t1n = t1n_norm[:, :, z_max]
        slice_t1c = t1c_norm[:, :, z_max]
        slice_t2w = t2w_norm[:, :, z_max]
        slice_t2f = t2f_norm[:, :, z_max]
        slice_seg = seg_3d[:, :, z_max]
        
        # Stack to 4 channels: shape [4, H, W]
        stacked_slice = np.stack([slice_t1n, slice_t1c, slice_t2w, slice_t2f], axis=0)
        
        # Save as compressed .npz
        npz_filename = f"{patient_id}_slice.npz"
        npz_filepath = output_dir / npz_filename
        np.savez_compressed(
            str(npz_filepath),
            image=stacked_slice, # [4, H, W]
            mask=slice_seg       # [H, W]
        )
        
        final_records.append({
            "patient_id": patient_id,
            "slice_index": z_max,
            "tumor_volume": row["tumor_volume"],
            "stratify_bin": row["bin"],
            "split": row["split"],
            "file_path": f"2d_slices/{npz_filename}"
        })
        
    # Write metadata manifest file
    metadata_df = pd.DataFrame(final_records)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    
    print("\nPreprocessing Complete!")
    print(f"Saved metadata file to: {metadata_path}")
    print(f"Total processed slices saved: {len(metadata_df)}")
    print(metadata_df["split"].value_counts())

if __name__ == "__main__":
    main()
