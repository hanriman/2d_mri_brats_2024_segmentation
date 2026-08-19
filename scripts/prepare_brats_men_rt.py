import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from brats_jepa.config import DATA_DIR
from brats_jepa.utils import get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS-MEN-RT 3D to 2D Slice Processing Pipeline")
    parser.add_argument("--target_size", type=int, default=240, help="Target 2D slice height/width")
    parser.add_argument("--min_brain_pixels", type=int, default=1000, help="Minimum non-zero brain pixels per slice")
    parser.add_argument("--slices_per_patient", type=int, default=5, help="Number of 2D axial slices to sample per 3D volume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize volume using non-zero brain tissue statistics."""
    mask = volume > 0
    if not np.any(mask):
        return volume.astype(np.float32)
    mean = volume[mask].mean()
    std = volume[mask].std() + 1e-8
    normed = (volume - mean) / std
    normed[~mask] = 0.0
    return normed.astype(np.float32)

def resize_slice(slice_2d: torch.Tensor, target_size: int = 240, is_mask: bool = False) -> np.ndarray:
    """Resizes a 2D tensor slice to target_size x target_size."""
    # slice_2d: [1, H, W]
    mode = "nearest" if is_mask else "bilinear"
    align_corners = None if is_mask else False
    
    resized = F.interpolate(
        slice_2d.unsqueeze(0),  # [1, 1, H, W]
        size=(target_size, target_size),
        mode=mode,
        align_corners=align_corners
    ).squeeze(0).cpu().numpy()  # [1, target_size, target_size]
    return resized

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

    # Ensure exactly slices_per_patient if possible
    if len(selected) < slices_per_patient and len(remaining_pool) > 0:
        for c in remaining_pool:
            if c["z"] not in {s["z"] for s in selected}:
                selected.append(c)
                if len(selected) == slices_per_patient:
                    break

    return sorted(selected, key=lambda c: c["z"])

def main():
    args = parse_args()
    set_seed(args.seed)
    
    raw_dir = (DATA_DIR / "raw" / "BraTS-MEN-RT").resolve()
    output_dir = (DATA_DIR / "processed" / "brats_men_rt_2d").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger("prepare_brats_men_rt")
    logger.info(f"Processing BraTS-MEN-RT NIfTI dataset from: {raw_dir}")
    logger.info(f"Sampling target: {args.slices_per_patient} slices per 3D volume")
    
    # Find all patient directories containing *_t1c.nii.gz
    t1c_files = sorted(list(raw_dir.rglob("*_t1c.nii.gz")))
    logger.info(f"Found {len(t1c_files)} t1c NIfTI volumes.")
    
    records = []
    global_slice_idx = 0
    
    for t1c_path in tqdm(t1c_files, desc="Processing BraTS-MEN-RT volumes"):
        patient_dir = t1c_path.parent
        patient_id = patient_dir.name
        
        # Look for corresponding ground-truth mask (*_gtv.nii.gz or *_seg.nii.gz)
        gtv_candidates = list(patient_dir.glob("*_gtv.nii.gz")) + list(patient_dir.glob("*_seg.nii.gz"))
        has_gtv = len(gtv_candidates) > 0
        
        try:
            t1c_nifti = nib.load(str(t1c_path))
            t1c_vol = t1c_nifti.get_fdata()
            
            if has_gtv:
                gtv_nifti = nib.load(str(gtv_candidates[0]))
                gtv_vol = gtv_nifti.get_fdata()
            else:
                gtv_vol = np.zeros_like(t1c_vol)
        except Exception as e:
            logger.warning(f"Error loading {patient_id}: {e}")
            continue
            
        # Z-score normalize T1c volume
        t1c_norm = zscore_normalize(t1c_vol)
        
        num_slices = t1c_vol.shape[2]
        patient_candidates = []
        
        for z in range(num_slices):
            slice_t1c = t1c_norm[:, :, z]
            slice_gtv = gtv_vol[:, :, z]
            
            non_zero_brain = np.sum(slice_t1c != 0)
            if non_zero_brain < args.min_brain_pixels:
                continue
                
            tumor_pixel_count = int(np.sum(slice_gtv > 0))
            patient_candidates.append({
                "z": z,
                "slice_t1c": slice_t1c,
                "slice_gtv": slice_gtv,
                "non_zero_brain": non_zero_brain,
                "tumor_pixel_count": tumor_pixel_count,
            })
            
        if not patient_candidates:
            continue
            
        selected_slices = select_patient_slices(patient_candidates, slices_per_patient=args.slices_per_patient)
        
        for item in selected_slices:
            z = item["z"]
            slice_t1c = item["slice_t1c"]
            slice_gtv = item["slice_gtv"]
            tumor_pixel_count = item["tumor_pixel_count"]
            has_tumor = tumor_pixel_count > 0
            
            # Convert to PyTorch tensors for fast resizing
            t1c_t = torch.from_numpy(slice_t1c).float().unsqueeze(0)  # [1, H, W]
            gtv_t = torch.from_numpy((slice_gtv > 0).astype(np.float32)).float().unsqueeze(0)
            
            t1c_resized = resize_slice(t1c_t, target_size=args.target_size, is_mask=False)
            gtv_resized = resize_slice(gtv_t, target_size=args.target_size, is_mask=True)
            
            slice_filename = f"slice_men_rt_{global_slice_idx:05d}.npz"
            slice_path = output_dir / slice_filename
            
            np.savez_compressed(
                slice_path,
                image=t1c_resized,  # [1, 240, 240]
                mask=gtv_resized,    # [1, 240, 240]
            )
            
            records.append({
                "slice_id": f"men_rt_{global_slice_idx:05d}",
                "patient_id": patient_id,
                "slice_index": z,
                "file_path": str(slice_path.relative_to(DATA_DIR)),
                "has_gtv": has_gtv,
                "has_tumor": has_tumor,
                "tumor_pixel_count": tumor_pixel_count,
            })
            
            global_slice_idx += 1
            
    df = pd.DataFrame(records)
    csv_path = output_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"BraTS-MEN-RT preprocessing complete!")
    logger.info(f"Extracted {len(df)} total 2D axial slices across {len(t1c_files)} patients.")
    logger.info(f"Saved metadata manifest to: {csv_path}")

if __name__ == "__main__":
    main()
