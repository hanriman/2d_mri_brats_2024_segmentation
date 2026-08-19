# Data Dictionary: 2D BraTS GLI Slice Dataset

## Overview

The dataset consists of 2D axial slices extracted from 3D multi-modal Magnetic Resonance Imaging (MRI) scans of glioma patients from the Brain Tumor Segmentation (BraTS) GLI dataset.

---

## Modalities & Input Channels

Each 2D image sample is stored as a 4-channel tensor of shape `[4, 240, 240]`:

| Channel | Modality | Description | Biological / Diagnostic Purpose |
| :--- | :--- | :--- | :--- |
| **0** | **T1** | T1-weighted native MRI | Anatomy baseline, brain structure boundaries |
| **1** | **T1c** | Post-contrast T1-weighted MRI | Contrast enhancement, gadolinium uptake in active tumor border |
| **2** | **T2** | T2-weighted MRI | Edema visualization, fluid and hyperintensity detection |
| **3** | **FLAIR** | Fluid Attenuated Inversion Recovery | Peritumoral edema and non-enhancing tumor tissue |

---

## Target Label

* **Mask (`label`)**: Shape `[1, 240, 240]`. Binary mask indicating active tumor segmentation (1 = tumor region, 0 = background/brain tissue).

---

## File Format & Metadata Schema

Slices are saved as compressed NumPy `.npz` files under `data/processed/2d_slices/`.

Manifest file: `data/processed/2d_slices/metadata.csv`

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `patient_id` | string | Unique patient identification code (e.g., `BraTS-GLI-00005`) |
| `slice_index` | integer | Selected axial slice index $z_{\text{max}}$ with maximum tumor area |
| `tumor_volume` | float | Total 3D voxel count of tumor mask |
| `stratify_bin` | integer | Stratification bin used for balanced train/val/test allocation |
| `split` | string | Data partition (`train` = 70%, `val` = 10%, `test` = 20%) |
| `file_path` | string | Relative path to slice `.npz` file |
