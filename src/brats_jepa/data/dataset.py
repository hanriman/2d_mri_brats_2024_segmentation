from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .transforms import get_segmentation_transforms


class BraTS2DDataset(Dataset):
    r"""
    Multi-Modal 2D Brain Tumor Slice Dataset (BraTS 2024 Benchmark).

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Multi-Modal MRI Physical Complementarity:
       Each 2D slice incorporates 4 core structural MRI sequences [C=4, H=240, W=240]:
       - **T1-weighted**: High-resolution anatomical contrast; clearly defines healthy gray and
         white matter parenchyma.
       - **T1-contrast (T1c)**: Gadolinium contrast-enhancement reveals vascular hyperpermeability
         and blood-brain barrier disruption, delineating viable active tumor boundaries.
       - **T2-weighted**: Demonstrates long transverse relaxation; highlights free water and vasogenic
         peritumoral edema.
       - **T2-FLAIR**: Suppresses free cerebrospinal fluid (CSF) signal, isolating subtle hyperintense
         infiltrative tumor margin signals that would otherwise blend into the ventricles.

    2. Whole Tumor (WT) Target Formulation:
       BraTS voxels contain multiclass sub-compartments: necrotic core (label 1), peritumoral
       edema (label 2), and enhancing tumor (label 3). In accordance with the clinical Whole Tumor (WT)
       evaluation standard, masks are binarized:
           \text{WT}(x, y) = \mathbb{I}(\text{mask}(x, y) > 0)
       capturing the entire contiguous pathologic lesion.

    3. Patient-Level Splitting & Leakage Prevention:
       Splits (train, val, test) are partitioned strictly by `patient_id` rather than random slice
       indexing. Because adjacent 2D axial slices from the same 3D volume share substantial spatial
       and anatomical correlation, patient-level isolation is mandatory to ensure valid out-of-sample
       generalization without test data leakage.

    References:
    -----------
    - Menze, B. H., et al. (2014). "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)."
      IEEE Transactions on Medical Imaging, 34(10), 1993-2024.
    - Baid, U., et al. (2021). "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
      Segmentation and Radiogenomic Classification." arXiv:2107.02314.
    """
    def __init__(
        self,
        metadata_csv: str | Path,
        split: str = "train",
        transforms: Callable | None = None,
        jepa_masking: Callable | None = None,
        cache_in_memory: bool = False,
    ):
        super().__init__()
        self.metadata_csv = Path(metadata_csv).resolve()
        self.split = split
        self.data_root = self.metadata_csv.parent
        self.jepa_masking = jepa_masking
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata manifest not found at: {self.metadata_csv}")
            
        df = pd.read_csv(self.metadata_csv)
        self.records = df[df["split"] == self.split].to_dict(orient="records")
        
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = get_segmentation_transforms(split=self.split)
            
    def __len__(self) -> int:
        return len(self.records)
        
    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        patient_id = record["patient_id"]
        
        if self.cache_in_memory and idx in self._cache:
            image, mask = self._cache[idx]
        else:
            file_name = Path(record["file_path"]).name
            npz_path = self.data_root / file_name
            data = np.load(str(npz_path))
            image = data["image"]  # shape: [4, H, W] (T1, T1c, T2, FLAIR)
            mask = data["mask"]    # shape: [H, W] (BraTS discrete tumor labels 0,1,2,3)
            if self.cache_in_memory:
                self._cache[idx] = (image, mask)
        
        # Binarize mask for Whole Tumor (WT) segmentation (1 = active tumor, 0 = background)
        binary_mask = (mask > 0).astype(np.float32)
        
        data_dict = {
            "image": image,
            "label": np.expand_dims(binary_mask, axis=0)  # shape: [1, H, W]
        }
        
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
            
        image_tensor = data_dict["image"]
        label_tensor = (data_dict["label"] > 0).float()  # Ensure strict binary float tensor
        
        output = {
            "image": image_tensor,
            "label": label_tensor,
            "patient_id": patient_id,
            "slice_index": record["slice_index"],
        }
        
        if self.jepa_masking is not None:
            mask_info = self.jepa_masking(image_tensor)
            output["context_indices"] = mask_info["context_indices"]
            output["target_indices"] = mask_info["target_indices"]
            
        return output
