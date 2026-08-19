from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .transforms import get_segmentation_transforms


class BraTS2DDataset(Dataset):
    """
    PyTorch Dataset for loading 2D multimodal brain tumor slices (T1, T1c, T2, FLAIR).
    Loads 4-channel image tensors and matching binary tumor segmentation masks (Whole Tumor).
    """
    def __init__(
        self,
        metadata_csv: str | Path,
        split: str = "train",
        transforms: Callable | None = None,
        jepa_masking: Callable | None = None,
    ):
        super().__init__()
        self.metadata_csv = Path(metadata_csv).resolve()
        self.split = split
        self.data_root = self.metadata_csv.parent
        self.jepa_masking = jepa_masking
        
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
        
        file_name = Path(record["file_path"]).name
        npz_path = self.data_root / file_name
        data = np.load(str(npz_path))
        
        image = data["image"]  # shape: [4, H, W] (T1, T1c, T2, FLAIR)
        mask = data["mask"]    # shape: [H, W] (BraTS discrete tumor labels 0,1,2,3)
        
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
