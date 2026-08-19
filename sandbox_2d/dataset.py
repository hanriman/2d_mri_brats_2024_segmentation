from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    CastToTyped,
    Compose,
    RandFlipd,
    RandRotated,
    SpatialPadd,
)
from torch.utils.data import Dataset


class BraTS2DDataset(Dataset):
    """
    PyTorch Dataset for loading 2D multimodal brain tumor slices.
    Loads stacked 4-channel image and matching 2D segmentation label from compressed .npz.
    """
    def __init__(self, metadata_csv: str, split: str = "train", transforms=None):
        super().__init__()
        self.metadata_csv = Path(metadata_csv).resolve()
        self.split = split
        self.data_root = self.metadata_csv.parent
        
        # Load and filter manifest
        if not self.metadata_csv.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_csv}")
            
        df = pd.read_csv(self.metadata_csv)
        self.records = df[df["split"] == self.split].to_dict(orient="records")
        
        self.transforms = transforms if transforms is not None else self.get_default_transforms()
        
    def get_default_transforms(self):
        """Returns standard data loading and normalization transforms."""
        if self.split == "train":
            return Compose([
                SpatialPadd(keys=["image", "label"], spatial_size=(240, 240)),
                # Random Augmentations for 2D training
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandRotated(keys=["image", "label"], range_x=0.3, prob=0.5, mode=("bilinear", "nearest")),
                CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.float32)),
            ])
        else:
            # Deterministic transforms for validation/testing
            return Compose([
                SpatialPadd(keys=["image", "label"], spatial_size=(240, 240)),
                CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.float32)),
            ])
            
    def __len__(self):
        return len(self.records)
        
    def __getitem__(self, idx):
        record = self.records[idx]
        patient_id = record["patient_id"]
        
        # Load .npz slice data
        npz_path = self.data_root / Path(record["file_path"]).name
        data = np.load(str(npz_path))
        
        image = data["image"]  # shape: [4, H, W] (T1, T1c, T2, FLAIR)
        mask = data["mask"]    # shape: [H, W]
        
        # Format as dictionaries for MONAI transform compatibility
        # Add channel dimension to label -> shape [1, H, W]
        data_dict = {
            "image": image,
            "label": np.expand_dims(mask, axis=0)
        }
        
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
            
        # Return tensors
        return {
            "image": data_dict["image"],
            "label": data_dict["label"],
            "patient_id": patient_id,
            "slice_index": record["slice_index"]
        }
