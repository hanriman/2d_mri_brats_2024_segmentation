import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def dummy_batch():
    """Generates a dummy 4-channel image batch [B=2, C=4, H=240, W=240] and label [B=2, 1, 240, 240]."""
    images = torch.randn(2, 4, 240, 240)
    labels = torch.randint(0, 2, (2, 1, 240, 240)).float()
    return {"image": images, "label": labels}

@pytest.fixture
def dummy_dataset_dir():
    """Creates a temporary dataset directory with dummy .npz slices and metadata.csv."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        slice_dir = tmp_path / "2d_slices"
        slice_dir.mkdir(parents=True, exist_ok=True)
        
        npz_name = "BraTS-GLI-00001-100_slice.npz"
        npz_file = tmp_path / npz_name
        np.savez_compressed(
            str(npz_file),
            image=np.random.randn(4, 240, 240).astype(np.float32),
            mask=np.random.randint(0, 2, (240, 240)).astype(np.float32)
        )
        
        metadata = pd.DataFrame([{
            "patient_id": "BraTS-GLI-00001",
            "slice_index": 100,
            "tumor_volume": 5000.0,
            "stratify_bin": 1,
            "split": "train",
            "file_path": f"2d_slices/{npz_name}"
        }, {
            "patient_id": "BraTS-GLI-00002",
            "slice_index": 101,
            "tumor_volume": 6000.0,
            "stratify_bin": 1,
            "split": "val",
            "file_path": f"2d_slices/{npz_name}"
        }])
        
        csv_path = tmp_path / "metadata.csv"
        metadata.to_csv(csv_path, index=False)
        yield csv_path
