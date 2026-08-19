import numpy as np
import torch
from monai.transforms import (
    CastToTyped,
    Compose,
    RandFlipd,
    RandRotated,
    SpatialPadd,
)


def get_segmentation_transforms(split: str = "train", spatial_size=(240, 240)):
    """Returns data augmentation and pre-processing pipeline for downstream segmentation."""
    if split == "train":
        return Compose([
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotated(keys=["image", "label"], range_x=0.3, prob=0.5, mode=("bilinear", "nearest")),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.float32)),
        ])
    else:
        return Compose([
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.float32)),
        ])

class JEPAMaskingTransform:
    """
    Generates fixed-size block patch context and target masks for JEPA models (I-JEPA, SigReg, VisReg).
    Ensures uniform tensor shapes across batches for seamless PyTorch collation.
    """
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 16,
        num_target_masks: int = 4,
        context_block_size: tuple[int, int] = (14, 14),
        target_block_size: tuple[int, int] = (5, 5),
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.num_target_masks = num_target_masks
        
        self.context_h, self.context_w = context_block_size
        self.target_h, self.target_w = target_block_size

    def _sample_fixed_block_mask(self, block_h: int, block_w: int) -> np.ndarray:
        """Samples a rectangular block of patch indices with fixed height and width at random grid coordinates."""
        h = min(block_h, self.grid_h)
        w = min(block_w, self.grid_w)
        
        top = np.random.randint(0, self.grid_h - h + 1)
        left = np.random.randint(0, self.grid_w - w + 1)
        
        mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        mask[top : top + h, left : left + w] = True
        return np.where(mask.flatten())[0]

    def __call__(self, x: torch.Tensor) -> dict:
        """
        Returns context patch indices [N_ctx] and list of target patch indices [N_tgt].
        x: [C, H, W]
        """
        context_indices = self._sample_fixed_block_mask(self.context_h, self.context_w)
        target_masks = []
        for _ in range(self.num_target_masks):
            target_idx = self._sample_fixed_block_mask(self.target_h, self.target_w)
            target_masks.append(torch.tensor(target_idx, dtype=torch.long))
            
        return {
            "image": x,
            "context_indices": torch.tensor(context_indices, dtype=torch.long),
            "target_indices": target_masks,
        }
