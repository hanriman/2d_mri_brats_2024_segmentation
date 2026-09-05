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

class RandomModalityDropout(torch.nn.Module):
    """
    Randomly drops (zeros out) 1, 2, or 3 modality channels during training with probability p_drop.
    Guarantees that at least one modality channel remains active for every slice in the batch.
    """
    def __init__(self, p_drop: float = 0.25):
        super().__init__()
        self.p_drop = p_drop
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_drop <= 0.0:
            return x
        B, C, H, W = x.shape
        mask = (torch.rand(B, C, 1, 1, device=x.device) > self.p_drop).float()
        all_zero = (mask.sum(dim=1, keepdim=True) == 0)
        # Fallback: activate a random channel for any slice where all modalities were dropped
        random_channel = torch.randint(0, C, (B, 1, 1, 1), device=x.device)
        fallback = torch.zeros_like(mask).scatter_(1, random_channel, 1.0)
        mask = torch.where(all_zero, fallback, mask)
        return x * mask

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
        num_context_patches: int = 96,
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_patches = self.grid_h * self.grid_w
        self.num_target_masks = num_target_masks
        
        self.context_h, self.context_w = context_block_size
        self.target_h, self.target_w = target_block_size
        self.num_context_patches = num_context_patches

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
        Context indices are guaranteed to NOT overlap with any target indices,
        and maintain a strictly uniform length across all samples for seamless PyTorch batch collation.
        x: [C, H, W]
        """
        target_masks = []
        all_target_indices = set()
        for _ in range(self.num_target_masks):
            target_idx = self._sample_fixed_block_mask(self.target_h, self.target_w)
            target_masks.append(torch.tensor(target_idx, dtype=torch.long))
            all_target_indices.update(target_idx.tolist())
        
        # Sample candidate context block
        ctx_candidate_block = self._sample_fixed_block_mask(self.context_h, self.context_w)
        
        # Filter out target overlaps from the context block
        non_overlap_candidates = [i for i in ctx_candidate_block if i not in all_target_indices]
        
        # Pool of all non-target patches across the entire grid
        all_non_target = [i for i in range(self.num_patches) if i not in all_target_indices]
        
        # Guarantee exactly self.num_context_patches for uniform batch collation
        target_ctx_len = min(self.num_context_patches, len(all_non_target))
        if len(non_overlap_candidates) >= target_ctx_len:
            chosen_ctx = non_overlap_candidates[:target_ctx_len]
        else:
            chosen_set = set(non_overlap_candidates)
            supplement = [i for i in all_non_target if i not in chosen_set]
            np.random.shuffle(supplement)
            chosen_ctx = non_overlap_candidates + supplement[:target_ctx_len - len(non_overlap_candidates)]
            
        return {
            "image": x,
            "context_indices": torch.tensor(chosen_ctx, dtype=torch.long),
            "target_indices": target_masks,
        }
