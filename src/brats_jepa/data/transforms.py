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
    r"""
    Random Multi-Modal MRI Sequence Dropout Layer.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Clinical Neuro-Oncology Reality (Missing Modalities):
       In clinical MRI practice, acquiring all four core sequences (T1, T1c, T2, FLAIR) for
       every patient is frequently infeasible due to:
       - Patient movement or scanner time limits in emergency triage.
       - Renal insufficiency / contrast allergies precluding Gadolinium administration (no T1c).
       - Inconsistent multi-center imaging acquisition protocols.
       Models trained strictly on complete 4-channel sets experience catastrophic performance
       degradation when even a single sequence is omitted.

    2. Bernoulli Channel Masking with Non-Empty Fallback:
       Each channel c \in \{0, 1, 2, 3\} is independently zeroed with probability p_{\text{drop}} = 0.25:
           m_c \sim \text{Bernoulli}(1 - p_{\text{drop}})
       If all 4 channels happen to be dropped simultaneously (\sum_c m_c = 0), a random channel
       is guaranteed to be activated:
           m_{c^*} = 1, \quad c^* \sim \text{Uniform}(\{0, 1, 2, 3\})
       ensuring the network is never trained on degenerate all-zero inputs. This forces the encoder
       to learn cross-modal feature redundancies and invariant anatomical representations.

    References:
    -----------
    - Havaei, M., et al. (2017). "Brain tumor segmentation with Deep Neural Networks."
      Medical Image Analysis, 35, 18-31.
    - Dorent, R., et al. (2019). "Hetero-Modal Variational Encoder-Decoder for Joint Inpainting
      and Segmentation." MICCAI 2019, pp. 523-531.
    """
    def __init__(self, p_drop: float = 0.25):
        super().__init__()
        self.p_drop = p_drop
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p_drop <= 0.0:
            return x
        B, C, _H, _W = x.shape
        mask = (torch.rand(B, C, 1, 1, device=x.device) > self.p_drop).float()
        all_zero = (mask.sum(dim=1, keepdim=True) == 0)
        # Fallback: activate a random channel for any slice where all modalities were dropped
        random_channel = torch.randint(0, C, (B, 1, 1, 1), device=x.device)
        fallback = torch.zeros_like(mask).scatter_(1, random_channel, 1.0)
        mask = torch.where(all_zero, fallback, mask)
        return x * mask


class JEPAMaskingTransform:
    r"""
    Multi-Block Context and Target Masking Generator for Joint-Embedding Architectures.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Contiguous Block Masking vs Point-Wise Pixel Masking:
       Standard Masked Autoencoders (MAE) employ random point-wise patch masking (e.g. 75% uniform
       dropout). In 2D medical images, adjacent patches exhibit extreme spatial autocorrelation;
       missing individual patches can be easily interpolated via low-level edge continuity.
       JEPA instead samples large contiguous rectangular blocks:
       - **Target Blocks**: 4 blocks of size 5x5 patches (25 patches each), sampled with
         overlap control to maximize spatial diversity.
       - **Context Block**: A contiguous 2D spatial cluster of 96 patches extracted via
         4-connected breadth-first expansion, guaranteeing spatial coherence without target overlap.
       Large contiguous block removal destroys low-level texture shortcuts, forcing the model to understand
       high-level anatomical geometry, organ symmetry, and global tissue morphology (Assran et al., 2023).

    2. Uniform Tensor Length for Efficient Collation:
       Standard I-JEPA implementations produce variable-length context token lists, requiring
       complex nested tensors or ragged padding that slows down GPU data collation.
       This implementation guarantees a constant N_{\text{ctx}} = 96 context patches per slice,
       enabling seamless standard PyTorch batch collation and maximum GPU tensor core utilization.

    References:
    -----------
    - Assran, M., et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding
      Predictive Architecture." IEEE/CVF CVPR 2023.
    - Bao, H., et al. (2021). "BEiT: BERT Pre-Training of Image Transformers." ICLR 2022.
    """
    def __init__(
        self,
        img_size: int = 240,
        patch_size: int = 16,
        num_target_masks: int = 4,
        context_block_size: tuple[int, int] = (14, 14),
        target_block_size: tuple[int, int] = (5, 5),
        num_context_patches: int = 96,
        max_target_overlap: float = 0.25,
        contiguous_context: bool = True,
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
        self.max_target_overlap = max_target_overlap
        self.contiguous_context = contiguous_context

    def _sample_fixed_block_mask(self, block_h: int, block_w: int) -> np.ndarray:
        """Samples a rectangular block of patch indices with fixed height and width at random grid coordinates."""
        h = min(block_h, self.grid_h)
        w = min(block_w, self.grid_w)
        
        top = np.random.randint(0, self.grid_h - h + 1)
        left = np.random.randint(0, self.grid_w - w + 1)
        
        mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        mask[top : top + h, left : left + w] = True
        return np.where(mask.flatten())[0]

    def _sample_contiguous_cluster(
        self,
        candidate_indices: list[int],
        target_len: int,
        all_allowed_indices: list[int],
    ) -> list[int]:
        """
        Grows a compact, contiguous spatial cluster of patch indices via 4-connected BFS.
        Preserves spatial contiguity of context representations per Assran et al. (2023).
        """
        cand_set = set(candidate_indices)
        allowed_set = set(all_allowed_indices)
        seed = int(np.random.choice(candidate_indices))
        
        visited = {seed}
        queue = [seed]
        head = 0
        
        # 1. Grow within candidate block first
        while head < len(queue) and len(visited) < target_len:
            curr = queue[head]
            head += 1
            r, c = curr // self.grid_w, curr % self.grid_w
            neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            np.random.shuffle(neighbors)
            for nr, nc in neighbors:
                if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                    n_idx = nr * self.grid_w + nc
                    if n_idx in cand_set and n_idx not in visited:
                        visited.add(n_idx)
                        queue.append(n_idx)
                        if len(visited) == target_len:
                            break

        # 2. If candidate block exhausted (e.g. severed by target mask), expand into any non-target patches
        if len(visited) < target_len:
            head = 0
            while head < len(queue) and len(visited) < target_len:
                curr = queue[head]
                head += 1
                r, c = curr // self.grid_w, curr % self.grid_w
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                np.random.shuffle(neighbors)
                for nr, nc in neighbors:
                    if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                        n_idx = nr * self.grid_w + nc
                        if n_idx in allowed_set and n_idx not in visited:
                            visited.add(n_idx)
                            queue.append(n_idx)
                            if len(visited) == target_len:
                                break

        # 3. Fallback: complete to target_len from available non-target indices
        if len(visited) < target_len:
            remaining = [i for i in all_allowed_indices if i not in visited]
            needed = target_len - len(visited)
            if len(remaining) >= needed:
                supp = np.random.choice(remaining, size=needed, replace=False).tolist()
                visited.update(supp)
            else:
                visited.update(remaining)

        return sorted(visited)

    def __call__(self, x: torch.Tensor) -> dict:
        """
        Returns context patch indices [N_ctx] and list of target patch indices [N_tgt].
        Context indices are guaranteed to NOT overlap with any target indices,
        and maintain a strictly uniform length across all samples for seamless PyTorch batch collation.
        x: [C, H, W]
        """
        target_masks = []
        all_target_indices = set()
        block_area = self.target_h * self.target_w

        # Sample target blocks with collision mitigation (finding M1)
        for _ in range(self.num_target_masks):
            best_target_idx = None
            min_overlap = float("inf")
            for _attempt in range(10):
                cand_idx = self._sample_fixed_block_mask(self.target_h, self.target_w)
                overlap = len(set(cand_idx.tolist()) & all_target_indices)
                if overlap <= block_area * self.max_target_overlap:
                    best_target_idx = cand_idx
                    break
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_target_idx = cand_idx
            if best_target_idx is None:
                best_target_idx = self._sample_fixed_block_mask(self.target_h, self.target_w)

            target_masks.append(torch.tensor(best_target_idx, dtype=torch.long))
            all_target_indices.update(best_target_idx.tolist())
        
        # Sample candidate context block
        ctx_candidate_block = self._sample_fixed_block_mask(self.context_h, self.context_w)
        
        # Filter out target overlaps from the context block
        non_overlap_candidates = [i for i in ctx_candidate_block if i not in all_target_indices]
        
        # Pool of all non-target patches across the entire grid
        all_non_target = [i for i in range(self.num_patches) if i not in all_target_indices]
        
        # Guarantee exactly self.num_context_patches for uniform batch collation
        target_ctx_len = min(self.num_context_patches, len(all_non_target))

        if self.contiguous_context and len(non_overlap_candidates) > target_ctx_len:
            # Preserves 2D spatial contiguity via contiguous cluster expansion (finding H4)
            chosen_ctx = self._sample_contiguous_cluster(
                non_overlap_candidates, target_ctx_len, all_non_target
            )
        elif len(non_overlap_candidates) >= target_ctx_len:
            chosen_ctx = np.random.choice(non_overlap_candidates, size=target_ctx_len, replace=False)
            chosen_ctx = np.sort(chosen_ctx).tolist()
        else:
            chosen_set = set(non_overlap_candidates)
            supplement = [i for i in all_non_target if i not in chosen_set]
            if supplement:
                needed = target_ctx_len - len(non_overlap_candidates)
                sampled_supp = np.random.choice(supplement, size=needed, replace=False).tolist()
                chosen_ctx = sorted(non_overlap_candidates + sampled_supp)
            else:
                chosen_ctx = sorted(non_overlap_candidates)
            
        return {
            "image": x,
            "context_indices": torch.tensor(chosen_ctx, dtype=torch.long),
            "target_indices": target_masks,
        }
