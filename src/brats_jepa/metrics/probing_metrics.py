
import torch
import torch.nn.functional as F


def compute_effective_rank(z: torch.Tensor) -> float:
    """
    Computes the Effective Rank (Entropy of singular values) of latent features z [N, D].
    Higher value indicates richer, non-collapsed representation space.
    """
    z_centered = z - z.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.svd(z_centered)
        singular_values = S / S.sum()
        # Shannon entropy of normalized singular values
        entropy = -torch.sum(singular_values * torch.log(singular_values + 1e-12))
        eff_rank = torch.exp(entropy).item()
        return eff_rank
    except (RuntimeError, ValueError):
        return 1.0

def compute_representation_collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    """
    Evaluates representation quality metrics:
    - effective_rank: Dimensionality utilization
    - avg_cosine_similarity: Pairwise feature similarity (high = collapse)
    - feature_variance: Average feature variance
    """
    z_flat = z.reshape(-1, z.shape[-1])  # [N, D]
    z_norm = F.normalize(z_flat, p=2, dim=-1)
    
    # Sample subset for pairwise cosine similarity if large
    if z_norm.shape[0] > 1000:
        indices = torch.randperm(z_norm.shape[0])[:1000]
        z_sample = z_norm[indices]
    else:
        z_sample = z_norm
        
    sim_matrix = z_sample @ z_sample.T
    N = z_sample.shape[0]
    mask = ~torch.eye(N, device=z.device, dtype=torch.bool)
    avg_cosine_sim = sim_matrix[mask].mean().item()
    
    eff_rank = compute_effective_rank(z_flat)
    feature_var = z_flat.var(dim=0).mean().item()
    
    return {
        "effective_rank": eff_rank,
        "avg_cosine_sim": avg_cosine_sim,
        "feature_variance": feature_var,
    }
