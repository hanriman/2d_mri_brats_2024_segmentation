
import torch
import torch.nn.functional as F


def compute_effective_rank(z: torch.Tensor) -> float:
    r"""
    Effective Rank (Spectral Entropy of Empirical Covariance) of Latent Feature Tokens.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Information-Theoretic Dimensionality (Roy & Vetterli, 2007):
       Measures the effective dimensionality occupied by latent feature vectors z \in \mathbb{R}^{N \times D}.
       Let \tilde{z} = z - \bar{z} be centered feature tokens. The empirical covariance matrix is:
           \Sigma = \frac{1}{N - 1} \tilde{z}^T \tilde{z}
       Its eigenvalues \lambda_k are proportional to the squared singular values S_k^2 from SVD(\tilde{z}):
           \lambda_k \propto S_k^2
       Normalizing eigenvalues produces a valid discrete probability distribution:
           p_k = \frac{\lambda_k}{\sum_{j=1}^D \lambda_j} = \frac{S_k^2}{\sum_{j=1}^D S_k^2}
       The effective rank is the exponential of the Shannon entropy of this spectral distribution:
           \text{erank}(z) = \exp\left(-\sum_{k=1}^D p_k \ln(p_k)\right)

    2. Why Squared Singular Values (S^2) Are Mathematically Mandatory:
       Using linear singular values S_k artificially flattens the spectral distribution (since \sqrt{x}
       compresses dynamic range), masking dimensional collapse and reporting spuriously high ranks.
       Using S_k^2 strictly reflects the variance explained along each principal axis.
       - Max value: D = 384 (uniform energy across all orthogonal dimensions, perfect isotropy).
       - Min value: 1.0 (all energy concentrated in a single 1D subspace, complete dimensional collapse).

    References:
    -----------
    - Roy, O., & Vetterli, M. (2007). "The effective rank: A measure of effective dimensionality."
      15th European Signal Processing Conference (EUSIPCO 2007), pp. 606-610.
    """
    z_centered = z - z.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
        # Use squared singular values (= eigenvalues of covariance matrix)
        # Linear singular values overestimate rank and mask true collapse.
        eigenvalues = S ** 2
        if eigenvalues.sum() < 1e-12:
            return 1.0  # Degenerate case: all-zero representations
        normalized = eigenvalues / eigenvalues.sum()
        # Shannon entropy of normalized eigenvalues
        entropy = -torch.sum(normalized * torch.log(normalized + 1e-12))
        eff_rank = torch.exp(entropy).item()
        return eff_rank
    except (RuntimeError, ValueError):
        return 1.0

def compute_representation_collapse_metrics(z: torch.Tensor) -> dict[str, float]:
    r"""
    Multi-Faceted Representation Collapse Diagnostic Suite.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Average Pairwise Cosine Similarity:
           \text{CosSim}_{\text{avg}} = \frac{1}{N(N - 1)} \sum_{i \neq j} \frac{z_i^T z_j}{\|z_i\|_2 \|z_j\|_2}
       Evaluates angular diversity across normalized latent tokens. In complete dimensional collapse
       (where all tokens map to the identical direction vector in \mathbb{R}^D), cosine similarity
       approaches 1.0. For uniformly distributed isotropic representations on the unit sphere,
       the expectation is 0.0. A sample size of up to 1,000 tokens is evaluated to bound O(N^2) memory.

    2. Feature Variance:
       Computes the mean coordinate-wise empirical variance \frac{1}{D} \sum_{d=1}^D \text{Var}(z_d).
       Diagnoses point collapse where representations contract to a constant zero or static centroid.
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
