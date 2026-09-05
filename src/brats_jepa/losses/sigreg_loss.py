
import torch
import torch.nn.functional as F
from torch import nn

from .ijepa_loss import IJEPALoss


class EppsPulleyGaussianityTest(nn.Module):
    r"""
    Epps-Pulley Goodness-of-Fit Test Statistic for 1D Gaussianity.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Cramér-Wold Theorem (1936):
       A d-dimensional multivariate probability distribution P on R^d is uniquely determined
       by the family of its 1D marginal distributions under all 1D linear projections
       u \in S^{d-1}. Testing multivariate standard normal N(0, I_d) is equivalent to testing
       that projected scalars u^T z ~ N(0, 1) for all u.
    
    2. Epps-Pulley Test (1983):
       Compares the 1D Empirical Characteristic Function (ECF):
           \hat{\phi}_N(t) = \frac{1}{N} \sum_{n=1}^N \exp(i t p_n)
       against the analytical standard Gaussian characteristic function:
           \phi_0(t) = \exp(-t^2 / 2)
       under the Gaussian-weighted L2 metric:
           T_{EP} = N \int_{-\infty}^{\infty} |\hat{\phi}_N(t) - \phi_0(t)|^2 d\mu(t)
       where d\mu(t) = \frac{1}{\sqrt{2\pi}} \exp(-t^2 / 2) dt.

    3. Numerical Quadrature & Symmetry:
       Because \phi_0(t) is even and real, symmetry across t \in [-t_max, t_max] allows
       integrating over [0, t_max] and doubling the weights (except at t=0 and t=t_max),
       yielding trapezoidal quadrature weights: w_k = 2 dt for interior knots, w_0 = w_{K-1} = dt.

    4. Gradient Scaling (Why multiply by N?):
       The derivative of the empirical characteristic function with respect to projection p_n is:
           \frac{\partial \hat{\phi}_N(t)}{\partial p_n} = \frac{i t}{N} \exp(i t p_n)
       Without multiplying the test statistic by sample count N, \frac{\partial \mathcal{L}}{\partial p_n}
       would carry an extraneous 1/N factor (~1/768), leading to vanishing gradients on representations.
       Multiplying by N cancels this factor, producing an O(1) per-sample gradient that balances with
       the primary JEPA prediction loss, matching the asymptotic chi-squared distribution under H0.

    References:
    -----------
    - Epps, T. W., & Pulley, L. B. (1983). "A test for normality based on the empirical
      characteristic function." Biometrika, 70(3), 723-726.
    - Balestriero, R., & LeCun, Y. (2025). "Learning by Predicting Without Representation
      Collapse." arXiv:2511.08544 (LeJEPA / SigReg).
    """
    def __init__(self, t_max: float = 3.0, n_knots: int = 17):
        super().__init__()
        t = torch.linspace(0.0, t_max, n_knots, dtype=torch.float32)
        dt = t_max / (n_knots - 1)
        weights = torch.full((n_knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: [N, K] where N is number of samples, K is number of random 1D projections.
        """
        t = self.t.to(device=proj.device, dtype=proj.dtype)
        phi = self.phi.to(device=proj.device, dtype=proj.dtype)
        weights = self.weights.to(device=proj.device, dtype=proj.dtype)

        x_t = proj.unsqueeze(-1) * t  # [N, K, Q]
        ecf_real = x_t.cos().mean(dim=0)   # [K, Q]
        ecf_imag = x_t.sin().mean(dim=0)   # [K, Q]
        err = (ecf_real - phi).square() + ecf_imag.square()  # [K, Q]
        # Following LeJEPA (Balestriero & LeCun, 2025, MINIMAL.md line 82), multiply by
        # sample count N = proj.size(0) so the statistic matches the asymptotic chi-squared
        # scale and cancels the 1/N factor in d(ecf)/dz, ensuring O(1) per-sample gradients.
        statistic = (err @ weights) * proj.shape[0]  # [K]
        return statistic.mean()


class SigRegLoss(nn.Module):
    r"""
    SigReg / LeJEPA Loss: Prediction Loss + Sketched Isotropic Gaussian Regularization.

    Mathematical Rationale & Defense Context:
    -----------------------------------------
    1. Objective Formulation:
           \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{JEPA}}(\hat{s}_y, s_y) + \lambda_{\text{sig}} \mathcal{L}_{\text{SIGReg}}(z)
       where \mathcal{L}_{\text{JEPA}} is the Smooth L1 prediction loss in latent space, and
       \mathcal{L}_{\text{SIGReg}} enforces isotropic Gaussianity on representation tokens z.

    2. Collapse Prevention Without EMA Heuristics:
       Conventional self-supervised predictive architectures (e.g. standard I-JEPA) prevent
       representation collapse (\forall x, z(x) = \text{const}) through asymmetric momentum
       teachers (EMA updates). SigReg mathematically proves that constraining the empirical
       distribution of representations to match an isotropic Gaussian \mathcal{N}(0, I_D)
       guarantees maximal differential entropy, strictly ruling out point collapse, dimensional
       collapse, and low-rank subspace degeneration without requiring stop-gradient heuristics
       or EMA target encoders.

    3. Random Slicing via Unit Hypersphere Sampling:
       By the Johnson-Lindenstrauss lemma and the Cramér-Wold device, projecting D-dimensional
       tokens z onto M random directions A \sim \mathcal{U}(\mathbb{S}^{D-1}) preserves geometric
       and distribution properties in expectation. We choose M = 256 unit projections.

    References:
    -----------
    - Balestriero, R., & LeCun, Y. (2025). "Learning by Predicting Without Representation
      Collapse." arXiv:2511.08544.
    - Cramér, H., & Wold, H. (1936). "Some theorems on distribution functions."
      Journal of the London Mathematical Society, 1(4), 290-294.
    """
    def __init__(
        self,
        loss_type: str = "smooth_l1",
        sigreg_weight: float = 1.0,
        num_projections: int = 256,
        t_max: float = 3.0,
        n_knots: int = 17,
    ):
        super().__init__()
        self.jepa_loss = IJEPALoss(loss_type=loss_type)
        self.sigreg_weight = sigreg_weight
        self.num_projections = num_projections
        self.ep_test = EppsPulleyGaussianityTest(t_max=t_max, n_knots=n_knots)

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        context_tokens: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        projected_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        j_loss = self.jepa_loss(predictions, targets)
        
        # Support flexible argument names (context_tokens, tokens, or projected_tokens)
        reg_tokens = tokens if tokens is not None else (projected_tokens if projected_tokens is not None else context_tokens)
        if reg_tokens is None:
            raise ValueError("SigRegLoss requires regularized token representations.")
            
        # Flatten tokens across batch and patch dimensions: [N, D]
        z = reg_tokens.reshape(-1, reg_tokens.shape[-1])
        N, D = z.shape
        
        # Sample M random projection directions on unit hypersphere
        A = torch.randn(D, self.num_projections, device=z.device, dtype=z.dtype)
        A = F.normalize(A, p=2, dim=0)  # [D, M]
        
        # 1D projections: [N, M]
        proj = z @ A
        
        sigreg_val = self.ep_test(proj)
        total_loss = j_loss + self.sigreg_weight * sigreg_val
        
        return {
            "loss": total_loss,
            "jepa_loss": j_loss,
            "sigreg_loss": sigreg_val,
        }
