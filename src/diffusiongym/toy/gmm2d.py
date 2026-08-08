"""Two-dimensional four-Gaussian mixture toy problem.

Provides:
  - GMMFlowModel  — pretrained 2-D velocity model (rectified flow)
  - GMMBaseSampler — N(0, I₂) base distribution
  - exact_velocity — analytic marginal flow velocity for verification
  - LinearReward / QuadraticReward / BoxReward — three reward families
  - tilted_target_params — analytic parameters of p*(x) ∝ p_base(x) exp(λ r(x))
  - TiltedDifferentiableCost — differentiable terminal cost for Adjoint Matching

Spec reference: specs_test_example.md
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Generator, Tensor, nn

from diffusiongym.core.model import PredictionKind
from diffusiongym.core.reward import RewardBatch
from diffusiongym.types import DDTensor


# ---------------------------------------------------------------------------
# Mixture parameters
# ---------------------------------------------------------------------------

MODES = torch.tensor([[-2.0, -2.0], [-2.0, 2.0], [2.0, -2.0], [2.0, 2.0]])
SIGMA_DATA = 0.35
NUM_MODES = 4


# ---------------------------------------------------------------------------
# MLP for velocity prediction (2-D → 2-D with sinusoidal time embedding)
# ---------------------------------------------------------------------------

class _SinusoidalEmbed(nn.Module):
    def __init__(self, dim: int = 64, t_mult: float = 1000.0) -> None:
        super().__init__()
        self.t_mult = t_mult
        half = dim // 2
        freqs = torch.exp(
            -math.log(1000.0) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, 128)
        )

    def forward(self, t: Tensor) -> Tensor:
        args = (t.float() * self.t_mult).unsqueeze(-1) * self.freqs  # type: ignore[operator]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class _FiLMBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.scale_shift = nn.Linear(cond_dim, 2 * width)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.norm(self.linear(x))
        gamma, beta = self.scale_shift(cond).chunk(2, dim=-1)
        return F.silu(h * (1 + gamma) + beta)


class VelocityMLP(nn.Module):
    """2-D velocity MLP: (x₁, x₂, t) → (v₁, v₂)."""

    def __init__(self, width: int = 128, depth: int = 3) -> None:
        super().__init__()
        self.embed = _SinusoidalEmbed(dim=64)
        self.input_proj = nn.Linear(2, width)
        self.blocks = nn.ModuleList([_FiLMBlock(width, 128) for _ in range(depth)])
        self.head = nn.Linear(width, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        cond = self.embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.head(h)


# ---------------------------------------------------------------------------
# FlowModel wrapper
# ---------------------------------------------------------------------------

class GMMFlowModel:
    """Wraps a VelocityMLP into the FlowModel protocol for DDTensor states."""

    prediction_kind = PredictionKind.VELOCITY

    def __init__(self, mlp: VelocityMLP, device: torch.device) -> None:
        self._mlp = mlp
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def parameters(self):
        return self._mlp.parameters()

    def __call__(self, x_t: DDTensor, t: Tensor, *, conditioning: dict) -> DDTensor:
        return DDTensor(self._mlp(x_t.data, t))

    def state_dict(self):
        return self._mlp.state_dict()

    def load_state_dict(self, sd):
        return self._mlp.load_state_dict(sd)

    def clone(self) -> "GMMFlowModel":
        new_mlp = copy.deepcopy(self._mlp)
        return GMMFlowModel(new_mlp, self._device)

    def train(self) -> None:
        self._mlp.train()

    def eval(self) -> None:
        self._mlp.eval()


# ---------------------------------------------------------------------------
# Analytic velocity
# ---------------------------------------------------------------------------

def exact_velocity(
    x: Tensor,
    t: Tensor,
    *,
    weights: Tensor | None = None,
    means: Tensor | None = None,
    sigma: float = SIGMA_DATA,
) -> Tensor:
    """Analytic marginal flow-matching velocity for a Gaussian mixture.

    For rectified flow  x_t = (1-t) z + t x₁,  z ~ N(0, I),
    the marginal at time t is a Gaussian mixture with component means t·μₖ
    and covariance C_{k,t} = (1-t)² I + t² σ² I.

    The conditional endpoint expectation given xₜ and component k is:
        m_k(x,t) = μ_k + t σ² C_{k,t}⁻¹ (x - t μ_k)

    The velocity is u*(x,t) = Σ_k γ_k(x,t) · (m_k(x,t) - x) / (1-t).

    Parameters
    ----------
    x : (n, 2)
    t : (n,)
    weights : (K,)  mixture weights, uniform if None
    means :   (K, 2) component means, MODES if None
    sigma :   data standard deviation
    """
    device = x.device
    if means is None:
        means = MODES.to(device)
    if weights is None:
        weights = torch.full((NUM_MODES,), 1.0 / NUM_MODES, device=device)

    K = means.shape[0]
    n = x.shape[0]
    t_ = t.clamp(1e-4, 1.0 - 1e-4)

    # Component covariance scalars: c_t = (1-t)² + t² σ²
    c_t = (1 - t_) ** 2 + t_ ** 2 * sigma ** 2  # (n,)

    # Component means at time t: μ_{k,t} = t · μ_k  → (n, K, 2)
    mu_t = t_.unsqueeze(1).unsqueeze(2) * means.unsqueeze(0)  # (n, K, 2)

    # Mahalanobis distance squared for each component: ||x - μ_{k,t}||² / c_t
    diff = x.unsqueeze(1) - mu_t  # (n, K, 2)
    sq = diff.square().sum(-1) / c_t.unsqueeze(1)  # (n, K)

    # Log responsibilities: log w_k - ½ (sq + 2 log c_t + 2 log 2π)
    log_w = weights.log().unsqueeze(0)  # (1, K)
    log_norm = -(sq + 2 * c_t.log().unsqueeze(1) + 2 * math.log(2 * math.pi)) / 2
    log_gamma = log_w + log_norm  # (n, K)
    gamma = torch.softmax(log_gamma, dim=1)  # (n, K)

    # Conditional endpoint expectation for each component
    # m_k = μ_k + t σ² / c_t · (x - t μ_k)
    t_sigma_sq_over_c = (t_ * sigma ** 2 / c_t).unsqueeze(1).unsqueeze(2)  # (n, 1, 1)
    m_k = means.unsqueeze(0) + t_sigma_sq_over_c * diff  # (n, K, 2)

    # Weighted endpoint
    m_bar = (gamma.unsqueeze(2) * m_k).sum(1)  # (n, 2)

    # Velocity: (m_bar - x) / (1-t)
    vel = (m_bar - x) / (1 - t_).unsqueeze(1)
    return vel


# ---------------------------------------------------------------------------
# Base sampler
# ---------------------------------------------------------------------------

class GMMBaseSampler:
    """N(0, I₂) base distribution."""

    def sample(
        self,
        n: int,
        *,
        conditioning: Mapping[str, object],
        device: torch.device,
        generator: Generator | None = None,
    ) -> tuple[DDTensor, Mapping[str, object]]:
        return DDTensor(torch.randn(n, 2, device=device, generator=generator)), conditioning

    def sample_like(self, x_data: DDTensor, *, generator: Generator | None = None) -> DDTensor:
        return DDTensor(torch.randn_like(x_data.data, generator=generator))


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class LinearReward:
    """r(x) = c·x,  c = (1, 0) — rewards the rightward component."""

    def __init__(self, direction: tuple[float, float] = (1.0, 0.0)) -> None:
        self.c = torch.tensor(direction)

    def __call__(
        self, *, sample: DDTensor, latent: DDTensor, conditioning: dict
    ) -> RewardBatch:
        c = self.c.to(sample.device)
        rewards = (sample.data * c).sum(-1)
        return RewardBatch(rewards=rewards)

    def differentiable(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        c = self.c.to(terminal_latent.device)
        return (terminal_latent.data * c).sum(-1)


class QuadraticReward:
    """r(x) = -1/(2τ²) ||x - y||²  — pulls toward target point y."""

    def __init__(
        self,
        target: tuple[float, float] = (3.0, 1.0),
        tau: float = 1.0,
    ) -> None:
        self.y = torch.tensor(target)
        self.tau = tau

    def __call__(
        self, *, sample: DDTensor, latent: DDTensor, conditioning: dict
    ) -> RewardBatch:
        y = self.y.to(sample.device)
        rewards = -0.5 / self.tau ** 2 * (sample.data - y).square().sum(-1)
        return RewardBatch(rewards=rewards)

    def differentiable(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        y = self.y.to(terminal_latent.device)
        return -0.5 / self.tau ** 2 * (terminal_latent.data - y).square().sum(-1)


class BoxReward:
    """r(x) = 1[x ∈ A]  — non-differentiable indicator reward."""

    def __init__(
        self,
        x_range: tuple[float, float] = (2.2, 3.2),
        y_range: tuple[float, float] = (0.2, 1.2),
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range

    def __call__(
        self, *, sample: DDTensor, latent: DDTensor, conditioning: dict
    ) -> RewardBatch:
        x = sample.data
        in_box = (
            (x[:, 0] >= self.x_range[0]) & (x[:, 0] <= self.x_range[1]) &
            (x[:, 1] >= self.y_range[0]) & (x[:, 1] <= self.y_range[1])
        )
        return RewardBatch(rewards=in_box.float())


# ---------------------------------------------------------------------------
# Analytic tilted target parameters
# ---------------------------------------------------------------------------

@dataclass
class TiltedMixture:
    """Parameters of p*(x) ∝ p_base(x) exp(λ r(x)) for Gaussian mixture.

    weights : (K,) normalized mixture weights
    means   : (K, 2) component means
    covs    : (K, 2, 2) component covariance matrices
    """

    weights: Tensor
    means: Tensor
    covs: Tensor


def tilted_target_linear(
    lam: float,
    direction: tuple[float, float] = (1.0, 0.0),
    sigma: float = SIGMA_DATA,
    modes: Tensor | None = None,
) -> TiltedMixture:
    """Analytic tilted target for linear reward r(x) = c·x.

    Under tilt:  w*_k ∝ exp(λ c·μ_k),  μ*_k = μ_k + λ σ² c,  Σ*_k = σ² I.
    """
    if modes is None:
        modes = MODES
    c = torch.tensor(direction)
    log_weights = lam * (modes @ c)
    weights = torch.softmax(log_weights, dim=0)
    means = modes + lam * sigma ** 2 * c.unsqueeze(0)
    cov = sigma ** 2 * torch.eye(2).unsqueeze(0).expand(NUM_MODES, -1, -1)
    return TiltedMixture(weights=weights, means=means, covs=cov)


def tilted_target_quadratic(
    lam: float,
    target: tuple[float, float] = (3.0, 1.0),
    tau: float = 1.0,
    sigma: float = SIGMA_DATA,
    modes: Tensor | None = None,
) -> TiltedMixture:
    """Analytic tilted target for quadratic reward r(x) = -||x-y||²/(2τ²).

    Gaussian-Gaussian conjugacy: posterior mean and covariance per component.
    """
    if modes is None:
        modes = MODES
    y = torch.tensor(target)
    # Prior covariance Σ = σ² I, reward precision R⁻¹ = λ/τ² I
    sigma_sq = sigma ** 2
    r_var = tau ** 2 / lam  # variance from reward factor

    # Posterior covariance: (1/σ² + λ/τ²)⁻¹ I
    post_var = 1.0 / (1.0 / sigma_sq + 1.0 / r_var)
    post_cov = post_var * torch.eye(2)

    # Posterior mean for each component
    post_means = post_var * (modes / sigma_sq + y.unsqueeze(0) / r_var)

    # Unnormalized log weights: log N(y; μ_k, (σ²+r_var) I)
    total_var = sigma_sq + r_var
    diff = modes - y.unsqueeze(0)
    log_w = -0.5 * diff.square().sum(-1) / total_var - math.log(2 * math.pi * total_var)
    weights = torch.softmax(log_w, dim=0)

    covs = post_cov.unsqueeze(0).expand(NUM_MODES, -1, -1)
    return TiltedMixture(weights=weights, means=post_means, covs=covs)


def tilted_target_box(
    lam: float,
    x_range: tuple[float, float] = (2.2, 3.2),
    y_range: tuple[float, float] = (0.2, 1.2),
    sigma: float = SIGMA_DATA,
    modes: Tensor | None = None,
) -> dict:
    """Return normalizer and P*(A) for the box reward analytic target.

    Returns a dict with keys: 'p_base_A' (prob of box under base), 'p_star_A'.
    The tilted distribution itself is not a simple mixture; use this for metric
    computation rather than sampling.
    """
    from scipy.stats import norm  # type: ignore[import-untyped]

    if modes is None:
        modes = MODES

    # P_base(A) = ¼ Σ_k Φ(x2_hi;μ_k1,σ)·... using CDF products
    p_base_a = 0.0
    for k in range(NUM_MODES):
        mu = modes[k].tolist()
        px = norm.cdf(x_range[1], mu[0], sigma) - norm.cdf(x_range[0], mu[0], sigma)
        py = norm.cdf(y_range[1], mu[1], sigma) - norm.cdf(y_range[0], mu[1], sigma)
        p_base_a += px * py / NUM_MODES

    exp_lam = math.exp(lam)
    z = 1.0 + (exp_lam - 1.0) * p_base_a
    p_star_a = exp_lam * p_base_a / z
    return {"p_base_A": p_base_a, "p_star_A": p_star_a, "Z": z}


# ---------------------------------------------------------------------------
# Differentiable terminal costs (for Adjoint Matching)
# ---------------------------------------------------------------------------

class LinearDifferentiableCost:
    """Differentiable cost -(c·x) corresponding to linear reward."""

    def __init__(self, direction: tuple[float, float] = (1.0, 0.0)) -> None:
        self.c = torch.tensor(direction)

    def __call__(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        c = self.c.to(terminal_latent.device)
        return -(terminal_latent.data * c).sum(-1)


class QuadraticDifferentiableCost:
    """Differentiable cost 1/(2τ²) ||x - y||² (positive = cost to minimize)."""

    def __init__(
        self,
        target: tuple[float, float] = (3.0, 1.0),
        tau: float = 1.0,
    ) -> None:
        self.y = torch.tensor(target)
        self.tau = tau

    def __call__(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        y = self.y.to(terminal_latent.device)
        return 0.5 / self.tau ** 2 * (terminal_latent.data - y).square().sum(-1)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def sample_gmm(n: int, device: torch.device, sigma: float = SIGMA_DATA) -> Tensor:
    """Sample n points from the 4-Gaussian mixture."""
    k = torch.randint(0, NUM_MODES, (n,))
    means = MODES[k].to(device)
    noise = torch.randn(n, 2, device=device) * sigma
    return means + noise


def pretrain_velocity_model(
    model: GMMFlowModel,
    *,
    steps: int = 20_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> list[float]:
    """Train a velocity MLP on the 2-D rectified-flow objective.

    Loss: E[||u_θ(x_t, t) - (x₁ - z)||²]  with  x_t = (1-t)z + t x₁.
    """
    if device is None:
        device = model.device
    model._mlp.to(device)
    model._mlp.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    log_every = max(1, steps // 20)
    for step in range(steps):
        x1 = sample_gmm(batch_size, device)
        z = torch.randn(batch_size, 2, device=device)
        t = torch.rand(batch_size, device=device)

        x_t = (1 - t).unsqueeze(1) * z + t.unsqueeze(1) * x1
        target = x1 - z

        pred = model._mlp(x_t, t)
        loss = (pred - target).square().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if verbose and (step + 1) % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"  pretrain step {step+1:5d}/{steps}  loss={avg:.4f}")

    model._mlp.eval()
    return losses


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def sample_model(
    model: GMMFlowModel,
    n: int = 2000,
    steps: int = 50,
    device: torch.device | None = None,
) -> Tensor:
    """Generate n samples from the flow model using the Euler ODE sampler."""
    if device is None:
        device = model.device
    model._mlp.eval()

    with torch.no_grad():
        x = torch.randn(n, 2, device=device)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = i / steps
            t = torch.full((n,), t_val, device=device)
            v = model._mlp(x, t)
            x = x + v * dt
    return x


def compute_metrics(
    samples: Tensor,
    target: TiltedMixture | None = None,
    *,
    reward_fn: LinearReward | QuadraticReward | BoxReward | None = None,
    modes: Tensor | None = None,
    sigma: float = SIGMA_DATA,
) -> dict[str, float]:
    """Compute moment, mode-mass, and reward metrics.

    Returns a dict with keys: mean_error, mode_weight_error, expected_reward.
    """
    if modes is None:
        modes = MODES.to(samples.device)

    metrics: dict[str, float] = {}

    # Expected reward
    if reward_fn is not None:
        fake_ddt = DDTensor(samples)
        rb = reward_fn(sample=fake_ddt, latent=fake_ddt, conditioning={})
        metrics["expected_reward"] = rb.rewards.mean().item()

    if target is not None:
        target_means = target.means.to(samples.device)
        target_weights = target.weights.to(samples.device)

        # Empirical mean vs. analytic mean
        empirical_mean = samples.mean(0)
        analytic_mean = (target_weights.unsqueeze(1) * target_means).sum(0)
        metrics["mean_error"] = (empirical_mean - analytic_mean).norm().item()

        # Mode assignment: closest original mode
        dists = (samples.unsqueeze(1) - modes.to(samples.device).unsqueeze(0)).norm(dim=-1)
        assignments = dists.argmin(dim=1)
        K = modes.shape[0]
        emp_weights = torch.zeros(K, device=samples.device)
        for k in range(K):
            emp_weights[k] = (assignments == k).float().mean()
        metrics["mode_weight_error"] = (emp_weights - target_weights).abs().mean().item()

    return metrics


# ---------------------------------------------------------------------------
# Exact densities (2-D only)
#
# In two dimensions the instantaneous change of variables is cheap: the
# divergence of the velocity field is an exact 2x2 trace, so log p_theta(x) can
# be computed exactly rather than estimated. That is what makes an E[r]-vs-KL
# frontier and a "where did the mass move" map possible on this problem.
# ---------------------------------------------------------------------------


def _velocity_and_divergence(
    mlp: nn.Module, x: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    """Velocity and its exact divergence at (x, t), shapes (n, 2) and (n,)."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        v = mlp(x, t)
        d_vx = torch.autograd.grad(v[:, 0].sum(), x, retain_graph=True)[0][:, 0]
        d_vy = torch.autograd.grad(v[:, 1].sum(), x)[0][:, 1]
    return v.detach(), (d_vx + d_vy).detach()


def _standard_normal_logpdf(z: Tensor) -> Tensor:
    return -0.5 * (z.square().sum(-1) + 2.0 * math.log(2.0 * math.pi))


def log_density(
    model: GMMFlowModel,
    x: Tensor,
    *,
    steps: int = 200,
) -> Tensor:
    """Exact log p_theta(x) for the flow's terminal distribution, shape (n,).

    Integrates the probability-flow ODE backward from t=1 to t=0 while
    accumulating the divergence:

        log p_1(x_1) = log N(x_0; 0, I) - integral_0^1 div v(x_t, t) dt.

    Both models in a KL are integrated with the same grid so the discretization
    error largely cancels in the difference. Validated by
    `check_density_normalization`.
    """
    mlp = model._mlp
    z = x.detach().clone()
    divergence_integral = torch.zeros(len(x), device=x.device)
    dt = 1.0 / steps

    for i in reversed(range(steps)):
        t = torch.full((len(x),), i * dt, device=x.device)
        v, div = _velocity_and_divergence(mlp, z, t)
        z = z - v * dt
        divergence_integral = divergence_integral + div * dt

    return _standard_normal_logpdf(z) - divergence_integral


def check_density_normalization(
    model: GMMFlowModel,
    *,
    limit: float = 6.0,
    resolution: int = 120,
    steps: int = 100,
) -> float:
    """Numerically integrate exp(log_density) over a box; should return ~1.0.

    A cheap end-to-end check that the change-of-variables bookkeeping (signs,
    step direction, divergence) is right, since every downstream KL depends on it.
    """
    axis = torch.linspace(-limit, limit, resolution)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).to(model.device)
    log_p = log_density(model, points, steps=steps)
    cell_area = (2.0 * limit / (resolution - 1)) ** 2
    return float(log_p.exp().sum().item() * cell_area)


def gmm_log_density(
    x: Tensor,
    *,
    weights: Tensor | None = None,
    means: Tensor | None = None,
    sigma: float = SIGMA_DATA,
) -> Tensor:
    """Analytic log density of the Gaussian mixture, shape (n,)."""
    device = x.device
    if means is None:
        means = MODES.to(device)
    if weights is None:
        weights = torch.full((means.shape[0],), 1.0 / means.shape[0], device=device)
    diff = x.unsqueeze(1) - means.unsqueeze(0)
    log_components = (
        -0.5 * diff.square().sum(-1) / sigma**2
        - math.log(2.0 * math.pi * sigma**2)
        + weights.log().unsqueeze(0)
    )
    return torch.logsumexp(log_components, dim=1)


# ---------------------------------------------------------------------------
# Reward / KL diagnostics
#
# The quantity to compare fine-tuning methods on is not E[r] alone — a policy
# that collapses onto the reward maximum wins that outright — but the pair
# (E[r], KL(p_theta || p_ref)) against the best achievable trade-off.
# ---------------------------------------------------------------------------


@dataclass
class DensityGrid:
    """A fixed evaluation grid carrying the reference log-density."""

    points: Tensor  # (G, 2)
    log_p_ref: Tensor  # (G,)
    cell_area: float
    resolution: int
    limit: float

    def as_image(self, values: Tensor):
        return values.reshape(self.resolution, self.resolution)


def make_density_grid(
    ref_model: GMMFlowModel | None = None,
    *,
    limit: float = 6.0,
    resolution: int = 120,
    steps: int = 100,
    device: torch.device | None = None,
) -> DensityGrid:
    """Build the evaluation grid and evaluate the reference density on it.

    Pass the pretrained flow to measure against the exact distribution the
    algorithms actually regularize toward; omit it to use the analytic mixture
    (faster, and within ~0.07 nats on this problem).
    """
    if device is None:
        device = ref_model.device if ref_model is not None else torch.device("cpu")
    axis = torch.linspace(-limit, limit, resolution, device=device)
    xx, yy = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    if ref_model is None:
        log_p_ref = gmm_log_density(points)
    else:
        log_p_ref = log_density(ref_model, points, steps=steps)

    # Renormalize on the grid so that the truncation and integrator error cancel
    # exactly: without this, KL(p*||p_ref) at lambda = 0 comes out at -log Z
    # (measured -0.028 here) instead of 0, biasing every frontier point.
    cell_area = (2.0 * limit / (resolution - 1)) ** 2
    log_p_ref = log_p_ref - torch.logsumexp(
        log_p_ref + math.log(cell_area), dim=0
    )
    return DensityGrid(
        points=points,
        log_p_ref=log_p_ref,
        cell_area=cell_area,
        resolution=resolution,
        limit=limit,
    )


def grid_rewards(grid: DensityGrid, reward_fn) -> Tensor:
    """Evaluate a reward on every grid point, shape (G,)."""
    batch = DDTensor(grid.points)
    return reward_fn(sample=batch, latent=batch, conditioning={}).rewards


def tilted_log_density(
    grid: DensityGrid, rewards: Tensor, lam: float
) -> Tensor:
    """Normalized log p*(x) = log p_ref(x) + lam r(x) - log Z on the grid."""
    unnormalized = grid.log_p_ref + lam * rewards
    log_z = torch.logsumexp(unnormalized + math.log(grid.cell_area), dim=0)
    return unnormalized - log_z


def analytic_frontier(
    grid: DensityGrid,
    reward_fn,
    lambdas,
) -> list[dict[str, float]]:
    """The best achievable (E[r], KL) trade-off, one point per lambda.

    p*(x) ∝ p_ref(x) exp(lam r(x)) is the exact maximizer of
    E_p[r] - (1/lam) KL(p || p_ref), so this curve upper-bounds every method:
    any run that sits below and to the right of it is spending KL it did not
    need to spend.
    """
    rewards = grid_rewards(grid, reward_fn)
    out = []
    for lam in lambdas:
        log_p_star = tilted_log_density(grid, rewards, float(lam))
        mass = log_p_star.exp() * grid.cell_area
        out.append(
            {
                "lam": float(lam),
                "expected_reward": float((mass * rewards).sum().item()),
                "kl": float((mass * (log_p_star - grid.log_p_ref)).sum().item()),
            }
        )
    return out


@dataclass
class TiltDiagnostics:
    """Measured behaviour of one fine-tuned model."""

    expected_reward: float
    kl_to_reference: float
    achieved_lambda: float
    tilt_r2: float
    samples: Tensor
    log_ratio: Tensor
    rewards: Tensor


def tilt_diagnostics(
    model: GMMFlowModel,
    ref_model: GMMFlowModel,
    reward_fn,
    *,
    n: int = 2000,
    sample_steps: int = 100,
    density_steps: int = 200,
) -> TiltDiagnostics:
    """Measure E[r], KL(p_theta || p_ref), and how reward-shaped the tilt is.

    For any KL-regularized optimum the achieved log-ratio is exactly linear in
    the reward,

        log p_theta(x) - log p_ref(x) = lam r(x) - log Z,

    so regressing the measured log-ratio on the reward recovers the tilt the
    method actually applied (`achieved_lambda`) and how much of the density
    change the reward explains at all (`tilt_r2`). An r^2 near 1 is the precise
    statement of "fine-tuning only moved probability mass where the reward asked
    for it"; a low r^2 means mass moved for reasons unrelated to the reward,
    however good E[r] looks.

    Both densities use the same integrator, so the O(1/steps) discretization
    bias largely cancels in the difference.
    """
    samples = sample_model(model, n=n, steps=sample_steps, device=model.device)
    log_p = log_density(model, samples, steps=density_steps)
    log_p_ref = log_density(ref_model, samples, steps=density_steps)
    log_ratio = log_p - log_p_ref

    batch = DDTensor(samples)
    rewards = reward_fn(sample=batch, latent=batch, conditioning={}).rewards

    # Ordinary least squares of log_ratio on reward.
    r_centered = rewards - rewards.mean()
    l_centered = log_ratio - log_ratio.mean()
    denominator = r_centered.square().sum().clamp_min(1e-12)
    slope = (r_centered * l_centered).sum() / denominator
    residual = l_centered - slope * r_centered
    total = l_centered.square().sum()
    # If the density never moved there is no variance to explain and r^2 would
    # come out at a misleading 1.0 ("perfectly localized") for a no-op update.
    r2 = (
        1.0 - residual.square().sum() / total
        if float(total.item()) > 1e-8
        else torch.tensor(float("nan"))
    )

    return TiltDiagnostics(
        expected_reward=float(rewards.mean().item()),
        kl_to_reference=float(log_ratio.mean().item()),
        achieved_lambda=float(slope.item()),
        tilt_r2=float(r2.item()),
        samples=samples,
        log_ratio=log_ratio,
        rewards=rewards,
    )


# ---------------------------------------------------------------------------
# Harder reward families
#
# The linear reward is easy: it is monotone and aligned with the mode structure,
# so any method that pushes in roughly the right direction passes. These two ask
# for something a direction cannot express.
# ---------------------------------------------------------------------------


class BimodalReward:
    """Two Gaussian bumps of unequal height on opposite modes.

    Non-monotone, so no single direction in x increases it. The target puts a
    specific *ratio* of mass on two opposite modes and drains the other two, so
    a method is only correct if it reproduces that ratio — "move right" scores
    nothing here.
    """

    def __init__(
        self,
        primary: tuple[float, float] = (2.0, 2.0),
        secondary: tuple[float, float] = (-2.0, -2.0),
        secondary_height: float = 0.6,
        width: float = 0.7,
    ) -> None:
        self.primary = torch.tensor(primary)
        self.secondary = torch.tensor(secondary)
        self.secondary_height = secondary_height
        self.width = width

    def _value(self, x: Tensor) -> Tensor:
        p = self.primary.to(x.device)
        s = self.secondary.to(x.device)
        w2 = 2.0 * self.width**2
        return torch.exp(-(x - p).square().sum(-1) / w2) + self.secondary_height * (
            torch.exp(-(x - s).square().sum(-1) / w2)
        )

    def __call__(
        self, *, sample: DDTensor, latent: DDTensor, conditioning: dict
    ) -> RewardBatch:
        return RewardBatch(rewards=self._value(sample.data))

    def differentiable(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        return self._value(terminal_latent.data)


class BimodalDifferentiableCost:
    """Terminal cost g(x) = -r(x) for BimodalReward."""

    def __init__(self, **kwargs) -> None:
        self.reward = BimodalReward(**kwargs)

    def __call__(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        return -self.reward._value(terminal_latent.data)


class RingReward:
    """Prefers a radius, r(x) = -(|x| - radius)^2 / (2 width^2).

    The modes sit at radius 2*sqrt(2) ~ 2.83, so a smaller target radius asks
    every mode to move *inward* by a fraction of its own width while keeping all
    four. This is within-mode geometry rather than mode reweighting, which is a
    different capability: reweighting the samples it already has cannot do it.
    """

    def __init__(self, radius: float = 2.0, width: float = 1.0) -> None:
        self.radius = radius
        self.width = width

    def _value(self, x: Tensor) -> Tensor:
        return -(x.norm(dim=-1) - self.radius).square() / (2.0 * self.width**2)

    def __call__(
        self, *, sample: DDTensor, latent: DDTensor, conditioning: dict
    ) -> RewardBatch:
        return RewardBatch(rewards=self._value(sample.data))

    def differentiable(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        return self._value(terminal_latent.data)


class RingDifferentiableCost:
    """Terminal cost g(x) = -r(x) for RingReward."""

    def __init__(self, **kwargs) -> None:
        self.reward = RingReward(**kwargs)

    def __call__(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        return -self.reward._value(terminal_latent.data)
