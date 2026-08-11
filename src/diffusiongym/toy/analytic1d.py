"""A 1-D flow whose model, marginals, and exponential tilts are all closed-form.

`gmm2d.py` is the reference *example* modality: a trained MLP, exact densities
recovered by integrating the instantaneous change of variables. That is the
right tool for judging a fine-tuning algorithm, but the wrong one for judging a
*sampler*, because a sampler's error is then entangled with the model's.

This module removes the model from the picture. The data law is a 1-D Gaussian
mixture with a shared scale, the base is N(0, 1), and the interpolation is
rectified flow, so:

  * the time-t marginal is again a mixture, in closed form (`Mixture1D.marginal`);
  * the exact path velocity ``E[x1 - x0 | x_t]`` is a responsibility-weighted
    combination of per-component linear fields (`ExactVelocityModel`), so the
    "model" is analytic and `AffineFlowMarginalPreservingSDE` on top of it has
    the mixture as its exact terminal law;
  * an exponential tilt ``exp(alpha * x)`` maps the mixture to another mixture
    in closed form (`Mixture1D.tilt`).

That last point is what makes this a ground truth for `core/smc.py`: the
distribution `SMCSampler` is *supposed* to produce, given `log_potential =
alpha * x`, is a mixture this module can write down exactly — means, variance,
and per-mode masses included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Generator, Tensor

from diffusiongym.core.codec import IdentityCodec
from diffusiongym.core.environment import FlowEnvironment
from diffusiongym.core.model import (
    PredictionConverter,
    PredictionKind,
    VelocityRegression,
)
from diffusiongym.core.process import AffineGaussianForwardProcess
from diffusiongym.core.reward import RewardBatch
from diffusiongym.core.schedule import RectifiedFlowSchedule
from diffusiongym.core.space import TensorGeometry
from diffusiongym.types import DDTensor

_LOG_2PI = math.log(2.0 * math.pi)


def _normal_logpdf(x: Tensor, mean: Tensor, var: Tensor) -> Tensor:
    return -0.5 * (_LOG_2PI + torch.log(var) + (x - mean) ** 2 / var)


@dataclass(frozen=True)
class Mixture1D:
    """A 1-D Gaussian mixture with one shared scale.

    The shared scale is what keeps every operation below closed-form and cheap;
    it is not a limitation that matters for testing a sampler.
    """

    weights: Tensor  # (K,), positive, sums to 1
    means: Tensor  # (K,)
    sigma: float

    def __post_init__(self) -> None:
        if self.weights.shape != self.means.shape:
            raise ValueError("weights and means must have the same shape.")
        if float(self.weights.min()) <= 0.0:
            raise ValueError("mixture weights must be strictly positive.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be strictly positive.")
        object.__setattr__(self, "weights", self.weights / self.weights.sum())

    @property
    def variance(self) -> float:
        return self.sigma**2

    def log_density(self, x: Tensor) -> Tensor:
        """log p(x) for x of shape (n,), returned with shape (n,)."""
        var = torch.full_like(self.means, self.variance)
        comp = _normal_logpdf(x.unsqueeze(-1), self.means, var)  # (n, K)
        return torch.logsumexp(comp + torch.log(self.weights), dim=-1)

    def mean(self) -> float:
        return float((self.weights * self.means).sum())

    def var(self) -> float:
        """Total variance: within-component plus between-component."""
        m = self.mean()
        between = float((self.weights * (self.means - m) ** 2).sum())
        return self.variance + between

    def sample(self, n: int, *, generator: Generator | None = None) -> Tensor:
        """Draw n scalars, shape (n,)."""
        k = torch.multinomial(
            self.weights, n, replacement=True, generator=generator
        )
        z = torch.randn(n, generator=generator)
        return self.means[k] + self.sigma * z

    def responsibilities(self, x: Tensor) -> Tensor:
        """P(component k | x), shape (n, K)."""
        var = torch.full_like(self.means, self.variance)
        comp = _normal_logpdf(x.unsqueeze(-1), self.means, var) + torch.log(
            self.weights
        )
        return torch.softmax(comp, dim=-1)

    def mode_masses(self, x: Tensor) -> Tensor:
        """Soft per-mode share of a sample set, shape (K,). Sums to 1.

        Soft rather than hard assignment because neighbouring modes overlap;
        the soft version is the unbiased estimator of the mixture weights.
        """
        return self.responsibilities(x).mean(dim=0)

    def tilt(self, alpha: float) -> Mixture1D:
        """The law of this mixture reweighted by exp(alpha * x), normalised.

        N(m, s^2) * exp(a x) = exp(a m + a^2 s^2 / 2) * N(m + a s^2, s^2), so a
        tilted mixture is a mixture with shifted means and reweighted, and the
        per-component scale is untouched. The variance staying exactly s^2
        within each component is the discriminating check for a sampler: a
        degenerate one shifts the mean but collapses the spread.
        """
        log_w = torch.log(self.weights) + alpha * self.means
        return Mixture1D(
            weights=torch.softmax(log_w, dim=0),
            means=self.means + alpha * self.variance,
            sigma=self.sigma,
        )

    def marginal(self, t: float | Tensor) -> Mixture1D:
        """The law of x_t = (1-t) * z + t * x_1 with z ~ N(0, 1) independent."""
        t = float(t)
        var_t = (1.0 - t) ** 2 + (t * self.sigma) ** 2
        return Mixture1D(
            weights=self.weights.clone(),
            means=t * self.means,
            sigma=math.sqrt(var_t),
        )


class ExactVelocityModel:
    """The exact rectified-flow velocity field for a `Mixture1D` data law.

    Conditional on a component, (x_0, x_1) is jointly Gaussian, so
    ``E[x1 - x0 | x_t, k]`` is the linear field ``m_k + V'(t)/(2 V(t)) * (x -
    t m_k)``; the mixture velocity is those fields weighted by the time-t
    responsibilities. Substituting it into `AffineFlowMarginalPreservingSDE`
    reproduces the mixture exactly in the continuum limit, so any deviation the
    tests observe is discretisation or sampler error, never model error.
    """

    prediction_kind = PredictionKind.VELOCITY

    def __init__(self, mixture: Mixture1D, device: torch.device | None = None) -> None:
        self.mixture = mixture
        self._device = device or torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def __call__(self, x_t: DDTensor, t: Tensor, *, conditioning: Any) -> DDTensor:
        x = x_t.data.squeeze(-1)  # (n,)
        s2 = self.mixture.variance
        var_t = (1.0 - t) ** 2 + t**2 * s2  # (n,)
        dvar_t = -2.0 * (1.0 - t) + 2.0 * t * s2  # (n,)

        means_t = t.unsqueeze(-1) * self.mixture.means  # (n, K)
        log_resp = _normal_logpdf(
            x.unsqueeze(-1), means_t, var_t.unsqueeze(-1)
        ) + torch.log(self.mixture.weights)
        resp = torch.softmax(log_resp, dim=-1)  # (n, K)

        per_component = self.mixture.means + (
            dvar_t / (2.0 * var_t)
        ).unsqueeze(-1) * (x.unsqueeze(-1) - means_t)  # (n, K)
        v = (resp * per_component).sum(dim=-1)  # (n,)
        return DDTensor(v.unsqueeze(-1))


class _StandardNormalSampler:
    def sample(self, n, *, conditioning, device, generator=None):
        return (
            DDTensor(torch.randn(n, 1, device=device, generator=generator)),
            conditioning,
        )

    def sample_like(self, x_data, *, generator=None):
        return DDTensor(torch.randn(x_data.data.shape, device=x_data.data.device))


class _CoordinateReward:
    """r(x) = x, so the reward and the tilt direction agree by construction."""

    def __call__(self, *, sample, latent, conditioning) -> RewardBatch:
        return RewardBatch(rewards=latent.data.squeeze(-1))


def make_environment(mixture: Mixture1D) -> FlowEnvironment:
    """A `FlowEnvironment` whose only non-analytic ingredient is the time grid."""
    geometry = TensorGeometry()
    schedule = RectifiedFlowSchedule()
    base_sampler = _StandardNormalSampler()
    converter = PredictionConverter(geometry=geometry, schedule=schedule)
    return FlowEnvironment(
        geometry=geometry,
        base_sampler=base_sampler,
        forward_process=AffineGaussianForwardProcess(
            geometry=geometry, base_sampler=base_sampler, schedule=schedule
        ),
        regression=VelocityRegression(geometry=geometry, converter=converter),
        codec=IdentityCodec(),
        reward=_CoordinateReward(),
    )


def weighted_moments(x: Tensor, log_w: Tensor) -> tuple[float, float]:
    """Self-normalised importance-weighted (mean, variance) of a scalar sample.

    This is the reference the SMC output is compared against: applying the tilt
    to a *plain* SDE rollout by reweighting targets exactly the same law SMC is
    supposed to sample, including whatever discretisation error the shared time
    grid introduces — so a mismatch isolates the sampler.
    """
    w = torch.softmax(log_w, dim=0)
    mean = float((w * x).sum())
    var = float((w * (x - mean) ** 2).sum())
    return mean, var


def effective_sample_size(log_w: Tensor) -> float:
    return float(
        torch.exp(2 * torch.logsumexp(log_w, 0) - torch.logsumexp(2 * log_w, 0))
    )
