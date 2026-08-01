"""Markov transition kernels for Euler-Maruyama sampling.

The GaussianMarkovKernel must be the same object used to generate a trajectory
step AND to evaluate its log-probability. This identity ensures that Flow-GRPO
importance ratios are exact, not approximations.

Transition kernel structure:
  X_{k+1} | X_k ~ N(X_k + dt * drift_k, sigma_k^2 * dt * I)

The same kernel is used to:
  1. Sample x_{k+1} = kernel.rsample()
  2. Evaluate log p(x_{k+1} | x_k) = kernel.log_prob(x_{k+1})
  3. Compute KL(policy || reference) = kernel.kl_divergence(ref_kernel)
"""

import math
from typing import Protocol

import torch
from torch import Generator, Tensor

from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch


class MarkovKernel[StateT](Protocol):
    """Protocol for a Markov transition kernel p(x_{k+1} | x_k)."""

    def rsample(self, *, generator: Generator | None = None) -> StateT:
        """Draw a sample from the kernel."""
        ...

    def log_prob(self, value: StateT) -> Tensor:
        """Compute log p(value), one scalar per batch element, shape (batch,)."""
        ...

    def kl_divergence(self, other: "MarkovKernel[StateT]") -> Tensor:
        """KL(self || other), shape (batch,)."""
        ...


class GaussianMarkovKernel[StateT: DDBatch]:
    """Isotropic Gaussian transition kernel with scalar variance.

    p(x | mean) = N(mean, variance * I)

    Parameters
    ----------
    geometry:
        LatentGeometry for norm computation and sampling.
    mean:
        Distribution mean, shape (batch, ...).
    variance:
        Scalar variance per batch element, shape (batch,).
        Must be strictly positive.
    """

    def __init__(
        self,
        geometry: LatentGeometry[StateT],
        mean: StateT,
        variance: Tensor,
    ) -> None:
        self.geometry = geometry
        self.mean = mean
        self.variance = variance

    def rsample(self, *, generator: Generator | None = None) -> StateT:
        """Sample x ~ N(mean, variance * I)."""
        noise = self.geometry.standard_normal_like(self.mean, generator=generator)
        std = self.variance.clamp_min(1e-12).sqrt()
        return self.mean + noise * std

    def log_prob(self, value: StateT) -> Tensor:
        """log N(value; mean, variance * I), shape (batch,).

        log p = -d/2 * log(2π) - d/2 * log(variance) - ||value - mean||² / (2*variance)
        """
        d = self.geometry.active_dimensions(self.mean)  # shape (batch,)
        sq = self.geometry.squared_norm(
            value - self.mean,
            reduction="sum",
        )  # shape (batch,)
        var = self.variance.clamp_min(1e-12)
        return -0.5 * (d * math.log(2 * math.pi) + d * torch.log(var) + sq / var)

    def kl_divergence(self, other: "GaussianMarkovKernel[StateT]") -> Tensor:
        """KL(self || other) for isotropic Gaussians, shape (batch,).

        When both kernels have the same scalar variance (same dynamics, different
        drift predictions), this simplifies to:
          KL = ||mean_self - mean_other||² / (2 * variance_other)

        For the general case:
          KL = d/2 * [log(var_other/var_self) + var_self/var_other - 1]
               + ||mean_self - mean_other||² / (2 * var_other)
        """
        d = self.geometry.active_dimensions(self.mean)
        var_p = self.variance.clamp_min(1e-12)
        var_q = other.variance.clamp_min(1e-12)

        mean_sq = self.geometry.squared_norm(
            self.mean - other.mean,
            reduction="sum",
        )

        return 0.5 * (
            d * (torch.log(var_q / var_p) + var_p / var_q - 1.0) + mean_sq / var_q
        )


class EulerGaussianKernelFactory[StateT: DDBatch](Protocol):
    """Protocol for building Euler-Maruyama Gaussian transition kernels."""

    def build(
        self,
        *,
        x: StateT,
        t: Tensor,
        dt: Tensor,
        drift: StateT,
        diffusion: Tensor,
    ) -> GaussianMarkovKernel[StateT]: ...


class DefaultEulerGaussianKernelFactory[StateT: DDBatch]:
    """Build Gaussian kernels for X_{k+1} | X_k ~ N(X_k + dt*drift, g²*dt*I).

    Parameters
    ----------
    geometry:
        LatentGeometry for Gaussian sampling and norm computation.
    """

    def __init__(self, geometry: LatentGeometry[StateT]) -> None:
        self.geometry = geometry

    def build(
        self,
        *,
        x: StateT,
        t: Tensor,
        dt: Tensor,
        drift: StateT,
        diffusion: Tensor,
    ) -> GaussianMarkovKernel[StateT]:
        mean = x + drift * dt
        variance = diffusion**2 * dt
        return GaussianMarkovKernel(self.geometry, mean, variance)
