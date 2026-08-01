"""Sampling dynamics: convert path velocity to SDE drift and diffusion.

Three dynamics profiles are supported (per specs.md):

  ProbabilityFlowODE          — deterministic, drift = velocity
  AffineFlowMarginalPreservingSDE — stochastic, marginal-preserving correction
  MemorylessFlowSDE           — stochastic, memoryless (required by Adjoint Matching)

For a given path velocity v_θ and diffusion g(t), the marginal-preserving SDE drift is:

  b(x, t) = kappa(t) * x + c(t) * (v_θ(x,t) - kappa(t) * x)

where:
  kappa(t) = db_dt(t) / b(t)     (scalar; singular at t=0)
  eta(t)   = a(t) * (kappa(t)*a(t) - da_dt(t))
  c(t)     = (eta(t) + g(t)^2/2) / eta(t)

For memoryless dynamics: g(t) = sqrt(2*eta(t)), so c=2 and drift = 2*v_θ - kappa*x.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from diffusiongym.core.schedule import (
    AffineSchedule,
    MemorylessDiffusionSchedule,
    ScalarDiffusionSchedule,
)
from diffusiongym.types import DDBatch


@dataclass(frozen=True)
class DynamicsCoefficients[StateT]:
    """SDE drift and scalar diffusion coefficient at one time step.

    drift:     shape matching the state (batch, ...)
    diffusion: scalar per sample, shape (batch,)
    """

    drift: StateT
    diffusion: Tensor


class FlowDynamics[StateT](ABC):
    """Convert path velocity into sampling dynamics coefficients.

    Subclasses specify whether they are stochastic and whether they are memoryless,
    so that fine-tuning algorithms can validate their requirements at construction.
    """

    stochastic: ClassVar[bool]
    memoryless: ClassVar[bool]

    @abstractmethod
    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]: ...

    def validate(self) -> None:
        """Perform any schedule-level consistency checks."""


class ProbabilityFlowODE[StateT](FlowDynamics[StateT]):
    """Deterministic ODE sampler: drift = velocity, diffusion = 0.

    Supports: standard generation, ORW-CFM endpoint collection, DiffusionNFT rollouts,
    ORW-CFM, SD3.5 inference, FlowMol sampling.
    """

    stochastic: ClassVar[bool] = False
    memoryless: ClassVar[bool] = False

    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]:
        return DynamicsCoefficients(
            drift=velocity,
            diffusion=torch.zeros_like(t),
        )


class AffineFlowMarginalPreservingSDE[StateT: DDBatch](FlowDynamics[StateT]):
    """Marginal-preserving SDE for affine Gaussian flows.

    Uses state-independent scalar diffusion g(t). The drift is the unique
    correction that preserves the marginal distribution p_t(x) of the trained model.

    Supports: Flow-GRPO stochastic exploration, stochastic ablations.
    """

    stochastic: ClassVar[bool] = True
    memoryless: ClassVar[bool] = False

    def __init__(
        self,
        *,
        affine_schedule: AffineSchedule,
        diffusion_schedule: ScalarDiffusionSchedule,
    ) -> None:
        self.affine_schedule = affine_schedule
        self.diffusion_schedule = diffusion_schedule

    def _kappa_eta_sigma(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        a = self.affine_schedule.a(t)
        b = self.affine_schedule.b(t).clamp_min(1e-6)
        da = self.affine_schedule.da_dt(t)
        db = self.affine_schedule.db_dt(t)

        kappa = db / b
        eta = a * (kappa * a - da)
        sigma = self.diffusion_schedule.value(t)
        return kappa, eta, sigma

    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]:
        kappa, eta, sigma = self._kappa_eta_sigma(t)
        c = (eta + 0.5 * sigma**2) / eta.clamp_min(1e-8)

        # drift = kappa*x + c*(v - kappa*x)
        kappa_x = x * kappa
        drift = kappa_x + (velocity - kappa_x) * c
        return DynamicsCoefficients(drift=drift, diffusion=sigma)

    def relative_control(
        self,
        *,
        current_drift: StateT,
        reference_drift: StateT,
        t: Tensor,
    ) -> StateT:
        """Compute u s.t. b_current = b_reference + g(t) * u.

        Used for KL running cost and adjoint target computation.
        """
        sigma = self.diffusion_schedule.value(t).clamp_min(1e-8)
        return (current_drift - reference_drift) * (1.0 / sigma)


class MemorylessFlowSDE[StateT: DDBatch](AffineFlowMarginalPreservingSDE[StateT]):
    """Memoryless marginal-preserving SDE: g(t) = sqrt(2 * eta(t)).

    Required by Adjoint Matching to ensure x_0 and x_1 are conditionally
    independent given x_t (avoids initial-value bias in the adjoint computation).

    The drift simplifies to: b(x, t) = 2 * v_θ(x, t) - kappa(t) * x.
    """

    memoryless: ClassVar[bool] = True

    def __init__(
        self,
        *,
        affine_schedule: AffineSchedule,
    ) -> None:
        super().__init__(
            affine_schedule=affine_schedule,
            diffusion_schedule=MemorylessDiffusionSchedule(affine_schedule),
        )

    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]:
        # For memoryless: c = 2, so drift = kappa*x + 2*(v - kappa*x) = 2*v - kappa*x
        kappa, _eta, sigma = self._kappa_eta_sigma(t)
        kappa_x = x * kappa
        drift = velocity * 2.0 - kappa_x
        return DynamicsCoefficients(drift=drift, diffusion=sigma)
