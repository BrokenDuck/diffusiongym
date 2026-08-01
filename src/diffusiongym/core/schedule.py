"""Affine interpolation schedules and diffusion schedules for flow matching.

Three distinct schedule concepts are kept separate (per specs.md):
  1. AffineSchedule   — interpolation path a(t), b(t)
  2. ScalarDiffusionSchedule — SDE diffusion coefficient g(t)
  3. Time grid (owned by the caller, e.g. torch.linspace)

Convention: t=0 → Gaussian base, t=1 → data
  x_t = a(t) * x_base + b(t) * x_data
  a(0)=1, b(0)=0, a(1)=0, b(1)=1
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class AffineSchedule(ABC):
    """Scalar affine interpolation schedule.

    Canonical direction: t=0 is the Gaussian base, t=1 is data.
    """

    @abstractmethod
    def a(self, t: Tensor) -> Tensor:
        """Noise coefficient a(t); a(0)=1, a(1)=0."""
        ...

    @abstractmethod
    def b(self, t: Tensor) -> Tensor:
        """Data coefficient b(t); b(0)=0, b(1)=1."""
        ...

    @abstractmethod
    def da_dt(self, t: Tensor) -> Tensor:
        """d/dt a(t)."""
        ...

    @abstractmethod
    def db_dt(self, t: Tensor) -> Tensor:
        """d/dt b(t)."""
        ...

    def validate(self) -> None:
        """Check endpoint conditions at t=0 and t=1 (approximate)."""
        eps = torch.tensor(1e-6)
        one = torch.tensor(1.0 - 1e-6)

        assert torch.allclose(self.a(eps), torch.ones(1), atol=1e-4), "a(0) should be 1"
        assert torch.allclose(self.b(eps), torch.zeros(1), atol=1e-4), (
            "b(0) should be 0"
        )
        assert torch.allclose(self.a(one), torch.zeros(1), atol=1e-4), (
            "a(1) should be 0"
        )
        assert torch.allclose(self.b(one), torch.ones(1), atol=1e-4), "b(1) should be 1"


class RectifiedFlowSchedule(AffineSchedule):
    """Default linear schedule: a(t)=1-t, b(t)=t.

    Gives x_t = (1-t)*x_base + t*x_data and target velocity = x_data - x_base.
    """

    def a(self, t: Tensor) -> Tensor:
        return 1.0 - t

    def b(self, t: Tensor) -> Tensor:
        return t

    def da_dt(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)

    def db_dt(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)


# ---------------------------------------------------------------------------
# Diffusion schedules g(t) — separate from the interpolation schedule
# ---------------------------------------------------------------------------


class ScalarDiffusionSchedule(ABC):
    """Scalar state-independent SDE diffusion coefficient g(t).

    The SDE is: dX = b(X,t) dt + g(t) dW
    where b is determined by the chosen dynamics profile (marginal-preserving
    correction of the learned velocity).

    g(t) must be strictly positive for t in the interior (0, 1).
    """

    @abstractmethod
    def value(self, t: Tensor) -> Tensor:
        """Compute g(t), shape matching t."""
        ...


class ConstantDiffusionSchedule(ScalarDiffusionSchedule):
    """Constant diffusion g(t) = c."""

    def __init__(self, c: float) -> None:
        if c <= 0:
            raise ValueError(f"Diffusion constant must be positive, got {c}")
        self.c = c

    def value(self, t: Tensor) -> Tensor:
        return torch.full_like(t, self.c)


class MemorylessDiffusionSchedule(ScalarDiffusionSchedule):
    """Memoryless diffusion schedule: g(t) = sqrt(2 * eta(t)).

    This is the unique diffusion that makes x_0 and x_1 conditionally independent
    given x_t, which is required by Adjoint Matching to avoid initial-value bias.

    eta(t) = a(t) * (kappa(t) * a(t) - da_dt(t))
    where kappa(t) = db_dt(t) / b(t)

    Singular at t=0 (b(0)=0). The caller must ensure t >= t_min > 0.
    """

    def __init__(self, schedule: AffineSchedule) -> None:
        self.schedule = schedule

    def eta(self, t: Tensor) -> Tensor:
        a = self.schedule.a(t)
        b = self.schedule.b(t).clamp_min(1e-6)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)
        kappa = db / b
        return a * (kappa * a - da)

    def value(self, t: Tensor) -> Tensor:
        return (2.0 * self.eta(t)).clamp_min(0.0).sqrt()
