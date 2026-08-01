"""Forward process: base sampler, forward batch, and affine Gaussian interpolation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Generator, Tensor

from diffusiongym.core.schedule import AffineSchedule
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch

Conditioning = Mapping[str, Any]


class BaseSampler[StateT](Protocol):
    """Standard Gaussian base distribution sampler.

    Implementations must sample from N(0, I), possibly projected or masked.
    Non-Gaussian bases are not supported.
    """

    def sample(
        self,
        n: int,
        *,
        conditioning: Conditioning,
        device: torch.device,
        generator: Generator | None = None,
    ) -> tuple[StateT, Conditioning]:
        """Sample n base states for generation.

        Returns the base states and a (possibly augmented) conditioning dict.
        """
        ...

    def sample_like(
        self,
        x_data: StateT,
        *,
        generator: Generator | None = None,
    ) -> StateT:
        """Sample base noise matching the shape of a training batch."""
        ...


@dataclass(frozen=True)
class ForwardBatch[StateT]:
    """Output of one forward-process interpolation step.

    All fields have the same batch size.
    """

    x_data: StateT
    x_base: StateT
    x_t: StateT             # a(t)*x_base + b(t)*x_data, projected
    target_velocity: StateT  # da_dt(t)*x_base + db_dt(t)*x_data, projected
    t: Tensor               # shape (n,)
    conditioning: Conditioning


class AffineGaussianForwardProcess[StateT: DDBatch]:
    """Affine Gaussian forward process.

    Computes x_t = a(t) * x_base + b(t) * x_data and the corresponding
    conditional target velocity u_t* = da_dt(t) * x_base + db_dt(t) * x_data.

    This is the only forward process supported; no abstraction is needed
    because the framework is deliberately narrow (affine-Gaussian only).
    """

    def __init__(
        self,
        *,
        geometry: LatentGeometry[StateT],
        base_sampler: BaseSampler[StateT],
        schedule: AffineSchedule,
    ) -> None:
        self.geometry = geometry
        self.base_sampler = base_sampler
        self.schedule = schedule

    def sample_time(
        self,
        n: int,
        *,
        device: torch.device,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
        generator: Generator | None = None,
    ) -> Tensor:
        """Sample n uniform times in [t_min, t_max]."""
        return torch.rand(n, device=device, generator=generator) * (t_max - t_min) + t_min

    def make_batch(
        self,
        x_data: StateT,
        *,
        conditioning: Conditioning,
        t: Tensor | None = None,
        generator: Generator | None = None,
    ) -> ForwardBatch[StateT]:
        """Construct a training batch by interpolating between base and data.

        Parameters
        ----------
        x_data:
            Clean data samples (x_1), shape (n, ...).
        conditioning:
            Conditioning inputs passed through to the batch.
        t:
            Times in [0, 1], shape (n,). Sampled uniformly if None.
        generator:
            Optional RNG for reproducibility.
        """
        n = len(x_data)
        device = x_data.device

        if t is None:
            t = self.sample_time(n, device=device, generator=generator)

        x_base = self.base_sampler.sample_like(x_data, generator=generator)

        a = self.schedule.a(t)
        b = self.schedule.b(t)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)

        x_t = x_base * a + x_data * b
        target_velocity = x_base * da + x_data * db

        return ForwardBatch(
            x_data=x_data,
            x_base=x_base,
            x_t=self.geometry.project(x_t),
            target_velocity=self.geometry.project(target_velocity),
            t=t,
            conditioning=conditioning,
        )
