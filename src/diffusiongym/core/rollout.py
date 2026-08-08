"""Rollout data structures and samplers.

Two samplers are provided:
  EulerODESampler      — deterministic, requires ProbabilityFlowODE dynamics
  EulerMaruyamaSampler — stochastic, requires stochastic dynamics + kernel factory

The Rollout and RolloutStep types carry exactly what each fine-tuning algorithm
needs, controlled by RolloutStorage flags.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.kernel import EulerGaussianKernelFactory, GaussianMarkovKernel
from diffusiongym.core.reward import RewardBatch
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch

type Conditioning = Mapping[str, Any]


@dataclass(frozen=True)
class RolloutStorage:
    """Flags controlling what data is retained during a rollout.

    Storing intermediate states, noises, etc. increases memory. Only enable
    what the algorithm actually needs.
    """

    states: bool = False  # x at each step (needed by Flow-GRPO, Adjoint Matching)
    noises: bool = False  # Gaussian noise samples (Adjoint Matching)
    drifts: bool = False  # drift at each step (Adjoint Matching adjoint)
    predictions: bool = False  # raw model predictions
    log_probs: bool = False  # per-step log-probabilities (Flow-GRPO)


@dataclass(frozen=True)
class RolloutRequest:
    """Parameters for a single rollout call."""

    time_grid: Tensor  # shape (T+1,), e.g. torch.linspace(0, 1, N+1)
    storage: RolloutStorage = field(default_factory=RolloutStorage)
    requires_grad: bool = False
    evaluate_reward: bool = True


@dataclass
class RolloutStep[StateT]:
    """Data from a single integration step.

    x:        state at t_k
    x_next:   state at t_{k+1}
    t:        time t_k, shape (batch,)
    dt:       step size t_{k+1} - t_k, scalar
    drift:    optional, stored if storage.drifts
    diffusion: optional scalar per sample, shape (batch,)
    noise:    optional Gaussian noise used in SDE step
    log_prob: optional log p(x_next | x_k), shape (batch,)
    """

    x: StateT
    x_next: StateT
    t: Tensor
    dt: Tensor
    drift: StateT | None = None
    diffusion: Tensor | None = None
    noise: StateT | None = None
    log_prob: Tensor | None = None


@dataclass
class SMCStats:
    """Per-rollout resampling diagnostics, produced only by `SMCSampler`.

    ess_trace:
        Effective sample size *before* the resample decision at each step,
        shape (num_steps,). ESS == n means the potential has not yet
        differentiated the particles at all; ESS well below n signals the
        tilt is concentrating weight on a shrinking subset.
    resampled:
        Whether a resample fired at each step, shape (num_steps,).
    num_resamples:
        `resampled.sum()`, kept as a plain int for cheap logging.
    """

    ess_trace: Tensor
    resampled: Tensor
    num_resamples: int


@dataclass
class Rollout[StateT, RawT]:
    """Complete rollout output."""

    terminal_latent: StateT
    terminal_sample: RawT | None
    reward: RewardBatch | None
    steps: list[RolloutStep[StateT]]
    conditioning: Conditioning
    smc: SMCStats | None = None


class EulerODESampler[StateT: DDBatch, RawT]:
    """Euler ODE sampler: X_{k+1} = X_k + dt * drift.

    Requires deterministic (non-stochastic) dynamics. Raises ValueError if
    stochastic dynamics are passed to ensure algorithms don't silently fall back
    to ODE when they need SDE.
    """

    def __init__(self, geometry: LatentGeometry[StateT]) -> None:
        self.geometry = geometry

    @torch.no_grad()
    def rollout(
        self,
        *,
        environment: Any,  # FlowEnvironment (avoid circular import)
        model: Any,  # FlowModel
        dynamics: FlowDynamics[StateT],
        n: int,
        conditioning: Conditioning,
        request: RolloutRequest,
        generator: Generator | None = None,
    ) -> Rollout[StateT, RawT]:
        if dynamics.stochastic:
            raise ValueError(
                "EulerODESampler requires deterministic dynamics (stochastic=False). "
                "Use EulerMaruyamaSampler for stochastic dynamics."
            )

        device = model.device
        time_grid = request.time_grid.to(device)

        # Sample initial state
        x_base, conditioning = environment.base_sampler.sample(
            n, conditioning=conditioning, device=device, generator=generator
        )
        x = x_base

        steps: list[RolloutStep[StateT]] = []

        for t0, t1 in pairwise(time_grid):
            dt = t1 - t0
            t_eval = t0.clamp_min(1e-2).expand(n)

            velocity = environment.predict_velocity(
                model, x_t=x, t=t_eval, conditioning=conditioning
            )
            coeffs = dynamics.coefficients(x=x, t=t_eval, velocity=velocity)

            x_next = x + coeffs.drift * dt

            step = RolloutStep(
                x=x.clone().detach() if request.storage.states else x,
                x_next=x_next.clone().detach(),
                t=t_eval.detach().cpu(),
                dt=dt.detach().cpu(),
                drift=coeffs.drift.detach() if request.storage.drifts else None,
                diffusion=coeffs.diffusion.detach().cpu()
                if request.storage.drifts
                else None,
            )
            steps.append(step)
            x = x_next

        terminal_latent = x
        terminal_sample = None
        reward_batch = None

        if request.evaluate_reward:
            terminal_sample, reward_batch = environment.evaluate_terminal(
                terminal_latent, conditioning=conditioning
            )

        return Rollout(
            terminal_latent=terminal_latent.detach(),
            terminal_sample=terminal_sample,
            reward=reward_batch,
            steps=steps,
            conditioning=conditioning,
        )


class EulerMaruyamaSampler[StateT: DDBatch, RawT]:
    """Euler-Maruyama SDE sampler: X_{k+1} ~ N(X_k + dt*drift, g²*dt*I).

    Requires stochastic dynamics. The kernel factory produces GaussianMarkovKernel
    objects that are the SAME objects used for both sampling AND log-prob evaluation,
    ensuring exact importance ratios in Flow-GRPO.
    """

    def __init__(
        self,
        geometry: LatentGeometry[StateT],
        kernel_factory: EulerGaussianKernelFactory[StateT],
    ) -> None:
        self.geometry = geometry
        self.kernel_factory = kernel_factory

    @torch.no_grad()
    def rollout(
        self,
        *,
        environment: Any,  # FlowEnvironment (avoid circular import)
        model: Any,  # FlowModel
        dynamics: FlowDynamics[StateT],
        n: int,
        conditioning: Conditioning,
        request: RolloutRequest,
        generator: Generator | None = None,
    ) -> Rollout[StateT, RawT]:
        if not dynamics.stochastic:
            raise ValueError(
                "EulerMaruyamaSampler requires stochastic dynamics (stochastic=True). "
                "Use EulerODESampler for deterministic dynamics."
            )

        device = model.device
        time_grid = request.time_grid.to(device)

        x_base, conditioning = environment.base_sampler.sample(
            n, conditioning=conditioning, device=device, generator=generator
        )
        x = x_base

        steps: list[RolloutStep[StateT]] = []

        for t0, t1 in pairwise(time_grid):
            dt = t1 - t0
            t_eval = t0.clamp_min(1e-2).expand(n)

            velocity = environment.predict_velocity(
                model, x_t=x, t=t_eval, conditioning=conditioning
            )
            coeffs = dynamics.coefficients(x=x, t=t_eval, velocity=velocity)

            kernel: GaussianMarkovKernel[StateT] = self.kernel_factory.build(
                x=x,
                t=t_eval,
                dt=dt.unsqueeze(0).expand(n),
                drift=coeffs.drift,
                diffusion=coeffs.diffusion,
            )

            x_next = kernel.rsample(generator=generator)

            log_prob = kernel.log_prob(x_next) if request.storage.log_probs else None
            noise = (x_next - kernel.mean) if request.storage.noises else None

            step = RolloutStep(
                x=x.clone().detach() if request.storage.states else x,
                x_next=x_next.clone().detach(),
                t=t_eval.detach().cpu(),
                dt=dt.detach().cpu(),
                drift=coeffs.drift.detach() if request.storage.drifts else None,
                diffusion=coeffs.diffusion.detach().cpu(),
                noise=noise.detach() if noise is not None else None,
                log_prob=log_prob.detach().cpu() if log_prob is not None else None,
            )
            steps.append(step)
            x = x_next

        terminal_latent = x
        terminal_sample = None
        reward_batch = None

        if request.evaluate_reward:
            terminal_sample, reward_batch = environment.evaluate_terminal(
                terminal_latent, conditioning=conditioning
            )

        return Rollout(
            terminal_latent=terminal_latent.detach(),
            terminal_sample=terminal_sample,
            reward=reward_batch,
            steps=steps,
            conditioning=conditioning,
        )
