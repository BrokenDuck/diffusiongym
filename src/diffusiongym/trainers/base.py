"""Base classes and shared types for fine-tuning algorithms.

All four algorithms share a common structure:
  1. validate() — check that requirements are met
  2. collect()  — sample online experience under the current rollout policy
  3. update()   — compute loss and take a gradient step on the train policy
  4. synchronize_rollout_policy() — optional EMA/hard-copy (algorithm-specific)

Experience types are distinct per algorithm because each algorithm needs
different trajectory data.
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Generator, Tensor

from diffusiongym.core import (
    EulerMaruyamaSampler,
    EulerODESampler,
    FlowDynamics,
    FlowEnvironment,
    LatentGeometry,
    PolicyBundle,
    Rollout,
    RolloutStorage,
)
from diffusiongym.types import DDBatch

type Conditioning = Mapping[str, Any]


# ---------------------------------------------------------------------------
# Time-grid stability (shared by every stochastic algorithm)
# ---------------------------------------------------------------------------


def deterministic_contraction[StateT: DDBatch](
    *,
    dynamics: FlowDynamics[StateT],
    geometry: LatentGeometry[StateT],
    x: StateT,
    t: Tensor,
    dt: Tensor,
) -> Tensor:
    """‖x + b(x, t; v=0) Δt‖ / ‖x‖ for each sample, shape (batch,).

    The drift of an affine-Gaussian SDE is linear in x, so with the velocity set
    to zero this is exactly the amplification factor of the state-dependent part
    of one Euler-Maruyama mean update. A value above 1 means the discretization
    is expansive: the step is not a usable approximation of the SDE, however
    small the model error is.
    """
    zero_velocity = x.map_state_tensors(torch.zeros_like)
    drift = dynamics.coefficients(x=x, t=t, velocity=zero_velocity).drift
    moved = x + drift * dt
    norm_x = geometry.squared_norm(x, reduction="sum").sqrt()
    norm_moved = geometry.squared_norm(moved, reduction="sum").sqrt()
    return norm_moved / norm_x.clamp_min(1e-12)


def check_time_grid_stability[StateT: DDBatch](
    *,
    rollout: Rollout[StateT, Any],
    dynamics: FlowDynamics[StateT],
    geometry: LatentGeometry[StateT],
    require: bool,
    algorithm: str,
) -> float:
    """Flag Euler-Maruyama steps whose deterministic part expands the state.

    Returns the worst amplification factor over the grid. Raises (``require``)
    or warns when any step is expansive.
    """
    worst = 0.0
    offenders: list[str] = []
    for k, step in enumerate(rollout.steps):
        factor = deterministic_contraction(
            dynamics=dynamics,
            geometry=geometry,
            x=step.x,
            t=step.t.to(step.x.device),
            dt=step.dt.to(step.x.device).reshape(1).expand(len(step.x)),
        ).max()
        worst = max(worst, float(factor.item()))
        if factor > 1.0 + 1e-4:
            offenders.append(
                f"step {k}: t={step.t[0].item():.4g}, dt={step.dt.item():.4g}, "
                f"|x| amplified by {factor.item():.2f}x"
            )
    if offenders:
        message = (
            f"{algorithm}: the time grid is unstable for these dynamics — the "
            "deterministic part of the Euler-Maruyama update expands the state "
            "instead of contracting it, so the transition is not an "
            "approximation of the SDE:\n  " + "\n  ".join(offenders) + "\n"
            "The drift of a marginal-preserving SDE carries a 1/t term, so a "
            "step needs dt small relative to t. Start the time grid at an "
            "interior t_min instead of 0 — torch.linspace(0, 1, T + 2)[1:] "
            "gives dt/t <= 1 everywhere — or use more steps."
        )
        if require:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return worst


@dataclass(frozen=True)
class FineTuningRequirements:
    """Capability requirements declared by each algorithm.

    Checked at validate() time so that misconfigurations fail early with a
    clear message rather than silently producing wrong results.
    """

    needs_reference_policy: bool = False
    needs_stochastic_rollout: bool = False
    needs_tractable_transitions: bool = False
    needs_memoryless_dynamics: bool = False
    needs_differentiable_terminal_cost: bool = False
    rollout_storage: RolloutStorage = field(default_factory=RolloutStorage)


@dataclass
class FineTuningContext[StateT: DDBatch, RawT]:
    """All resources an algorithm needs to collect experience and update.

    Owned by the caller; algorithms mutate policies.train in-place.
    """

    environment: FlowEnvironment[StateT, RawT]
    policies: PolicyBundle[StateT]
    optimizer: torch.optim.Optimizer
    ode_sampler: EulerODESampler[StateT, RawT]
    sde_sampler: EulerMaruyamaSampler[StateT, RawT]


# ---------------------------------------------------------------------------
# Experience types
# ---------------------------------------------------------------------------


@dataclass
class EndpointExperience[StateT]:
    """Terminal-only experience for endpoint-based algorithms (ORW-CFM, DiffusionNFT).

    Stores only the terminal latent and reward; no trajectory is retained.
    """

    latent: StateT
    rewards: Tensor
    valid: Tensor | None
    conditioning: Conditioning


@dataclass
class TrajectoryExperience[StateT]:
    """Full trajectory experience for policy-gradient algorithms (Flow-GRPO).

    Stores the complete rollout, per-sample advantages, and the dynamics used
    during collection so update() can recompute SDE drifts with the same kernel.
    """

    rollout: Rollout[StateT, Any]
    advantages: Tensor  # shape (n,)
    dynamics: FlowDynamics[StateT]


@dataclass
class AdjointExperience[StateT]:
    """Trajectory + adjoint targets for Adjoint Matching.

    velocity_targets[k] is the path-velocity regression target at step k,
    v_ref(x_k, t_k) - eta(t_k) * a_k, and loss_weights[k] is the 2 / sigma(t_k)^2
    factor that turns the velocity error back into the control-space Adjoint
    Matching loss (see trainers/adjoint_matching.py). Both are already detached.

    dynamics is stored so update() stays consistent with collection.
    """

    rollout: Rollout[StateT, Any]
    velocity_targets: list[StateT]  # one per rollout step
    loss_weights: list[Tensor]  # one per rollout step, shape (n,)
    dynamics: FlowDynamics[StateT]


# ---------------------------------------------------------------------------
# Base algorithm
# ---------------------------------------------------------------------------


class FineTuningAlgorithm[StateT: DDBatch, RawT, ExperienceT](ABC):
    """Abstract base for all fine-tuning algorithms.

    Subclasses declare requirements and implement collect() + update().
    The training loop is:

        for iteration in range(num_iterations):
            algorithm.validate(context=ctx, dynamics=dynamics)
            experience = algorithm.collect(context=ctx, dynamics=dynamics, ...)
            metrics = algorithm.update(context=ctx, experience=experience)
            algorithm.synchronize_rollout_policy(context=ctx)
    """

    @property
    @abstractmethod
    def requirements(self) -> FineTuningRequirements: ...

    def validate(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
    ) -> None:
        """Check that context and dynamics satisfy algorithm requirements.

        Raises ValueError on the first unmet requirement.
        """
        req = self.requirements

        if req.needs_reference_policy and context.policies.reference is None:
            raise ValueError(
                f"{type(self).__name__} requires a reference policy "
                "(policies.reference must not be None)."
            )

        if req.needs_stochastic_rollout and not dynamics.stochastic:
            raise ValueError(
                f"{type(self).__name__} requires stochastic dynamics "
                "(dynamics.stochastic must be True). "
                "Use AffineFlowMarginalPreservingSDE or MemorylessFlowSDE."
            )

        if req.needs_memoryless_dynamics and not dynamics.memoryless:
            raise ValueError(
                f"{type(self).__name__} requires memoryless dynamics "
                "(dynamics.memoryless must be True). "
                "Use MemorylessFlowSDE."
            )

        if (
            req.needs_differentiable_terminal_cost
            and context.environment.terminal_cost is None
        ):
            raise ValueError(
                f"{type(self).__name__} requires a differentiable terminal cost "
                "(environment.terminal_cost must not be None). "
                "Supply a DifferentiableTerminalCost — a black-box RewardEvaluator "
                "is not sufficient because it may have zero gradients."
            )

    @abstractmethod
    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        n: int,
        time_grid: Tensor,
        conditioning: Conditioning,
        generator: Generator | None = None,
    ) -> ExperienceT: ...

    @abstractmethod
    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: ExperienceT,
    ) -> Mapping[str, float]:
        """Compute loss, take a gradient step, return metrics dict."""
        ...

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Optionally update the rollout policy from the train policy.

        Default: no-op. Override for EMA (DiffusionNFT) or hard-copy (Flow-GRPO).
        """
