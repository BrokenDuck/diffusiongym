"""Base classes and shared types for fine-tuning algorithms.

All four algorithms share a common structure:
  1. validate() — check that requirements are met
  2. collect()  — sample online experience under the current rollout policy
  3. update()   — compute loss and take a gradient step on the train policy
  4. synchronize_rollout_policy() — optional EMA/hard-copy (algorithm-specific)

Experience types are distinct per algorithm because each algorithm needs
different trajectory data.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.environment import FlowEnvironment, PolicyBundle
from diffusiongym.core.rollout import (
    EulerMaruyamaSampler,
    EulerODESampler,
    Rollout,
    RolloutStorage,
)
from diffusiongym.types import DDBatch

type Conditioning = Mapping[str, Any]


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

    Stores the complete rollout and per-sample advantages.
    """

    rollout: Rollout[StateT, Any]
    advantages: Tensor  # shape (n,)


@dataclass
class AdjointExperience[StateT]:
    """Trajectory + adjoint targets for Adjoint Matching.

    adjoint_targets[k] is the per-step drift regression target at step k.
    dynamics is stored so update() can recompute SDE drifts without re-acquiring it.
    """

    rollout: Rollout[StateT, Any]
    adjoint_targets: list[StateT]  # one per rollout step
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
