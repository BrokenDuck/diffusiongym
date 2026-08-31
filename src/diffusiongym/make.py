"""Assemble a ready-to-run fine-tuning setup from registry ids.

`make()` turns three strings into everything the training loop needs:

    setup = diffusiongym.make(
        modality="toy/gmm2d",
        reward="toy/linear",
        algorithm="adjoint_matching",
        discretization_steps=40,
    )
    for _ in range(iterations):
        experience = setup.algorithm.collect(
            context=setup.context, dynamics=setup.dynamics,
            n=64, time_grid=setup.time_grid, conditioning={},
        )
        metrics = setup.algorithm.update(
            context=setup.context, experience=experience
        )
        setup.algorithm.synchronize_rollout_policy(context=setup.context)

The wiring it removes is not boilerplate — most of it is a choice that has to
agree with the algorithm, and each disagreement is a silent failure rather than
a crash:

  * the SDE profile. Adjoint Matching is only correct under the memoryless
    schedule; Flow-GRPO needs a stochastic but not necessarily memoryless one;
    ORW-CFM and DiffusionNFT need the deterministic ODE. `make()` reads this off
    `algorithm.requirements` instead of asking.
  * the time grid. A marginal-preserving drift carries a kappa(t) = 1/t term, so
    a stochastic rollout on a grid that touches t=0 is expansive and destroys the
    trajectory. `make()` returns an interior grid whenever the dynamics are
    stochastic.
  * the reference policy, built only for the algorithms that anchor to one.
  * the differentiable terminal cost, which only Adjoint Matching needs and only
    some rewards can supply.

`make()` finishes by calling `algorithm.validate()`, so a bad combination fails
here with a message rather than after a rollout.

Nothing in this module is specific to dense tensors — a `ModalityProvider`
returning a graph geometry and a graph base sampler assembles identically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from diffusiongym.core.dynamics import (
    AffineFlowMarginalPreservingSDE,
    FlowDynamics,
    MemorylessFlowSDE,
    ProbabilityFlowODE,
)
from diffusiongym.core.environment import FlowEnvironment, PolicyBundle
from diffusiongym.core.kernel import DefaultEulerGaussianKernelFactory
from diffusiongym.core.model import PredictionConverter, VelocityRegression
from diffusiongym.core.process import AffineGaussianForwardProcess
from diffusiongym.core.rollout import EulerMaruyamaSampler, EulerODESampler
from diffusiongym.core.schedule import ScaledMemorylessDiffusionSchedule
from diffusiongym.registry import (
    algorithm_registry,
    domain_of,
    modality_registry,
    reward_provider_registry,
)
from diffusiongym.trainers.base import FineTuningContext, FineTuningRequirements

type OptimizerFactory = Callable[[Any], torch.optim.Optimizer]


@dataclass(frozen=True)
class FineTuningSetup:
    """Everything a training loop needs, already checked for consistency.

    Attributes
    ----------
    environment:
        Immutable services (geometry, forward process, regression, reward).
    context:
        Environment + policies + optimizer + samplers.
    algorithm:
        The configured `FineTuningAlgorithm`, or `None` when `make()` was given
        bare `requirements` instead of a registered id.
    dynamics:
        The SDE/ODE profile chosen from the requirements.
    time_grid:
        A grid valid for those dynamics — interior when they are stochastic.
    """

    environment: FlowEnvironment
    context: FineTuningContext
    algorithm: Any | None
    dynamics: FlowDynamics
    time_grid: Tensor


def _register_builtins() -> None:
    """Register the shipped algorithms and providers on first use.

    Imported lazily: `toy/` depends on `core/`, so importing it at module scope
    would make `diffusiongym.make` part of an import cycle, and it also keeps
    optional modality dependencies (a VAE, a graph library) off the base import
    path. Registering the algorithms here rather than in `trainers/__init__.py`
    keeps the trainers themselves free of any registry dependency.
    """
    if "adjoint_matching" in algorithm_registry:
        return

    from diffusiongym.trainers import (
        ORWCFM,
        AdjointMatching,
        DiffusionNFT,
        FlowGRPO,
        ForwardKLDistillation,
    )

    algorithm_registry.register("orw_cfm", ORWCFM)
    algorithm_registry.register("diffusion_nft", DiffusionNFT)
    algorithm_registry.register("flow_grpo", FlowGRPO)
    algorithm_registry.register("adjoint_matching", AdjointMatching)
    algorithm_registry.register("forward_kl_distillation", ForwardKLDistillation)

    import diffusiongym.toy.providers  # noqa: F401


def make(
    *,
    modality: str,
    reward: str,
    algorithm: str | None = None,
    requirements: FineTuningRequirements | None = None,
    discretization_steps: int = 40,
    device: torch.device | str | None = None,
    learning_rate: float = 3e-4,
    optimizer_factory: OptimizerFactory | None = None,
    noise_scale: float = 0.75,
    modality_kwargs: dict[str, Any] | None = None,
    reward_kwargs: dict[str, Any] | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
) -> FineTuningSetup:
    """Build a `FineTuningSetup` from registered ids.

    Parameters
    ----------
    modality:
        Id in `modality_registry`, e.g. "toy/gmm2d".
    reward:
        Id in `reward_provider_registry`, e.g. "toy/linear".
    algorithm:
        Id in `algorithm_registry`: "orw_cfm", "diffusion_nft", "flow_grpo",
        or "adjoint_matching". Mutually exclusive with `requirements`.
    requirements:
        Wire the setup from these capability requirements instead of from a
        registered algorithm's. `setup.algorithm` is then `None` and nothing is
        validated against it — there is no algorithm object to validate.

        This is for a caller that owns its own training step and only wants the
        wiring: the SDE/ODE profile, the interior time grid, the reference
        policy and the terminal cost still have to agree with what that step
        assumes, and each disagreement is a silent failure rather than a crash,
        which is precisely what `make()` is for. Registering a class whose only
        real content is a `requirements` property, purely to reach this
        function, would be a registry entry that names nothing.
    discretization_steps:
        Rollout steps. Adjoint Matching integrates its adjoint backward with
        explicit Euler, so its error accumulates over the trajectory and shows
        up as an *under-applied* tilt rather than as instability — give it
        several times more steps than the others (see its module docstring).
    device:
        Defaults to CPU.
    learning_rate:
        Learning rate for the default Adam optimizer.
    optimizer_factory:
        Callable taking the train model and returning an optimizer. Overrides
        `learning_rate` when given.
    noise_scale:
        SDE noise level for algorithms that need stochastic-but-not-memoryless
        dynamics (Flow-GRPO). Ignored otherwise; memoryless pins it to 1.
    modality_kwargs, reward_kwargs, algorithm_kwargs:
        Constructor overrides for the three registered classes.

    Raises
    ------
    KeyError
        If an id is not registered; the message lists what is.
    ValueError
        If the modality and reward come from different domains, or the
        algorithm's requirements cannot be met (for instance Adjoint Matching
        with a reward that has no differentiable form).
    """
    if (algorithm is None) == (requirements is None):
        raise ValueError(
            "make() takes exactly one of `algorithm` (a registered id) and "
            "`requirements` (capability requirements for a training step the "
            f"caller owns); got algorithm={algorithm!r} and "
            f"requirements={requirements!r}."
        )

    _register_builtins()

    modality_domain = domain_of(modality)
    reward_domain = domain_of(reward)
    if modality_domain and reward_domain and modality_domain != reward_domain:
        raise ValueError(
            f"Incompatible domains: modality {modality!r} is "
            f"{modality_domain!r} but reward {reward!r} is {reward_domain!r}. "
            "They must describe the same kind of data."
        )

    provider = modality_registry.get(modality).instantiate(**(modality_kwargs or {}))
    reward_source = reward_provider_registry.get(reward).instantiate(
        **(reward_kwargs or {})
    )
    algo = (
        algorithm_registry.get(algorithm).instantiate(**(algorithm_kwargs or {}))
        if algorithm is not None
        else None
    )
    if algo is not None:
        requirements = algo.requirements
    assert requirements is not None  # guaranteed by the exactly-one check above

    device = torch.device(device) if device is not None else torch.device("cpu")
    geometry = provider.geometry()
    schedule = provider.schedule()
    base_sampler = provider.base_sampler()

    terminal_cost = reward_source.terminal_cost()
    if requirements.needs_differentiable_terminal_cost and terminal_cost is None:
        raise ValueError(
            f"{algorithm or 'the given requirements'!r} requires a "
            f"differentiable terminal cost, but reward {reward!r} does not "
            "provide one. Pair it with a differentiable reward, or choose an "
            "algorithm that only needs a black-box reward."
        )

    environment = FlowEnvironment(
        geometry=geometry,
        base_sampler=base_sampler,
        forward_process=AffineGaussianForwardProcess(
            geometry=geometry, base_sampler=base_sampler, schedule=schedule
        ),
        regression=VelocityRegression(
            geometry=geometry,
            converter=PredictionConverter(geometry=geometry, schedule=schedule),
        ),
        codec=provider.codec(),
        reward=reward_source.reward(),
        terminal_cost=terminal_cost,
    )

    dynamics = _make_dynamics(
        requirements=requirements, schedule=schedule, noise_scale=noise_scale
    )

    train_model = provider.model(device=device)
    rollout_model = _replica(provider, train_model, device)
    reference_model = (
        _replica(provider, train_model, device)
        if requirements.needs_reference_policy
        else None
    )

    optimizer = (
        optimizer_factory(train_model)
        if optimizer_factory is not None
        else torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    )

    context = FineTuningContext(
        environment=environment,
        policies=PolicyBundle(
            train=train_model, rollout=rollout_model, reference=reference_model
        ),
        optimizer=optimizer,
        ode_sampler=EulerODESampler(geometry),
        sde_sampler=EulerMaruyamaSampler(
            geometry, DefaultEulerGaussianKernelFactory(geometry)
        ),
    )

    time_grid = make_time_grid(
        discretization_steps, stochastic=dynamics.stochastic, device=device
    )

    # Fail here rather than after a rollout. Nothing to check when the caller
    # supplied the requirements directly: `validate` compares an algorithm
    # against them, and both sides would be the same object.
    if algo is not None:
        algo.validate(context=context, dynamics=dynamics)

    return FineTuningSetup(
        environment=environment,
        context=context,
        algorithm=algo,
        dynamics=dynamics,
        time_grid=time_grid,
    )


def make_time_grid(
    discretization_steps: int,
    *,
    stochastic: bool,
    device: torch.device | str | None = None,
) -> Tensor:
    """Time grid valid for the given dynamics.

    Stochastic dynamics get an interior grid. The drift of a marginal-preserving
    SDE carries a kappa(t) = 1/t term, so the deterministic part of an
    Euler-Maruyama step at t=0 *expands* the state instead of contracting it —
    with T=10 the mean update becomes x -> -9x and the trajectory leaves the data
    manifold entirely. Steps of 1/(T+1) starting at t=1/(T+1) keep dt/t <= 1
    everywhere.
    """
    if discretization_steps < 1:
        raise ValueError(
            f"discretization_steps must be at least 1, got {discretization_steps}."
        )
    if stochastic:
        return torch.linspace(0.0, 1.0, discretization_steps + 2, device=device)[1:]
    return torch.linspace(0.0, 1.0, discretization_steps + 1, device=device)


def _make_dynamics(*, requirements, schedule, noise_scale: float) -> FlowDynamics:
    """Pick the SDE/ODE profile the algorithm's requirements imply."""
    if requirements.needs_memoryless_dynamics:
        return MemorylessFlowSDE(affine_schedule=schedule)
    if requirements.needs_stochastic_rollout:
        return AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ScaledMemorylessDiffusionSchedule(schedule, noise_scale),
        )
    return ProbabilityFlowODE()


def _replica(provider, train_model, device: torch.device):
    """A second model instance carrying the train model's weights.

    Built from the provider rather than deep-copied so that modalities holding
    non-copyable resources (a VAE handle, a remote weight cache) stay in control
    of how a replica is produced.
    """
    replica = provider.model(device=device)
    if hasattr(train_model, "state_dict") and hasattr(replica, "load_state_dict"):
        replica.load_state_dict(train_model.state_dict())
    return replica
