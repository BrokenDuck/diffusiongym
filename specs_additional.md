The clean boundary is:

* **One constrained affine-Gaussian flow environment**
* **One canonical training field: path velocity**
* **Model adapters for velocity, endpoint, or noise prediction**
* **Two samplers: Euler ODE and Euler–Maruyama SDE**
* **Algorithm-owned policy bundles**
* **Fine-tuning algorithms built from shared forward-regression, rollout, and transition primitives**

Do not create separate “velocity trainer,” “endpoint trainer,” and “noise trainer.” Those are model parameterizations, not distinct training frameworks.

## 1. Core types

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Generic, Literal, Protocol, TypeVar

import torch
from torch import Tensor, nn


StateT = TypeVar("StateT")
RawT = TypeVar("RawT")
ExperienceT = TypeVar("ExperienceT")

Conditioning = Mapping[str, Any]
NormReduction = Literal["mean", "sum"]
```

---

# 2. Latent state interface

The model state may be:

* A tensor for toy models and SD3.5
* A structured graph batch for FlowMol
* A dataclass containing coordinates, atom features, bonds, masks, and metadata

Do not require the state itself to implement arithmetic operators. Put all state arithmetic and geometry in one adapter.

```python
class LatentSpace(Protocol[StateT]):
    """Linear operations and geometry for a continuous latent state.

    Implementations:
      - TensorLatentSpace
      - MolecularGraphLatentSpace

    The state may be constrained to a known linear subspace, such as
    zero-center-of-mass molecular coordinates.
    """

    def batch_size(self, x: StateT) -> int:
        ...

    def device(self, x: StateT) -> torch.device:
        ...

    def to(self, x: StateT, device: torch.device | str) -> StateT:
        ...

    def detach(self, x: StateT) -> StateT:
        ...

    def clone(self, x: StateT) -> StateT:
        ...

    def add(self, x: StateT, y: StateT) -> StateT:
        ...

    def subtract(self, x: StateT, y: StateT) -> StateT:
        ...

    def scale(self, x: StateT, coefficient: Tensor | float) -> StateT:
        """Multiply each batch element by a broadcastable scalar."""
        ...

    def linear_combination(
        self,
        terms: Sequence[tuple[Tensor | float, StateT]],
    ) -> StateT:
        ...

    def project(self, x: StateT) -> StateT:
        """Enforce masks and known linear constraints."""
        ...

    def randn_like(
        self,
        x: StateT,
        *,
        generator: torch.Generator | None = None,
    ) -> StateT:
        """Gaussian noise with the same structure, masks and constraints."""
        ...

    def squared_norm(
        self,
        x: StateT,
        *,
        reduction: NormReduction,
    ) -> Tensor:
        """Return one scalar per batch element."""
        ...

    def active_dimensions(self, x: StateT) -> Tensor:
        """Number of active scalar dimensions per batch element.

        Needed for normalized Gaussian log densities.
        """
        ...

    def concatenate(self, xs: Sequence[StateT]) -> StateT:
        ...
```

For molecular graphs, `scale()` and `linear_combination()` act only on continuous modeled fields. Graph indices, masks, node counts, and conditioning metadata remain unchanged.

## Gaussian base sampler

```python
class BaseSampler(Protocol[StateT]):
    """Standard Gaussian base distribution in the latent state space."""

    def sample(
        self,
        n: int,
        *,
        conditioning: Conditioning,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[StateT, Conditioning]:
        """Sample a new base batch for generation."""
        ...

    def sample_like(
        self,
        x_data: StateT,
        *,
        generator: torch.Generator | None = None,
    ) -> StateT:
        """Sample base noise matching a training batch."""
        ...
```

This lets:

* SD3.5 choose the latent tensor shape.
* FlowMol preserve node counts and graph masks.
* Toy models use a fixed vector dimension.

The V1 contract should require this sampler to represent a standard Gaussian, possibly projected or masked.

---

# 3. Affine flow schedule and forward process

Use one fixed process family:

[
x_t = a(t)x_{\mathrm{base}} + b(t)x_{\mathrm{data}}.
]

```python
class AffineSchedule(ABC):
    """Scalar affine interpolation schedule.

    Canonical direction:
      t = 0: Gaussian base
      t = 1: data
    """

    @abstractmethod
    def a(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def b(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def da_dt(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def db_dt(self, t: Tensor) -> Tensor:
        ...

    def validate(self) -> None:
        """Check endpoint conditions and finite derivatives."""
        ...
```

Default implementation:

```python
class RectifiedFlowSchedule(AffineSchedule):
    def a(self, t: Tensor) -> Tensor:
        return 1.0 - t

    def b(self, t: Tensor) -> Tensor:
        return t

    def da_dt(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)

    def db_dt(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)
```

## Forward batch

```python
@dataclass(frozen=True)
class ForwardBatch(Generic[StateT]):
    x_data: StateT
    x_base: StateT
    x_t: StateT
    target_velocity: StateT
    t: Tensor
    conditioning: Conditioning
```

## Forward process

This can be a concrete class rather than an abstraction because you deliberately support only affine-Gaussian flows.

```python
class AffineGaussianForwardProcess(Generic[StateT]):
    def __init__(
        self,
        *,
        space: LatentSpace[StateT],
        base_sampler: BaseSampler[StateT],
        schedule: AffineSchedule,
    ) -> None:
        self.space = space
        self.base_sampler = base_sampler
        self.schedule = schedule

    def sample_time(
        self,
        n: int,
        *,
        device: torch.device,
        t_min: float = 1e-3,
        t_max: float = 1.0 - 1e-3,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return (
            torch.rand(n, device=device, generator=generator)
            * (t_max - t_min)
            + t_min
        )

    def make_batch(
        self,
        x_data: StateT,
        *,
        conditioning: Conditioning,
        t: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ForwardBatch[StateT]:
        n = self.space.batch_size(x_data)

        if t is None:
            t = self.sample_time(
                n,
                device=self.space.device(x_data),
                generator=generator,
            )

        x_base = self.base_sampler.sample_like(
            x_data,
            generator=generator,
        )

        x_t = self.space.linear_combination(
            [
                (self.schedule.a(t), x_base),
                (self.schedule.b(t), x_data),
            ]
        )

        target_velocity = self.space.linear_combination(
            [
                (self.schedule.da_dt(t), x_base),
                (self.schedule.db_dt(t), x_data),
            ]
        )

        return ForwardBatch(
            x_data=x_data,
            x_base=x_base,
            x_t=self.space.project(x_t),
            target_velocity=self.space.project(target_velocity),
            t=t,
            conditioning=conditioning,
        )
```

---

# 4. Model and prediction parameterization

## Model protocol

```python
class PredictionKind(Enum):
    VELOCITY = auto()
    ENDPOINT = auto()
    NOISE = auto()
```

```python
class FlowModel(Protocol[StateT]):
    """Minimal model contract.

    Concrete implementations may wrap:
      - an nn.Module
      - SD3.5 transformer and conditioning stack
      - FlowMol
      - a toy MLP
    """

    prediction_kind: PredictionKind

    @property
    def device(self) -> torch.device:
        ...

    def __call__(
        self,
        x_t: StateT,
        t: Tensor,
        *,
        conditioning: Conditioning,
    ) -> StateT:
        ...
```

In practice, use an adapter around the concrete model rather than modifying third-party model classes.

```python
class TorchFlowModelAdapter(nn.Module, Generic[StateT], ABC):
    prediction_kind: PredictionKind

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def forward(
        self,
        x_t: StateT,
        t: Tensor,
        *,
        conditioning: Conditioning,
    ) -> StateT:
        ...
```

## Prediction conversion

All fine-tuning algorithms should request **velocity**, regardless of the native model output.

```python
class PredictionConverter(Generic[StateT]):
    """Convert native affine-flow predictions to path velocity."""

    def __init__(
        self,
        *,
        space: LatentSpace[StateT],
        schedule: AffineSchedule,
        denominator_epsilon: float = 1e-6,
    ) -> None:
        self.space = space
        self.schedule = schedule
        self.denominator_epsilon = denominator_epsilon

    def to_velocity(
        self,
        *,
        prediction: StateT,
        kind: PredictionKind,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        match kind:
            case PredictionKind.VELOCITY:
                return prediction

            case PredictionKind.ENDPOINT:
                return self._endpoint_to_velocity(
                    endpoint=prediction,
                    x_t=x_t,
                    t=t,
                )

            case PredictionKind.NOISE:
                return self._noise_to_velocity(
                    noise=prediction,
                    x_t=x_t,
                    t=t,
                )

            case _:
                raise ValueError(f"Unsupported prediction kind: {kind}")

    def _endpoint_to_velocity(
        self,
        *,
        endpoint: StateT,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        a = self.schedule.a(t)
        b = self.schedule.b(t)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)

        self._require_nonzero(a, name="a(t)")

        inferred_noise = self.space.scale(
            self.space.subtract(
                x_t,
                self.space.scale(endpoint, b),
            ),
            1.0 / a,
        )

        return self.space.linear_combination(
            [
                (da, inferred_noise),
                (db, endpoint),
            ]
        )

    def _noise_to_velocity(
        self,
        *,
        noise: StateT,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        a = self.schedule.a(t)
        b = self.schedule.b(t)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)

        self._require_nonzero(b, name="b(t)")

        inferred_endpoint = self.space.scale(
            self.space.subtract(
                x_t,
                self.space.scale(noise, a),
            ),
            1.0 / b,
        )

        return self.space.linear_combination(
            [
                (da, noise),
                (db, inferred_endpoint),
            ]
        )

    def _require_nonzero(self, value: Tensor, *, name: str) -> None:
        if torch.any(value.abs() <= self.denominator_epsilon):
            raise ValueError(
                f"Cannot convert prediction because {name} is too close to zero. "
                "Use interior training or sampling times."
            )
```

## Shared regression primitive

```python
class VelocityRegression(Generic[StateT]):
    def __init__(
        self,
        *,
        space: LatentSpace[StateT],
        converter: PredictionConverter[StateT],
    ) -> None:
        self.space = space
        self.converter = converter

    def predict(
        self,
        model: FlowModel[StateT],
        *,
        x_t: StateT,
        t: Tensor,
        conditioning: Conditioning,
    ) -> StateT:
        native_prediction = model(
            x_t,
            t,
            conditioning=conditioning,
        )

        return self.converter.to_velocity(
            prediction=native_prediction,
            kind=model.prediction_kind,
            x_t=x_t,
            t=t,
        )

    def per_example_error(
        self,
        predicted_velocity: StateT,
        target_velocity: StateT,
    ) -> Tensor:
        residual = self.space.subtract(
            predicted_velocity,
            target_velocity,
        )
        return self.space.squared_norm(
            residual,
            reduction="mean",
        )
```

This replaces three separate trainers.

---

# 5. Codec and reward interfaces

## Codec

```python
class DataCodec(Protocol[RawT, StateT]):
    """Convert between user-facing samples and model latent states."""

    def encode(
        self,
        raw: RawT,
        *,
        conditioning: Conditioning,
    ) -> StateT:
        ...

    def decode(
        self,
        latent: StateT,
        *,
        conditioning: Conditioning,
    ) -> RawT:
        ...
```

Examples:

* Toy data: identity codec.
* SD3.5: VAE latent decoder and encoder.
* FlowMol: graph latent to molecule representation.

## Reward output

```python
@dataclass(frozen=True)
class RewardBatch:
    rewards: Tensor
    valid: Tensor | None = None
    metadata: Mapping[str, Any] | None = None
```

```python
class RewardEvaluator(Protocol[RawT, StateT]):
    def __call__(
        self,
        *,
        sample: RawT,
        latent: StateT,
        conditioning: Conditioning,
    ) -> RewardBatch:
        ...
```

## Differentiable terminal cost

Only Adjoint Matching requires this.

```python
class DifferentiableTerminalCost(Protocol[StateT]):
    def __call__(
        self,
        terminal_latent: StateT,
        *,
        conditioning: Conditioning,
    ) -> Tensor:
        """One differentiable scalar cost per batch element."""
        ...
```

Keep `RewardEvaluator` and `DifferentiableTerminalCost` separate. A black-box reward is not automatically a valid Adjoint Matching terminal cost.

---

# 6. Sampling dynamics

Separate the learned path velocity from the chosen sampling process.

```python
@dataclass(frozen=True)
class DynamicsCoefficients(Generic[StateT]):
    drift: StateT
    diffusion: Tensor
```

The diffusion is a scalar per sample and time. The latent-space implementation handles broadcasting.

```python
class FlowDynamics(ABC, Generic[StateT]):
    """Convert path velocity into sampling dynamics."""

    stochastic: bool
    memoryless: bool

    @abstractmethod
    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]:
        ...

    def validate(self) -> None:
        ...
```

## Deterministic ODE

```python
class ProbabilityFlowODE(FlowDynamics[StateT]):
    stochastic = False
    memoryless = False

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
```

## Scalar diffusion schedule

```python
class ScalarDiffusionSchedule(ABC):
    @abstractmethod
    def value(self, t: Tensor) -> Tensor:
        ...
```

## Marginal-preserving SDE

The concrete implementation contains the affine-flow ODE-to-SDE identity.

```python
class MarginalPreservingSDE(FlowDynamics[StateT], ABC):
    stochastic = True
    memoryless = False

    def __init__(
        self,
        *,
        space: LatentSpace[StateT],
        affine_schedule: AffineSchedule,
        diffusion_schedule: ScalarDiffusionSchedule,
    ) -> None:
        self.space = space
        self.affine_schedule = affine_schedule
        self.diffusion_schedule = diffusion_schedule

    @abstractmethod
    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]:
        """Return the marginal-preserving SDE drift and scalar diffusion."""
        ...

    @abstractmethod
    def relative_control(
        self,
        *,
        current_drift: StateT,
        reference_drift: StateT,
        t: Tensor,
    ) -> StateT:
        """u satisfying b_current = b_reference + g(t) u."""
        ...
```

## Memoryless dynamics

Make this an explicit implementation, not an inferred property.

```python
class MemorylessFlowSDE(MarginalPreservingSDE[StateT]):
    memoryless = True

    def coefficients(
        self,
        *,
        x: StateT,
        t: Tensor,
        velocity: StateT,
    ) -> DynamicsCoefficients[StateT]:
        ...

    def relative_control(
        self,
        *,
        current_drift: StateT,
        reference_drift: StateT,
        t: Tensor,
    ) -> StateT:
        ...
```

Adjoint Matching requires `MemorylessFlowSDE`. Flow-GRPO can use another `MarginalPreservingSDE`.

---

# 7. Markov transition interface

Flow-GRPO requires transition likelihoods. The transition kernel used to calculate the likelihood must be the same object used to generate the trajectory.

```python
class MarkovKernel(Protocol[StateT]):
    def rsample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> StateT:
        ...

    def log_prob(self, value: StateT) -> Tensor:
        """One log probability per batch element."""
        ...

    def kl_divergence(
        self,
        other: MarkovKernel[StateT],
    ) -> Tensor:
        ...
```

```python
class EulerGaussianKernelFactory(Protocol[StateT]):
    def build(
        self,
        *,
        x: StateT,
        t: Tensor,
        dt: Tensor,
        drift: StateT,
        diffusion: Tensor,
    ) -> MarkovKernel[StateT]:
        ...
```

The Gaussian kernel implementation should use:

* `LatentSpace.squared_norm(..., reduction="sum")`
* `LatentSpace.active_dimensions(...)`
* masks and projections appropriate to the modality

---

# 8. Rollout and sampler interfaces

## Storage specification

```python
@dataclass(frozen=True)
class RolloutStorage:
    states: bool = False
    noises: bool = False
    drifts: bool = False
    predictions: bool = False
    log_probs: bool = False
```

## Rollout request

```python
@dataclass(frozen=True)
class RolloutRequest:
    time_grid: Tensor
    storage: RolloutStorage
    requires_grad: bool = False
    evaluate_reward: bool = True
```

## Step and rollout data

```python
@dataclass
class RolloutStep(Generic[StateT]):
    x: StateT
    x_next: StateT
    t: Tensor
    dt: Tensor
    drift: StateT | None = None
    diffusion: Tensor | None = None
    noise: StateT | None = None
    log_prob: Tensor | None = None
```

```python
@dataclass
class Rollout(Generic[StateT, RawT]):
    terminal_latent: StateT
    terminal_sample: RawT | None
    reward: RewardBatch | None
    steps: list[RolloutStep[StateT]]
    conditioning: Conditioning
```

## Sampler protocol

```python
class Sampler(Protocol[StateT, RawT]):
    def rollout(
        self,
        *,
        environment: FlowEnvironment[StateT, RawT],
        model: FlowModel[StateT],
        dynamics: FlowDynamics[StateT],
        n: int,
        conditioning: Conditioning,
        request: RolloutRequest,
        generator: torch.Generator | None = None,
    ) -> Rollout[StateT, RawT]:
        ...
```

## ODE Euler sampler

```python
class EulerODESampler(Generic[StateT, RawT]):
    def rollout(
        self,
        *,
        environment: FlowEnvironment[StateT, RawT],
        model: FlowModel[StateT],
        dynamics: FlowDynamics[StateT],
        n: int,
        conditioning: Conditioning,
        request: RolloutRequest,
        generator: torch.Generator | None = None,
    ) -> Rollout[StateT, RawT]:
        if dynamics.stochastic:
            raise ValueError("EulerODESampler requires deterministic dynamics.")
        ...
```

## Euler–Maruyama sampler

```python
class EulerMaruyamaSampler(Generic[StateT, RawT]):
    def __init__(
        self,
        kernel_factory: EulerGaussianKernelFactory[StateT],
    ) -> None:
        self.kernel_factory = kernel_factory

    def rollout(
        self,
        *,
        environment: FlowEnvironment[StateT, RawT],
        model: FlowModel[StateT],
        dynamics: FlowDynamics[StateT],
        n: int,
        conditioning: Conditioning,
        request: RolloutRequest,
        generator: torch.Generator | None = None,
    ) -> Rollout[StateT, RawT]:
        if not dynamics.stochastic:
            raise ValueError(
                "EulerMaruyamaSampler requires stochastic dynamics."
            )

        # Each step:
        #
        # velocity = environment.predict_velocity(...)
        # coeffs = dynamics.coefficients(...)
        # kernel = kernel_factory.build(...)
        # x_next = kernel.rsample(...)
        #
        # Store kernel.log_prob(x_next) when requested.
        ...
```

---

# 9. Environment facade

The environment contains immutable process and modality services. It does not own current, old, or reference policies.

```python
@dataclass(frozen=True)
class FlowEnvironment(Generic[StateT, RawT]):
    space: LatentSpace[StateT]
    base_sampler: BaseSampler[StateT]
    forward_process: AffineGaussianForwardProcess[StateT]
    regression: VelocityRegression[StateT]
    codec: DataCodec[RawT, StateT]
    reward: RewardEvaluator[RawT, StateT]
    terminal_cost: DifferentiableTerminalCost[StateT] | None = None

    def make_forward_batch(
        self,
        x_data: StateT,
        *,
        conditioning: Conditioning,
        t: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> ForwardBatch[StateT]:
        return self.forward_process.make_batch(
            x_data,
            conditioning=conditioning,
            t=t,
            generator=generator,
        )

    def predict_velocity(
        self,
        model: FlowModel[StateT],
        *,
        x_t: StateT,
        t: Tensor,
        conditioning: Conditioning,
    ) -> StateT:
        return self.regression.predict(
            model,
            x_t=x_t,
            t=t,
            conditioning=conditioning,
        )

    def velocity_error(
        self,
        predicted: StateT,
        target: StateT,
    ) -> Tensor:
        return self.regression.per_example_error(
            predicted,
            target,
        )

    def evaluate_terminal(
        self,
        latent: StateT,
        *,
        conditioning: Conditioning,
    ) -> tuple[RawT, RewardBatch]:
        sample = self.codec.decode(
            latent,
            conditioning=conditioning,
        )
        reward = self.reward(
            sample=sample,
            latent=latent,
            conditioning=conditioning,
        )
        return sample, reward
```

---

# 10. Algorithm-owned policy bundle

```python
@dataclass
class PolicyBundle(Generic[StateT]):
    """Policies used by a fine-tuning algorithm.

    train:
        Receives gradients.

    rollout:
        Generates online samples. May be a frozen copy or EMA of train.

    reference:
        Fixed pretrained policy for KL or control regularization.
    """

    train: FlowModel[StateT]
    rollout: FlowModel[StateT]
    reference: FlowModel[StateT] | None = None
```

The fine-tuning algorithm owns cloning, freezing, synchronization, and EMA updates.

---

# 11. Fine-tuning algorithm interface

All four algorithms naturally divide into:

1. Collect online experience.
2. Update the training policy.
3. Optionally synchronize the rollout policy.

## Capability requirements

```python
@dataclass(frozen=True)
class FineTuningRequirements:
    needs_reference_policy: bool = False
    needs_stochastic_rollout: bool = False
    needs_tractable_transitions: bool = False
    needs_memoryless_dynamics: bool = False
    needs_differentiable_terminal_cost: bool = False
    rollout_storage: RolloutStorage = RolloutStorage()
```

## Fine-tuning context

```python
@dataclass
class FineTuningContext(Generic[StateT, RawT]):
    environment: FlowEnvironment[StateT, RawT]
    policies: PolicyBundle[StateT]
    optimizer: torch.optim.Optimizer
    ode_sampler: EulerODESampler[StateT, RawT]
    sde_sampler: EulerMaruyamaSampler[StateT, RawT]
```

## Base algorithm

```python
class FineTuningAlgorithm(
    ABC,
    Generic[StateT, RawT, ExperienceT],
):
    @property
    @abstractmethod
    def requirements(self) -> FineTuningRequirements:
        ...

    def validate(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
    ) -> None:
        req = self.requirements

        if req.needs_reference_policy:
            if context.policies.reference is None:
                raise ValueError("A reference policy is required.")

        if req.needs_stochastic_rollout and not dynamics.stochastic:
            raise ValueError("This algorithm requires stochastic dynamics.")

        if req.needs_memoryless_dynamics and not dynamics.memoryless:
            raise ValueError("This algorithm requires memoryless dynamics.")

        if req.needs_differentiable_terminal_cost:
            if context.environment.terminal_cost is None:
                raise ValueError(
                    "A differentiable terminal cost is required."
                )

    @abstractmethod
    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        conditioning: Conditioning,
        generator: torch.Generator | None = None,
    ) -> ExperienceT:
        ...

    @abstractmethod
    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: ExperienceT,
    ) -> Mapping[str, float]:
        ...

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Optional hard-copy or EMA update."""
        return None
```

---

# 12. Experience types

## Endpoint experience

Used by DiffusionNFT and ORW-CFM-W2.

```python
@dataclass
class EndpointExperience(Generic[StateT]):
    latent: StateT
    rewards: Tensor
    valid: Tensor | None
    conditioning: Conditioning
```

## Policy-gradient trajectory experience

Used by Flow-GRPO.

```python
@dataclass
class TrajectoryExperience(Generic[StateT]):
    rollout: Rollout[StateT, Any]
    advantages: Tensor
```

## Adjoint experience

```python
@dataclass
class AdjointExperience(Generic[StateT]):
    rollout: Rollout[StateT, Any]
    adjoint_targets: list[StateT]
```

---

# 13. Algorithm interfaces

## DiffusionNFT

```python
class DiffusionNFT(
    FineTuningAlgorithm[
        StateT,
        RawT,
        EndpointExperience[StateT],
    ],
):
    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            rollout_storage=RolloutStorage(states=False),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        conditioning: Conditioning,
        generator: torch.Generator | None = None,
    ) -> EndpointExperience[StateT]:
        ...

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: EndpointExperience[StateT],
    ) -> Mapping[str, float]:
        # batch = env.make_forward_batch(experience.latent)
        # v_old = env.predict_velocity(policies.rollout, ...)
        # v_train = env.predict_velocity(policies.train, ...)
        #
        # v_pos = (1-beta) v_old + beta v_train
        # v_neg = (1+beta) v_old - beta v_train
        #
        # loss = r * error(v_pos, target)
        #      + (1-r) * error(v_neg, target)
        ...
```

## ORW-CFM-W2

```python
class ORWCFMW2(
    FineTuningAlgorithm[
        StateT,
        RawT,
        EndpointExperience[StateT],
    ],
):
    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=True,
            rollout_storage=RolloutStorage(states=False),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        conditioning: Conditioning,
        generator: torch.Generator | None = None,
    ) -> EndpointExperience[StateT]:
        ...

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: EndpointExperience[StateT],
    ) -> Mapping[str, float]:
        # Reward-weighted velocity regression plus W2 regularization.
        ...
```

## Flow-GRPO

```python
class FlowGRPO(
    FineTuningAlgorithm[
        StateT,
        RawT,
        TrajectoryExperience[StateT],
    ],
):
    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=True,
            needs_stochastic_rollout=True,
            needs_tractable_transitions=True,
            rollout_storage=RolloutStorage(
                states=True,
                log_probs=True,
            ),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        conditioning: Conditioning,
        generator: torch.Generator | None = None,
    ) -> TrajectoryExperience[StateT]:
        ...

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: TrajectoryExperience[StateT],
    ) -> Mapping[str, float]:
        # Rebuild the same Euler Gaussian kernels under:
        #   train policy
        #   rollout/old policy
        #   reference policy
        #
        # Compute PPO/GRPO ratios and KL terms.
        ...
```

## Adjoint Matching

```python
class AdjointMatching(
    FineTuningAlgorithm[
        StateT,
        RawT,
        AdjointExperience[StateT],
    ],
):
    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=True,
            needs_stochastic_rollout=True,
            needs_memoryless_dynamics=True,
            needs_differentiable_terminal_cost=True,
            rollout_storage=RolloutStorage(
                states=True,
                noises=True,
                drifts=True,
            ),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        conditioning: Conditioning,
        generator: torch.Generator | None = None,
    ) -> AdjointExperience[StateT]:
        # Run memoryless SDE.
        # Compute terminal adjoint from differentiable terminal cost.
        # Integrate adjoint targets backward.
        ...

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: AdjointExperience[StateT],
    ) -> Mapping[str, float]:
        # Regress current relative control or drift correction
        # against the computed adjoint targets.
        ...
```

---

# 14. Concrete implementations required

For your initial experiments, implement only these classes.

## Shared framework

```text
TensorLatentSpace
MolecularGraphLatentSpace

TensorGaussianBaseSampler
FlowMolGaussianBaseSampler

RectifiedFlowSchedule
OptionalGeneralAffineSchedule

AffineGaussianForwardProcess
PredictionConverter
VelocityRegression

IdentityCodec
StableDiffusionVAECodec
FlowMolCodec

ProbabilityFlowODE
RectifiedFlowMarginalPreservingSDE
MemorylessFlowSDE

TensorGaussianEulerKernel
GraphGaussianEulerKernel

EulerODESampler
EulerMaruyamaSampler

FlowEnvironment
PolicyBundle
```

## Model adapters

```text
ToyFlowModelAdapter
StableDiffusion35Adapter
FlowMolGaussianAdapter
```

Each adapter only needs to define:

```python
prediction_kind
device
forward(x_t, t, conditioning)
```

plus modality-specific conditioning preparation outside the common mathematical layer.

## Fine-tuning algorithms

```text
RewardWeightedFlowMatching
RejectionFineTuning
DiffusionNFT
ORWCFMW2
FlowGRPO
AdjointMatching
ACTFLOWFineTuner
```

Implement reward-weighted flow matching and rejection fine-tuning first. They provide simple end-to-end tests for every layer before adding the more complicated algorithms.

---

# 15. Deliberately unsupported features

The interfaces should reject rather than partially support:

* Non-Gaussian base distributions
* Non-affine conditional interpolants
* State-dependent diffusion
* Matrix-valued diffusion
* Arbitrary manifolds
* Discrete diffusion
* Adaptive SDE solvers
* Transition likelihoods for non-Euler samplers
* General score-based diffusion
* Models whose state cannot support linear combinations
* Adjoint Matching with a black-box terminal reward

This architecture gives you a controlled system rather than a universal library. New fine-tuning algorithms can usually be added by implementing only `collect()` and `update()`, while the image, molecular, and toy-specific behavior remains confined to the latent-space, codec, base-sampler, and model-adapter implementations.

