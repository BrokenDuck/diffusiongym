"""FlowEnvironment: immutable environment facade and PolicyBundle.

FlowEnvironment holds all the process/modality services but owns NO policies.
Policies are owned by the fine-tuning algorithm via PolicyBundle.

This separation prevents the environment from accumulating mutable state
that would make multi-algorithm comparisons and checkpointing error-prone.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from diffusiongym.core.codec import DataCodec
from diffusiongym.core.model import FlowModel, VelocityRegression
from diffusiongym.core.process import (
    AffineGaussianForwardProcess,
    BaseSampler,
    ForwardBatch,
)
from diffusiongym.core.reward import (
    DifferentiableTerminalCost,
    RewardBatch,
    RewardEvaluator,
)
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch

Conditioning = Mapping[str, Any]


@dataclass(frozen=True)
class FlowEnvironment[StateT: DDBatch, RawT]:
    """Immutable container for all process and modality services.

    Does not own policies. Use PolicyBundle to manage train/rollout/reference models.

    Parameters
    ----------
    geometry:
        Latent geometry (projection, norms, Gaussian sampling).
    base_sampler:
        Gaussian base distribution sampler.
    forward_process:
        Affine Gaussian forward process (interpolation + target velocity).
    regression:
        Velocity regression (predict + error computation).
    codec:
        Encode/decode between raw samples and latent states.
    reward:
        Black-box reward evaluator.
    terminal_cost:
        Differentiable terminal cost for Adjoint Matching. None if not needed.
    """

    geometry: LatentGeometry[StateT]
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
    ) -> ForwardBatch[StateT]:
        """Sample a training batch by interpolating between base and data."""
        return self.forward_process.make_batch(
            x_data,
            conditioning=conditioning,
            t=t,
        )

    def predict_velocity(
        self,
        model: FlowModel[StateT],
        *,
        x_t: StateT,
        t: Tensor,
        conditioning: Conditioning,
    ) -> StateT:
        """Run the model and convert its output to path velocity."""
        return self.regression.predict(model, x_t=x_t, t=t, conditioning=conditioning)

    def velocity_error(
        self,
        predicted: StateT,
        target: StateT,
    ) -> Tensor:
        """Per-sample mean squared error between predicted and target velocity, shape (n,)."""
        return self.regression.per_example_error(predicted, target)

    def evaluate_terminal(
        self,
        latent: StateT,
        *,
        conditioning: Conditioning,
    ) -> tuple[RawT, RewardBatch]:
        """Decode the terminal latent state and evaluate the reward."""
        sample = self.codec.decode(latent, conditioning=conditioning)
        reward = self.reward(sample=sample, latent=latent, conditioning=conditioning)
        return sample, reward


@dataclass
class PolicyBundle[StateT]:
    """Policies owned by a fine-tuning algorithm.

    train:
        Receives gradients; updated each iteration.
    rollout:
        Generates online samples. May be a frozen snapshot of train,
        an EMA copy, or identical to train (for on-policy methods).
    reference:
        Fixed pretrained anchor. Used for KL penalties and adjoint dynamics.
        None for algorithms that don't need a reference (ORW-CFM, DiffusionNFT).
    """

    train: FlowModel[StateT]
    rollout: FlowModel[StateT]
    reference: FlowModel[StateT] | None = None
