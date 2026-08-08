"""Registered providers for the 2-D four-Gaussian-mixture toy problem.

These are the reference implementations of `ModalityProvider` and
`RewardProvider`: the smallest complete answer to "what does a new modality have
to supply". An SD3.5 provider swaps `geometry` for a latent-tensor geometry and
`codec` for the VAE; a FlowMol provider swaps `geometry` for a graph geometry
and `base_sampler` for one that draws molecule sizes — nothing else changes.

Importing this module registers everything it defines.
"""

from __future__ import annotations

from pathlib import Path

import torch

from diffusiongym.core.codec import IdentityCodec
from diffusiongym.core.schedule import RectifiedFlowSchedule
from diffusiongym.core.space import TensorGeometry
from diffusiongym.registry import modality_registry, reward_provider_registry
from diffusiongym.toy.gmm2d import (
    BimodalDifferentiableCost,
    BimodalReward,
    BoxReward,
    GMMBaseSampler,
    GMMFlowModel,
    LinearDifferentiableCost,
    LinearReward,
    QuadraticDifferentiableCost,
    QuadraticReward,
    RingDifferentiableCost,
    RingReward,
    VelocityMLP,
    pretrain_velocity_model,
)


@modality_registry.register("toy/gmm2d")
class GMM2DModality:
    """2-D four-Gaussian mixture on dense `DDTensor` states.

    Parameters
    ----------
    checkpoint:
        Path to a saved `VelocityMLP` state dict. Loaded if it exists.
    pretrain_steps:
        Flow-matching steps to run when no checkpoint is available. Zero leaves
        the model at its (zero-output-head) initialization, which is enough to
        exercise wiring but produces no meaningful samples.
    width, depth:
        `VelocityMLP` size.
    """

    domain = "toy"

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        pretrain_steps: int = 0,
        width: int = 128,
        depth: int = 3,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.pretrain_steps = pretrain_steps
        self.width = width
        self.depth = depth
        self._weights: dict | None = None

    def geometry(self) -> TensorGeometry:
        return TensorGeometry()

    def schedule(self) -> RectifiedFlowSchedule:
        return RectifiedFlowSchedule()

    def base_sampler(self) -> GMMBaseSampler:
        return GMMBaseSampler()

    def codec(self) -> IdentityCodec:
        return IdentityCodec()

    def model(self, *, device: torch.device) -> GMMFlowModel:
        """A fresh model carrying this provider's weights.

        The weights are resolved once and cached, so the train, rollout, and
        reference policies start identical — pretraining per call would leave
        them silently different.
        """
        network = VelocityMLP(width=self.width, depth=self.depth).to(device)
        network.load_state_dict(self._resolve_weights(device))
        return GMMFlowModel(network, device)

    def _resolve_weights(self, device: torch.device) -> dict:
        if self._weights is not None:
            return self._weights
        network = VelocityMLP(width=self.width, depth=self.depth).to(device)
        if self.checkpoint is not None and self.checkpoint.exists():
            network.load_state_dict(torch.load(self.checkpoint, map_location=device))
        elif self.pretrain_steps > 0:
            pretrain_velocity_model(
                GMMFlowModel(network, device),
                steps=self.pretrain_steps,
                device=device,
                verbose=False,
            )
        self._weights = network.state_dict()
        return self._weights


class _RewardProvider:
    """Pairs a reward with its differentiable cost, if it has one."""

    domain = "toy"
    reward_cls: type
    cost_cls: type | None = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def reward(self):
        return self.reward_cls(**self.kwargs)

    def terminal_cost(self):
        return None if self.cost_cls is None else self.cost_cls(**self.kwargs)


@reward_provider_registry.register("toy/linear")
class LinearRewardProvider(_RewardProvider):
    """r(x) = c.x — monotone and aligned with the mode structure."""

    reward_cls = LinearReward
    cost_cls = LinearDifferentiableCost


@reward_provider_registry.register("toy/quadratic")
class QuadraticRewardProvider(_RewardProvider):
    """r(x) = -||x - y||^2 / (2 tau^2)."""

    reward_cls = QuadraticReward
    cost_cls = QuadraticDifferentiableCost


@reward_provider_registry.register("toy/bimodal")
class BimodalRewardProvider(_RewardProvider):
    """Two bumps of unequal height on opposite modes — non-monotone."""

    reward_cls = BimodalReward
    cost_cls = BimodalDifferentiableCost


@reward_provider_registry.register("toy/ring")
class RingRewardProvider(_RewardProvider):
    """A preferred radius — within-mode geometry rather than reweighting."""

    reward_cls = RingReward
    cost_cls = RingDifferentiableCost


@reward_provider_registry.register("toy/box")
class BoxRewardProvider(_RewardProvider):
    """Indicator reward. No differentiable form, so Adjoint Matching is refused.

    Note it is also a poor test at moderate lambda: the tilted target is very
    close to the base distribution and the reward is almost never observed.
    """

    reward_cls = BoxReward
    cost_cls = None
