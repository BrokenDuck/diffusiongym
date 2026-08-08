"""Registry system for modalities, rewards, and fine-tuning algorithms.

The registry machinery (`Registry`, `RegistryEntry`) is deliberately generic —
it maps a string id to a class plus default constructor kwargs and knows nothing
about what it holds. What it holds is the part that tracks the `core/`
interfaces, via two provider protocols:

  ModalityProvider  everything that changes when the *data type* changes —
                    geometry, base distribution, codec, and the network. This is
                    the seam between the toy 2-D tensors, SD3.5 image latents,
                    and a FlowMol graph state.
  RewardProvider    a black-box `RewardEvaluator` and, when the reward happens
                    to be differentiable, the `DifferentiableTerminalCost` that
                    Adjoint Matching additionally requires.

Nothing here assumes a dense tensor: a provider whose `geometry()` returns a
graph geometry and whose `base_sampler()` emits a graph state works unchanged.

`base_model_registry` and `reward_registry` are retained for the legacy
`BaseModel`/`Reward` interfaces used by `toy/gmm.py` and `toy/rewards.py`; new
work should use the three registries defined at the bottom of this module.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any, Protocol, overload, runtime_checkable

import torch

from diffusiongym.core.codec import DataCodec
from diffusiongym.core.model import FlowModel
from diffusiongym.core.process import BaseSampler
from diffusiongym.core.reward import DifferentiableTerminalCost, RewardEvaluator
from diffusiongym.core.schedule import AffineSchedule
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch

# ---------------------------------------------------------------------------
# What can be registered
# ---------------------------------------------------------------------------


@runtime_checkable
class ModalityProvider[StateT: DDBatch, RawT](Protocol):
    """Everything that changes when the state type changes.

    A provider owns the choice of state representation, so it is the only place
    that needs to know whether states are dense tensors or graphs. Implement
    `geometry()` and `base_sampler()` against the same `StateT` and the rest of
    the framework follows.

    `domain` is a coarse tag ("toy", "images", "molecules") used only to reject
    obviously mismatched modality/reward pairs early.
    """

    domain: str

    def geometry(self) -> LatentGeometry[StateT]:
        """Projection, Gaussian sampling, and per-sample norms for this state."""
        ...

    def schedule(self) -> AffineSchedule:
        """Interpolation schedule the pretrained model was trained under."""
        ...

    def base_sampler(self) -> BaseSampler[StateT]:
        """Gaussian base distribution, including any structure the state needs."""
        ...

    def codec(self) -> DataCodec[RawT, StateT]:
        """Latent state to user-facing sample and back."""
        ...

    def model(self, *, device: torch.device) -> FlowModel[StateT]:
        """A fresh copy of the pretrained flow model on `device`.

        Called more than once per setup — the train, rollout, and reference
        policies are separate objects — so this must return independent
        instances carrying identical weights.
        """
        ...


@runtime_checkable
class RewardProvider[StateT: DDBatch, RawT](Protocol):
    """A reward, and its differentiable form when one exists."""

    domain: str

    def reward(self) -> RewardEvaluator[RawT, StateT]:
        """Black-box reward. Every algorithm can use this."""
        ...

    def terminal_cost(self) -> DifferentiableTerminalCost[StateT] | None:
        """Differentiable terminal cost, or None if unavailable.

        Only Adjoint Matching needs this; returning None makes `make()` reject
        that pairing with a clear message instead of failing later.
        """
        ...


# ---------------------------------------------------------------------------
# Registry machinery
# ---------------------------------------------------------------------------


class RegistryEntry[T]:
    """A registered class plus the default kwargs to build it with."""

    def __init__(self, cls: type[T], kwargs: dict[str, Any] | None = None):
        self.cls = cls
        self.default_kwargs = kwargs or {}

    def instantiate(self, **override_kwargs: Any) -> T:
        """Instantiate the registered class; overrides beat default kwargs."""
        return self.cls(**{**self.default_kwargs, **override_kwargs})


class Registry[T]:
    """Generic registry mapping string identifiers to classes.

    Parameters
    ----------
    name : str
        Name of the registry, used in error messages.
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, RegistryEntry[T]] = {}

    @overload
    def register(
        self, id: str, entry_point: type[T], **default_kwargs: Any
    ) -> type[T]: ...

    @overload
    def register(
        self, id: str, entry_point: None = None, **default_kwargs: Any
    ) -> Callable[[type[T]], type[T]]: ...

    def register(
        self,
        id: str,
        entry_point: type[T] | None = None,
        **default_kwargs: Any,
    ) -> type[T] | Callable[[type[T]], type[T]]:
        """Register a class, as a decorator or by direct call.

        Parameters
        ----------
        id : str
            Unique identifier, conventionally "<domain>/<name>".
        entry_point : type[T], optional
            The class to register. If None, returns a decorator.
        **default_kwargs : Any
            Default keyword arguments for the constructor.

        Examples
        --------
        As a decorator:
        >>> @modality_registry.register("toy/gmm2d")
        ... class GMM2DModality:
        ...     domain = "toy"

        Direct call:
        >>> modality_registry.register("toy/gmm2d", GMM2DModality)

        Re-registering the same id with the same class is a no-op, so a module
        that registers on import can be imported twice.
        """

        def _insert(cls: type[T]) -> type[T]:
            existing = self._registry.get(id)
            if existing is not None and existing.cls is not cls:
                raise ValueError(
                    f"{id} is already registered in {self.name} with a different class"
                )
            self._registry[id] = RegistryEntry(cls, default_kwargs)
            return cls

        return _insert(entry_point) if entry_point is not None else _insert

    def get(self, id: str) -> RegistryEntry[T]:
        """Get an entry by id, raising KeyError listing the valid options."""
        if id not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(nothing registered)"
            raise KeyError(
                f"{id} is not registered in {self.name}. Available: {available}"
            )
        return self._registry[id]

    def list(self) -> builtins.list[str]:
        # `builtins.list` because the method name shadows the builtin in class scope.
        """Sorted list of registered ids."""
        return sorted(self._registry)

    def __contains__(self, id: str) -> bool:
        return id in self._registry


def domain_of(identifier: str) -> str | None:
    """Leading "<domain>/" segment of a registry id, if it has one."""
    return identifier.split("/")[0] if "/" in identifier else None


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------

#: Current interfaces — assembled by `diffusiongym.make`.
modality_registry: Registry[ModalityProvider] = Registry("MODALITY_REGISTRY")
reward_provider_registry: Registry[RewardProvider] = Registry(
    "REWARD_PROVIDER_REGISTRY"
)
algorithm_registry: Registry[Any] = Registry("ALGORITHM_REGISTRY")

#: Legacy interfaces (`BaseModel` / `Reward`), kept only for `toy/gmm.py` and
#: `toy/rewards.py`. Do not use for new work.
base_model_registry: Registry[Any] = Registry("BASE_MODEL_REGISTRY")
reward_registry: Registry[Any] = Registry("REWARD_REGISTRY")
