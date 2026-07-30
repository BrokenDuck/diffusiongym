"""Base reward classes and interfaces for diffusiongym."""

from abc import ABC, abstractmethod

import torch

from diffusiongym.types import DDMixin


class Reward[D: DDMixin](ABC):
    """Abstract base class for all rewards."""

    @abstractmethod
    def __call__(self, sample: D, latent: D, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward and validity for the given input x."""


class DummyReward[D: DDMixin](Reward[D]):
    """Dummy reward that always returns zero."""

    def __call__(self, sample: D, latent: D, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(len(sample), device=sample.device),
            torch.ones(len(sample), device=sample.device),
        )
