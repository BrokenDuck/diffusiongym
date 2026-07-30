"""Shared utilities for trainers."""

import torch

from diffusiongym.environments.base import Sample
from diffusiongym.types import DDBatch


def filter_valid[T: DDBatch](sample: Sample[T]) -> tuple[T, torch.Tensor, dict]:
    """Return (latents, rewards, kwargs) for valid samples only."""
    valid_idx = sample.valids.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0:
        raise ValueError("No valid samples in batch.")
    latents = type(sample.latent).collate([sample.latent[int(i)] for i in valid_idx])
    rewards = sample.rewards[valid_idx]
    kwargs = {
        k: ([v[int(i)] for i in valid_idx] if isinstance(v, list) else v[valid_idx])
        for k, v in sample.kwargs.items()
    }
    return latents, rewards, kwargs


class RunningRewardStats:
    """EMA-based running statistics for reward normalization."""

    def __init__(self, halflife_iters: float):
        beta_raw = 1 - 2 ** (-1 / halflife_iters)
        self._beta = beta_raw
        self._mean: torch.Tensor | None = None
        self._var: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, rewards: torch.Tensor) -> None:
        batch_mean = rewards.mean()
        batch_var = ((rewards - batch_mean) ** 2).mean().clamp_min(1e-6)
        if self._mean is None or self._var is None:
            self._mean = batch_mean
            self._var = batch_var
        else:
            b = self._beta
            self._mean = (1 - b) * self._mean + b * batch_mean
            self._var = ((1 - b) * self._var + b * batch_var).clamp_min(1e-6)

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._var is None:
            return rewards
        return (rewards - self._mean) / (self._var.sqrt() + 1e-8)
