"""Reward module package for diffusiongym."""

import warnings

warnings.warn(
    "Environments are part of the old codebasem, they are deprecated.",
    category=DeprecationWarning,
)

from .base import DummyReward, Reward

__all__ = [
    "DummyReward",
    "Reward",
]
