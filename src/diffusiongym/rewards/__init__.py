"""Reward module package for diffusiongym."""

from diffusiongym.rewards.base import DummyReward, Reward
from diffusiongym.rewards.one_dim import BinaryReward, GaussianReward

__all__ = [
    "BinaryReward",
    "DummyReward",
    "GaussianReward",
    "Reward",
]
