"""Environments."""

import warnings

warnings.warn(
    "Environments are part of the old codebasem, they are deprecated.",
    category=DeprecationWarning,
)

from .base import Environment, EnvironmentMode, Sample
from .endpoint import EndpointEnvironment
from .epsilon import EpsilonEnvironment
from .score import ScoreEnvironment
from .velocity import VelocityEnvironment

__all__ = [
    "EndpointEnvironment",
    "Environment",
    "EnvironmentCapabilities",
    "EnvironmentMode",
    "EpsilonEnvironment",
    "PolicyBundle",
    "Sample",
    "ScoreEnvironment",
    "VelocityEnvironment",
]
