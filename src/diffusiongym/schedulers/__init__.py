"""Schedulers for flow matching and diffusion models."""

import warnings

warnings.warn(
    "Environments are part of the old codebasem, they are deprecated.",
    category=DeprecationWarning,
)

from .base import MemorylessNoiseSchedule, NoiseSchedule, Scheduler
from .noise_schedules import ConstantNoiseSchedule
from .schedulers import CosineScheduler, DiffusionScheduler, OptimalTransportScheduler

__all__ = [
    "ConstantNoiseSchedule",
    "CosineScheduler",
    "DiffusionScheduler",
    "MemorylessNoiseSchedule",
    "NoiseSchedule",
    "OptimalTransportScheduler",
    "Scheduler",
]
