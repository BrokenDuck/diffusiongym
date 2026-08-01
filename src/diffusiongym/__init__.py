"""Diffusion Gym package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("diffusiongym")
except PackageNotFoundError:
    __version__ = "0.0.0"

from diffusiongym.base_models import BaseModel
from diffusiongym.core import (
    AffineFlowMarginalPreservingSDE,
    AffineGaussianForwardProcess,
    AffineSchedule,
    BaseSampler,
    ConstantDiffusionSchedule,
    DataCodec,
    DefaultEulerGaussianKernelFactory,
    DifferentiableTerminalCost,
    DynamicsCoefficients,
    EulerGaussianKernelFactory,
    EulerMaruyamaSampler,
    EulerODESampler,
    FlowDynamics,
    FlowEnvironment,
    FlowModel,
    ForwardBatch,
    GaussianMarkovKernel,
    IdentityCodec,
    LatentGeometry,
    MarkovKernel,
    MemorylessDiffusionSchedule,
    MemorylessFlowSDE,
    PredictionConverter,
    PredictionKind,
    PolicyBundle,
    ProbabilityFlowODE,
    RectifiedFlowSchedule,
    RewardBatch,
    RewardEvaluator,
    Rollout,
    RolloutRequest,
    RolloutStep,
    RolloutStorage,
    ScalarDiffusionSchedule,
    TensorGeometry,
    TorchBaseSampler,
    TorchFlowModelAdapter,
    VelocityRegression,
)
from diffusiongym.registry import base_model_registry, reward_registry
from diffusiongym.rewards import DummyReward, Reward
from diffusiongym.schedulers import (
    ConstantNoiseSchedule,
    CosineScheduler,
    DiffusionScheduler,
    MemorylessNoiseSchedule,
    NoiseSchedule,
    OptimalTransportScheduler,
    Scheduler,
)
from diffusiongym.types import DDBatch, DDTensor

__all__ = [
    # core
    "AffineFlowMarginalPreservingSDE",
    "AffineGaussianForwardProcess",
    "AffineSchedule",
    "BaseSampler",
    "ConstantDiffusionSchedule",
    "DataCodec",
    "DefaultEulerGaussianKernelFactory",
    "DifferentiableTerminalCost",
    "DynamicsCoefficients",
    "EulerGaussianKernelFactory",
    "EulerMaruyamaSampler",
    "EulerODESampler",
    "FlowDynamics",
    "FlowEnvironment",
    "FlowModel",
    "ForwardBatch",
    "GaussianMarkovKernel",
    "IdentityCodec",
    "LatentGeometry",
    "MarkovKernel",
    "MemorylessDiffusionSchedule",
    "MemorylessFlowSDE",
    "PredictionConverter",
    "PredictionKind",
    "PolicyBundle",
    "ProbabilityFlowODE",
    "RectifiedFlowSchedule",
    "RewardBatch",
    "RewardEvaluator",
    "Rollout",
    "RolloutRequest",
    "RolloutStep",
    "RolloutStorage",
    "ScalarDiffusionSchedule",
    "TensorGeometry",
    "TorchBaseSampler",
    "TorchFlowModelAdapter",
    "VelocityRegression",
    # base
    "BaseModel",
    # registry
    "base_model_registry",
    "reward_registry",
    # rewards
    "DummyReward",
    "Reward",
    # schedulers (legacy)
    "ConstantNoiseSchedule",
    "CosineScheduler",
    "DiffusionScheduler",
    "MemorylessNoiseSchedule",
    "NoiseSchedule",
    "OptimalTransportScheduler",
    "Scheduler",
    # types
    "DDBatch",
    "DDTensor",
]
