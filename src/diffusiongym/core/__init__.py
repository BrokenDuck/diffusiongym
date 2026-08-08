"""Core affine-Gaussian flow-matching framework.

This module implements the narrow affine-Gaussian flow-matching environment
described in specs.md and specs_additional.md. It is separate from the
legacy environments/ and train.py code.

Supported:
  - Euclidean latent state with optional linear constraints (TensorGeometry)
  - Gaussian base distribution (TorchBaseSampler)
  - Affine interpolation schedule (RectifiedFlowSchedule, general AffineSchedule)
  - Velocity, endpoint, and noise prediction (PredictionConverter)
  - ODE and SDE sampling (EulerODESampler, EulerMaruyamaSampler)
  - SMC guidance: SDE sampling twisted by a caller-supplied terminal potential,
    via incremental resampling (SMCSampler) — same paths, same kernels, no new
    dynamics profile
  - Gaussian Markov transition kernels (GaussianMarkovKernel)
  - Immutable environment facade (FlowEnvironment)
  - Algorithm-owned policy bundle (PolicyBundle)

Not supported (explicit exclusion):
  - Non-Gaussian base distributions
  - Non-affine interpolants
  - State-dependent diffusion
  - Manifold-valued states
  - Adaptive integrators
"""

from .codec import DataCodec, IdentityCodec
from .dynamics import (
    AffineFlowMarginalPreservingSDE,
    DynamicsCoefficients,
    FlowDynamics,
    MemorylessFlowSDE,
    ProbabilityFlowODE,
)
from .environment import FlowEnvironment, PolicyBundle
from .kernel import (
    DefaultEulerGaussianKernelFactory,
    EulerGaussianKernelFactory,
    GaussianMarkovKernel,
    MarkovKernel,
)
from .model import (
    FlowModel,
    PredictionConverter,
    PredictionKind,
    TorchBaseSampler,
    TorchFlowModelAdapter,
    VelocityRegression,
)
from .process import (
    AffineGaussianForwardProcess,
    BaseSampler,
    ForwardBatch,
)
from .reward import (
    DifferentiableTerminalCost,
    RewardBatch,
    RewardEvaluator,
)
from .rollout import (
    EulerMaruyamaSampler,
    EulerODESampler,
    Rollout,
    RolloutRequest,
    RolloutStep,
    RolloutStorage,
    SMCStats,
)
from .schedule import (
    AffineSchedule,
    ConstantDiffusionSchedule,
    MemorylessDiffusionSchedule,
    RectifiedFlowSchedule,
    ScalarDiffusionSchedule,
    ScaledMemorylessDiffusionSchedule,
)
from .smc import SMCSampler
from .space import LatentGeometry, TensorGeometry

__all__ = [  # noqa: RUF022
    # geometry
    "LatentGeometry",
    "TensorGeometry",
    # schedule
    "AffineSchedule",
    "RectifiedFlowSchedule",
    "ScalarDiffusionSchedule",
    "ConstantDiffusionSchedule",
    "MemorylessDiffusionSchedule",
    "ScaledMemorylessDiffusionSchedule",
    # process
    "BaseSampler",
    "ForwardBatch",
    "AffineGaussianForwardProcess",
    # model
    "PredictionKind",
    "FlowModel",
    "PredictionConverter",
    "VelocityRegression",
    "TorchFlowModelAdapter",
    "TorchBaseSampler",
    # codec
    "DataCodec",
    "IdentityCodec",
    # reward
    "RewardBatch",
    "RewardEvaluator",
    "DifferentiableTerminalCost",
    # dynamics
    "DynamicsCoefficients",
    "FlowDynamics",
    "ProbabilityFlowODE",
    "AffineFlowMarginalPreservingSDE",
    "MemorylessFlowSDE",
    # kernel
    "MarkovKernel",
    "GaussianMarkovKernel",
    "EulerGaussianKernelFactory",
    "DefaultEulerGaussianKernelFactory",
    # rollout
    "RolloutStorage",
    "RolloutRequest",
    "RolloutStep",
    "Rollout",
    "SMCStats",
    "EulerODESampler",
    "EulerMaruyamaSampler",
    "SMCSampler",
    # environment
    "FlowEnvironment",
    "PolicyBundle",
]
