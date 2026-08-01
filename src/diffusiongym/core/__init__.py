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

from diffusiongym.core.codec import DataCodec, IdentityCodec
from diffusiongym.core.dynamics import (
    AffineFlowMarginalPreservingSDE,
    DynamicsCoefficients,
    FlowDynamics,
    MemorylessFlowSDE,
    ProbabilityFlowODE,
)
from diffusiongym.core.environment import FlowEnvironment, PolicyBundle
from diffusiongym.core.kernel import (
    DefaultEulerGaussianKernelFactory,
    EulerGaussianKernelFactory,
    GaussianMarkovKernel,
    MarkovKernel,
)
from diffusiongym.core.model import (
    FlowModel,
    PredictionConverter,
    PredictionKind,
    TorchBaseSampler,
    TorchFlowModelAdapter,
    VelocityRegression,
)
from diffusiongym.core.process import (
    AffineGaussianForwardProcess,
    BaseSampler,
    ForwardBatch,
)
from diffusiongym.core.reward import (
    DifferentiableTerminalCost,
    RewardBatch,
    RewardEvaluator,
)
from diffusiongym.core.rollout import (
    EulerMaruyamaSampler,
    EulerODESampler,
    Rollout,
    RolloutRequest,
    RolloutStep,
    RolloutStorage,
)
from diffusiongym.core.schedule import (
    AffineSchedule,
    ConstantDiffusionSchedule,
    MemorylessDiffusionSchedule,
    RectifiedFlowSchedule,
    ScalarDiffusionSchedule,
)
from diffusiongym.core.space import LatentGeometry, TensorGeometry

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
    "EulerODESampler",
    "EulerMaruyamaSampler",
    # environment
    "FlowEnvironment",
    "PolicyBundle",
]
