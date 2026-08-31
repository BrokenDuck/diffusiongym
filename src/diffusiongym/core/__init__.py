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
  - Local exploration: partial noising to an intermediate level, then denoising
    back (local_proposals, tail_time_grid)
  - Temporal Score Rescaling: a local sampling temperature, applied as a model
    wrapper so every sampler inherits it unchanged (rescale.py)
  - Entropic optimal transport on particle sets, whose dual potential is the
    first variation a locality penalty needs (ot.py) — plain tensors, no state
    type, no critic network
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
from .ot import (
    sinkhorn_cost,
    sinkhorn_divergence_potential,
    sinkhorn_potentials,
)
from .refine import local_proposals, tail_time_grid
from .rescale import TemporalScoreRescaling
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
    "TemporalScoreRescaling",
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
    "local_proposals",
    "tail_time_grid",
    # optimal transport
    "sinkhorn_potentials",
    "sinkhorn_cost",
    "sinkhorn_divergence_potential",
    # environment
    "FlowEnvironment",
    "PolicyBundle",
]
