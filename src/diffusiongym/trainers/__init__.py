"""Fine-tuning algorithms for affine-Gaussian flow models.

All algorithms implement the FineTuningAlgorithm interface:
  - validate()  — check requirements
  - collect()   — sample online experience
  - update()    — gradient step on train policy
  - synchronize_rollout_policy() — optional EMA/hard-copy
"""

from .adjoint_matching import AdjointMatching
from .base import (
    AdjointExperience,
    EndpointExperience,
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
    TrajectoryExperience,
)
from .diffusion_nft import DiffusionNFT
from .flow_grpo import FlowGRPO
from .orw_cfm import ORWCFM

__all__ = [
    "ORWCFM",
    "AdjointExperience",
    "AdjointMatching",
    "DiffusionNFT",
    "EndpointExperience",
    "FineTuningAlgorithm",
    "FineTuningContext",
    "FineTuningRequirements",
    "FlowGRPO",
    "TrajectoryExperience",
]
