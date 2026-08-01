"""Fine-tuning algorithms for affine-Gaussian flow models.

All algorithms implement the FineTuningAlgorithm interface:
  - validate()  — check requirements
  - collect()   — sample online experience
  - update()    — gradient step on train policy
  - synchronize_rollout_policy() — optional EMA/hard-copy
"""

from diffusiongym.trainers.adjoint_matching import AdjointMatching
from diffusiongym.trainers.base import (
    AdjointExperience,
    EndpointExperience,
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
    TrajectoryExperience,
)
from diffusiongym.trainers.diffusion_nft import DiffusionNFT
from diffusiongym.trainers.flow_grpo import FlowGRPO
from diffusiongym.trainers.orw_cfm import ORWCFM

__all__ = [
    "FineTuningAlgorithm",
    "FineTuningContext",
    "FineTuningRequirements",
    "EndpointExperience",
    "TrajectoryExperience",
    "AdjointExperience",
    "ORWCFM",
    "DiffusionNFT",
    "FlowGRPO",
    "AdjointMatching",
]
