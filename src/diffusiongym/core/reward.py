"""Reward interfaces for fine-tuning.

Two separate protocols are provided:

  RewardEvaluator  — black-box reward; used by all fine-tuning algorithms.
  DifferentiableTerminalCost — differentiable scalar cost; required ONLY by
      Adjoint Matching. A black-box reward is not a valid terminal cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from torch import Tensor

Conditioning = Mapping[str, Any]


@dataclass(frozen=True)
class RewardBatch:
    """Output of a reward evaluation.

    Parameters
    ----------
    rewards:
        Scalar reward per sample, shape (n,).
    valid:
        Boolean mask for samples with defined rewards, shape (n,).
        None means all samples are valid.
    metadata:
        Optional auxiliary data (e.g., per-sample diagnostics).
    """

    rewards: Tensor
    valid: Tensor | None = None
    metadata: Mapping[str, Any] | None = None


class RewardEvaluator[RawT, StateT](Protocol):
    """Black-box reward evaluator.

    Receives the decoded sample (RawT) and the latent state (StateT).
    May be non-differentiable.
    """

    def __call__(
        self,
        *,
        sample: RawT,
        latent: StateT,
        conditioning: Conditioning,
    ) -> RewardBatch:
        """Evaluate reward for a batch of terminal samples.

        Parameters
        ----------
        sample:
            Decoded terminal sample (output of codec.decode).
        latent:
            Terminal latent state before decoding.
        conditioning:
            Conditioning inputs used during generation.
        """
        ...


class DifferentiableTerminalCost[StateT](Protocol):
    """Differentiable terminal cost for Adjoint Matching.

    Unlike RewardEvaluator, this must be differentiable w.r.t. terminal_latent
    so that the terminal adjoint a_K = -∇_{x_K} cost(x_K) can be computed.

    Do not use a black-box RewardEvaluator here — autograd through an
    opaque function will silently return zero gradients.
    """

    def __call__(
        self,
        terminal_latent: StateT,
        *,
        conditioning: Conditioning,
    ) -> Tensor:
        """Compute one differentiable scalar cost per batch element.

        Parameters
        ----------
        terminal_latent:
            Terminal latent state x_K with requires_grad=True.
        conditioning:
            Conditioning inputs.

        Returns
        -------
        Tensor of shape (n,).
        """
        ...
