"""Shared fixtures for diffusiongym tests.

Uses the minimal 1D GMM toy model to test the full pipeline without
requiring GPU or large model weights.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch
from torch import Tensor

from diffusiongym.core import (
    AffineGaussianForwardProcess,
    DefaultEulerGaussianKernelFactory,
    DifferentiableTerminalCost,
    EulerMaruyamaSampler,
    EulerODESampler,
    FlowEnvironment,
    IdentityCodec,
    MemorylessFlowSDE,
    PolicyBundle,
    ProbabilityFlowODE,
    RectifiedFlowSchedule,
    RewardBatch,
    RewardEvaluator,
    TensorGeometry,
    TorchBaseSampler,
    TorchFlowModelAdapter,
    VelocityRegression,
    PredictionConverter,
)
from diffusiongym.types import DDTensor


# ---------------------------------------------------------------------------
# Schedule and geometry
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def schedule():
    return RectifiedFlowSchedule()


@pytest.fixture(scope="session")
def geometry():
    return TensorGeometry()


# ---------------------------------------------------------------------------
# Tiny toy model (1D velocity predictor, no pretraining)
# ---------------------------------------------------------------------------

class _TinyMLP(torch.nn.Module):
    """Tiny velocity-predicting MLP for tests (no pretraining, just structure)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        inp = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(inp)


class _TinyFlowModel:
    """Wraps _TinyMLP into a FlowModel for DDTensor states."""

    from diffusiongym.core.model import PredictionKind

    prediction_kind = PredictionKind.VELOCITY

    def __init__(self, mlp: _TinyMLP, device: torch.device) -> None:
        self._mlp = mlp
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def parameters(self):
        return self._mlp.parameters()

    def __call__(self, x_t: DDTensor, t: Tensor, *, conditioning: dict) -> DDTensor:
        return DDTensor(self._mlp(x_t.data, t))

    def state_dict(self):
        return self._mlp.state_dict()

    def load_state_dict(self, sd):
        return self._mlp.load_state_dict(sd)

    def train(self):
        self._mlp.train()

    def eval(self):
        self._mlp.eval()


@pytest.fixture
def tiny_model():
    mlp = _TinyMLP()
    return _TinyFlowModel(mlp, torch.device("cpu"))


@pytest.fixture
def tiny_model_clone(tiny_model):
    mlp_clone = copy.deepcopy(tiny_model._mlp)
    return _TinyFlowModel(mlp_clone, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Base sampler (1D isotropic Gaussian)
# ---------------------------------------------------------------------------

class _GaussianSampler:
    def sample(self, n, *, conditioning, device, generator=None):
        return DDTensor(torch.randn(n, 1, device=device, generator=generator)), conditioning

    def sample_like(self, x_data, *, generator=None):
        return DDTensor(torch.randn_like(x_data.data, generator=generator))


@pytest.fixture(scope="session")
def base_sampler():
    return _GaussianSampler()


# ---------------------------------------------------------------------------
# Reward evaluators
# ---------------------------------------------------------------------------

class _GaussianReward:
    """Simple Gaussian reward centered at x=-2.5."""

    def __call__(self, *, sample: DDTensor, latent: DDTensor, conditioning: dict) -> RewardBatch:
        mu, sigma = -2.5, 0.8
        rewards = torch.exp(-0.5 * ((sample.data - mu) / sigma) ** 2).squeeze(-1)
        return RewardBatch(rewards=rewards)


class _DifferentiableReward:
    """Differentiable version of the Gaussian reward."""

    def __call__(self, terminal_latent: DDTensor, *, conditioning: dict) -> Tensor:
        mu, sigma = -2.5, 0.8
        return torch.exp(-0.5 * ((terminal_latent.data - mu) / sigma) ** 2).squeeze(-1)


@pytest.fixture(scope="session")
def gaussian_reward():
    return _GaussianReward()


@pytest.fixture(scope="session")
def differentiable_terminal_cost():
    return _DifferentiableReward()


# ---------------------------------------------------------------------------
# FlowEnvironment
# ---------------------------------------------------------------------------

@pytest.fixture
def flow_env(geometry, schedule, base_sampler, gaussian_reward, differentiable_terminal_cost):
    converter = PredictionConverter(geometry=geometry, schedule=schedule)
    regression = VelocityRegression(geometry=geometry, converter=converter)
    forward_process = AffineGaussianForwardProcess(
        geometry=geometry,
        base_sampler=base_sampler,
        schedule=schedule,
    )
    return FlowEnvironment(
        geometry=geometry,
        base_sampler=base_sampler,
        forward_process=forward_process,
        regression=regression,
        codec=IdentityCodec(),
        reward=gaussian_reward,
        terminal_cost=differentiable_terminal_cost,
    )


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ode_sampler(geometry):
    return EulerODESampler(geometry)


@pytest.fixture(scope="session")
def sde_sampler(geometry):
    kernel_factory = DefaultEulerGaussianKernelFactory(geometry)
    return EulerMaruyamaSampler(geometry, kernel_factory)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ode_dynamics():
    return ProbabilityFlowODE()


@pytest.fixture(scope="session")
def memoryless_sde_dynamics(schedule):
    return MemorylessFlowSDE(affine_schedule=schedule)


# ---------------------------------------------------------------------------
# FineTuningContext helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def policy_bundle_no_ref(tiny_model, tiny_model_clone):
    return PolicyBundle(train=tiny_model, rollout=tiny_model_clone)


@pytest.fixture
def policy_bundle_with_ref(tiny_model, tiny_model_clone):
    import copy as _copy
    mlp_ref = _copy.deepcopy(tiny_model._mlp)
    ref_model = _TinyFlowModel(mlp_ref, torch.device("cpu"))
    return PolicyBundle(train=tiny_model, rollout=tiny_model_clone, reference=ref_model)


@pytest.fixture
def time_grid_short():
    return torch.linspace(0.0, 1.0, 6)  # 5 steps
