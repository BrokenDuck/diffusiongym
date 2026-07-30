"""Tests for adjoint matching finetuning across all output types using the 1D GMM.

Validates:
- Memoryless schedule invariant (sigma^2 == 2*eta)
- Running costs become non-zero after policy diverges from reference
- Reward improves after finetuning
- Running costs stay bounded (KL regularization is effective)
"""

import copy

import pytest
import torch
import torch.distributions as dist

from diffusiongym.toy.gmm import MLP, OneDimensionalBaseModel
from diffusiongym.environments.base import EnvironmentMode
from diffusiongym.make import construct_env
from diffusiongym.toy.rewards import GaussianReward
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.train import train_base_model
from diffusiongym.types import DDTensor

DEVICE = torch.device("cpu")

DISCRETIZATION_STEPS = 20
REWARD_SCALE = 10.0
NUM_ITERS = 10
SAMPLES_PER_ITER = 128
TRAIN_STEPS = 100
TRAIN_BATCH = 64
LR = 1e-4
BASE_MODEL_TRAIN_STEPS = 2000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base_model(output_type: str) -> OneDimensionalBaseModel:
    """Create and train a base model with the given output type."""
    model = OneDimensionalBaseModel.__new__(OneDimensionalBaseModel)
    torch.nn.Module.__init__(model)
    model.device = DEVICE
    model.output_type = output_type
    model._scheduler = OptimalTransportScheduler()
    model.model = MLP(1, 1).to(DEVICE)

    p1 = dist.MixtureSameFamily(
        dist.Categorical(torch.ones(2)),
        dist.Normal(torch.Tensor([0.0, 3.0]), torch.Tensor([1.0, 0.4])),
    )
    data = [DDTensor(p1.sample((4096, 1)).to(DEVICE))]
    opt = torch.optim.Adam(model.model.parameters(), lr=1e-3)
    train_base_model(model, opt, data, steps=BASE_MODEL_TRAIN_STEPS, batch_size=512)
    model.eval()
    return model


def _run_finetuning(output_type: str) -> dict:
    """Run adjoint matching finetuning for one output type and return diagnostics."""
    torch.manual_seed(42)

    base_model = _make_base_model(output_type)
    reward = GaussianReward()

    env = construct_env(base_model, reward, DISCRETIZATION_STEPS, REWARD_SCALE)
    env.mode = EnvironmentMode.ADJOINT_MATCHING

    reference_model = copy.deepcopy(base_model)
    reference_model.eval()
    reference_model.requires_grad_(False)

    env.base_model = reference_model
    env.policy = base_model

    opt = torch.optim.Adam(base_model.parameters(), lr=LR)

    rewards_history = []
    running_costs_history = []

    for _ in range(NUM_ITERS):
        base_model.eval()
        sample = env.sample(SAMPLES_PER_ITER, pbar=False)

        rewards_history.append(sample.rewards.mean().item())
        running_costs_history.append(sample.running_costs.sum(dim=0).mean().item())

        advantages = sample.rewards - sample.rewards.mean()
        if advantages.std() > 1e-8:
            advantages = advantages / advantages.std()

        data = sample.trajectory[:-1]
        weights = [advantages.clamp(min=0) for _ in range(sample.num_steps)]

        train_base_model(
            base_model,
            opt,
            data=data,
            kwargs=[sample.kwargs] * sample.num_steps,
            weights=weights,
            steps=TRAIN_STEPS,
            batch_size=TRAIN_BATCH,
        )

    base_model.eval()
    final_sample = env.sample(512, pbar=False)

    return {
        "output_type": output_type,
        "rewards": rewards_history,
        "running_costs": running_costs_history,
        "final_reward": final_sample.rewards.mean().item(),
        "final_running_cost": final_sample.running_costs.sum(dim=0).mean().item(),
    }


# ---------------------------------------------------------------------------
# Tests: memoryless schedule invariant
# ---------------------------------------------------------------------------


class TestMemorylessSchedule:
    """sigma^2 == 2*eta must hold for adjoint matching."""

    @pytest.mark.parametrize("output_type", ["velocity", "endpoint", "score", "epsilon"])
    def test_sigma_squared_equals_two_eta(self, output_type: str) -> None:
        base_model = _make_base_model(output_type)
        scheduler = base_model.scheduler

        x = DDTensor(torch.randn(32, 1))
        t = torch.rand(32)

        sigma = scheduler.sigma(x, t)
        eta = scheduler.eta(x, t)

        sigma_sq = (sigma * sigma).aggregate("sum")
        two_eta = (2.0 * eta).aggregate("sum")

        torch.testing.assert_close(sigma_sq, two_eta, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: finetuning dynamics
# ---------------------------------------------------------------------------


class TestFinetuning:
    """End-to-end finetuning validates that the full pipeline works."""

    @pytest.fixture(params=["velocity", "endpoint", "score", "epsilon"])
    def finetuning_result(self, request) -> dict:
        return _run_finetuning(request.param)

    def test_running_costs_nonzero_after_training(self, finetuning_result: dict) -> None:
        """After the policy diverges from the reference, KL cost must be non-zero."""
        costs = finetuning_result["running_costs"]
        assert any(abs(c) > 1e-6 for c in costs[1:]), (
            f"[{finetuning_result['output_type']}] Running costs stayed zero — "
            f"policy may not be diverging from reference."
        )

    def test_reward_improved(self, finetuning_result: dict) -> None:
        """Final reward should exceed the initial (pre-finetuning) reward."""
        initial = finetuning_result["rewards"][0]
        final = finetuning_result["final_reward"]
        assert final > initial, (
            f"[{finetuning_result['output_type']}] Reward did not improve: "
            f"{initial:.4f} -> {final:.4f}"
        )

    def test_running_costs_bounded(self, finetuning_result: dict) -> None:
        """Running costs should not explode (KL regularization is working)."""
        max_cost = max(abs(c) for c in finetuning_result["running_costs"])
        assert max_cost < 1000, (
            f"[{finetuning_result['output_type']}] Running costs exploded: "
            f"max={max_cost:.2f}"
        )
