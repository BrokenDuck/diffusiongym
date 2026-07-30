"""Tests for all trainers (ORW-CFM, Diffusion-NFT, Reward-Weighted MLE, Adjoint Matching)
using the 1D GMM toy environment.

Each test verifies that after a small number of fine-tuning iterations on the 1D Gaussian
reward, the mean reward of sampled trajectories improves relative to the pre-fine-tuning baseline.
"""

import copy

import pytest
import torch
import torch.distributions as dist

from diffusiongym.toy.gmm import MLP, OneDimensionalBaseModel
from diffusiongym.make import construct_env
from diffusiongym.toy.rewards import GaussianReward
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.train import train_base_model
from diffusiongym.trainers import adjoint_matching, diffusion_nft, orw_cfm, reward_weighted_mle
from diffusiongym.types import DDTensor

DEVICE = torch.device("cpu")
DISCRETIZATION_STEPS = 10
REWARD_SCALE = 5.0
NUM_ITERS = 5
SAMPLES_PER_ITER = 64
LR = 1e-3
BASE_MODEL_TRAIN_STEPS = 1000


def _make_env(output_type: str = "velocity"):
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

    reward = GaussianReward()
    env = construct_env(model, reward, DISCRETIZATION_STEPS, REWARD_SCALE)
    return env


def _baseline_reward(env) -> float:
    env.base_model.eval()
    sample = env.sample(256, pbar=False)
    return sample.rewards.mean().item()


class TestORWCFM:
    def test_reward_improves(self):
        torch.manual_seed(0)
        env = _make_env()
        baseline = _baseline_reward(env)

        orw_cfm(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            batch_size=SAMPLES_PER_ITER,
            steps_per_iter=50,
            num_iterations=NUM_ITERS,
            lr=LR,
        )

        final = _baseline_reward(env)
        assert final > baseline, f"ORW-CFM reward did not improve: {baseline:.4f} -> {final:.4f}"

    def test_importable_from_diffusiongym(self):
        from diffusiongym.trainers import orw_cfm as _orw_cfm  # noqa: F401


class TestDiffusionNFT:
    def test_reward_improves(self):
        torch.manual_seed(1)
        env = _make_env()
        baseline = _baseline_reward(env)

        diffusion_nft(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            sample_batch_size=SAMPLES_PER_ITER,
            ft_batch_size=32,
            num_iterations=NUM_ITERS,
            lr=LR,
        )

        final = _baseline_reward(env)
        assert final > baseline, f"Diffusion-NFT reward did not improve: {baseline:.4f} -> {final:.4f}"

    def test_policy_reset_after_training(self):
        torch.manual_seed(2)
        env = _make_env()

        diffusion_nft(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            sample_batch_size=SAMPLES_PER_ITER,
            ft_batch_size=32,
            num_iterations=2,
            lr=LR,
        )

        assert env.policy is env.base_model, "policy should be reset to base_model after training"


class TestRewardWeightedMLE:
    def test_reward_improves(self):
        torch.manual_seed(3)
        env = _make_env()
        baseline = _baseline_reward(env)

        reward_weighted_mle(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            num_iterations=NUM_ITERS,
            lr=LR,
        )

        final = _baseline_reward(env)
        assert final > baseline, f"Reward-Weighted MLE reward did not improve: {baseline:.4f} -> {final:.4f}"

    def test_noise_schedule_override_applied(self):
        from diffusiongym.schedulers.base import MemorylessNoiseSchedule

        torch.manual_seed(4)
        env = _make_env()
        original_schedule = env.base_model.scheduler.noise_schedule

        class FakeSchedule(MemorylessNoiseSchedule):
            called = False

            def __call__(self, x, t):
                FakeSchedule.called = True
                return super().__call__(x, t)

        override = FakeSchedule(env.base_model.scheduler)
        reward_weighted_mle(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            num_iterations=1,
            lr=LR,
            noise_schedule_override=override,
        )

        assert FakeSchedule.called, "noise_schedule_override was not called"
        assert env.base_model.scheduler.noise_schedule is override


class TestAdjointMatching:
    def test_reward_improves(self):
        torch.manual_seed(5)
        env = _make_env()
        baseline = _baseline_reward(env)

        adjoint_matching(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            train_steps=50,
            train_batch_size=32,
            num_iterations=NUM_ITERS,
            lr=LR,
        )

        final = _baseline_reward(env)
        assert final > baseline, f"Adjoint Matching reward did not improve: {baseline:.4f} -> {final:.4f}"

    def test_env_restored_after_training(self):
        from diffusiongym.environments.base import EnvironmentMode

        torch.manual_seed(6)
        env = _make_env()
        original_model = env.base_model

        adjoint_matching(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            train_steps=10,
            train_batch_size=32,
            num_iterations=2,
            lr=LR,
        )

        assert env.policy is None, "policy should be None after adjoint_matching"
        assert env.mode == EnvironmentMode.BASE_INFERENCE, "mode should be reset to BASE_INFERENCE"
        assert env.base_model is not original_model, "base_model should be updated to the trained policy"

    @pytest.mark.parametrize("output_type", ["velocity", "endpoint", "score", "epsilon"])
    def test_all_output_types(self, output_type: str):
        torch.manual_seed(7)
        env = _make_env(output_type)
        baseline = _baseline_reward(env)

        adjoint_matching(
            env,
            samples_per_iter=SAMPLES_PER_ITER,
            train_steps=50,
            train_batch_size=32,
            num_iterations=NUM_ITERS,
            lr=LR,
        )

        final = _baseline_reward(env)
        assert final > baseline, f"[{output_type}] Adjoint Matching reward did not improve: {baseline:.4f} -> {final:.4f}"
