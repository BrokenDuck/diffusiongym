"""Tests for environment drift implementations.

Verifies:
- the key invariant: drift_from_prediction(policy_pred) == reference_drift + sigma * control
- current_drift respects mode and policy
- mode-dependent cost computation
- schedule validation guards
"""

import pytest
import torch

from diffusiongym.environments import (
    EndpointEnvironment,
    EnvironmentMode,
    EpsilonEnvironment,
    ScoreEnvironment,
    VelocityEnvironment,
)
from diffusiongym.schedulers import OptimalTransportScheduler
from diffusiongym.types import DDTensor

ALL_ENV_CLASSES = [EndpointEnvironment, EpsilonEnvironment, ScoreEnvironment, VelocityEnvironment]


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class DummyBaseModel:
    """Minimal base model that returns a fixed output."""

    output_type = "endpoint"

    def __init__(self, output: DDTensor, scheduler=None):
        self._output = output
        self._scheduler = scheduler or OptimalTransportScheduler()
        self.device = torch.device("cpu")

    @property
    def scheduler(self):
        return self._scheduler

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs) -> DDTensor:
        return self._output


class DummyReward:
    def __call__(self, sample, latent, **kwargs):
        n = len(sample)
        return torch.zeros(n), torch.ones(n, dtype=torch.bool)


def make_env(env_class, n=4, dim=3, t_val=0.5, mode=EnvironmentMode.KL_REGULARIZED_RL):
    """Create an environment with deterministic state for testing."""
    torch.manual_seed(42)
    x = DDTensor(torch.randn(n, dim))
    base_output = DDTensor(torch.randn(n, dim))
    t = torch.full((n,), t_val)

    scheduler = OptimalTransportScheduler()
    base_model = DummyBaseModel(base_output, scheduler)
    reward = DummyReward()

    env = env_class(base_model, reward, discretization_steps=10, mode=mode)
    return env, x, t, base_output


# ---------------------------------------------------------------------------
# Test: key invariant — drift_from_prediction(policy) == reference + sigma * control
# ---------------------------------------------------------------------------


class TestDriftControlInvariant:
    """The key invariant: drift(policy_pred) == drift(base_pred) + sigma * control_delta."""

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_invariant_holds(self, env_class):
        env, x, t, base_pred = make_env(env_class)

        torch.manual_seed(123)
        policy_pred = DDTensor(torch.randn_like(x.data))

        # LHS: drift from policy prediction directly
        lhs = env.drift_from_prediction(x, t, policy_pred)

        # RHS: reference drift + sigma * control_from_prediction_delta
        ref = env.drift_from_prediction(x, t, base_pred)
        sigma = env.scheduler.sigma(x, t)
        delta = policy_pred - base_pred
        control = env.control_from_prediction_delta(x, t, delta)
        rhs = ref + sigma * control

        torch.testing.assert_close(lhs.data, rhs.data, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Test: current_drift matches drift_from_prediction(policy_output)
# ---------------------------------------------------------------------------


class TestCurrentDrift:
    """Verify current_drift uses drift_from_prediction with policy output."""

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_policy_inference_uses_policy_prediction(self, env_class):
        env, x, t, _base_pred = make_env(env_class, mode=EnvironmentMode.POLICY_INFERENCE)

        torch.manual_seed(123)
        policy_output = DDTensor(torch.randn_like(x.data))
        env.policy = lambda x, t, **kw: policy_output

        drift = env.current_drift(x, t)
        expected = env.drift_from_prediction(x, t, policy_output)

        torch.testing.assert_close(drift.data, expected.data, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_base_inference_ignores_policy(self, env_class):
        env, x, t, base_pred = make_env(env_class, mode=EnvironmentMode.BASE_INFERENCE)

        torch.manual_seed(123)
        policy_output = DDTensor(torch.randn_like(x.data))
        env.policy = lambda x, t, **kw: policy_output

        drift = env.current_drift(x, t)
        expected = env.drift_from_prediction(x, t, base_pred)

        torch.testing.assert_close(drift.data, expected.data, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_control_policy_adds_sigma_control(self, env_class):
        env, x, t, _base_pred = make_env(env_class, mode=EnvironmentMode.POLICY_INFERENCE)

        torch.manual_seed(99)
        control = DDTensor(torch.randn_like(x.data))
        env.control_policy = lambda x, t, **kw: control

        drift = env.current_drift(x, t)

        # Without control policy (base prediction used since no policy set)
        base_drift = env.reference_drift(x, t)
        sigma = env.scheduler.sigma(x, t)
        expected = base_drift + sigma * control

        torch.testing.assert_close(drift.data, expected.data, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: running cost modes
# ---------------------------------------------------------------------------


class TestRunningCostModes:
    """Verify running cost is zero in inference modes and nonzero in RL/AM modes."""

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_zero_cost_in_base_inference(self, env_class):
        env, x, t, _base_pred = make_env(env_class, mode=EnvironmentMode.BASE_INFERENCE)
        _, cost = env.drift(x, t)
        torch.testing.assert_close(cost, torch.zeros(len(x)))

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_zero_cost_in_policy_inference(self, env_class):
        env, x, t, _base_pred = make_env(env_class, mode=EnvironmentMode.POLICY_INFERENCE)
        env.policy = lambda x, t, **kw: DDTensor(torch.randn_like(x.data))
        _, cost = env.drift(x, t)
        torch.testing.assert_close(cost, torch.zeros(len(x)))

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_positive_cost_in_kl_mode_with_policy(self, env_class):
        env, x, t, base_pred = make_env(env_class, mode=EnvironmentMode.KL_REGULARIZED_RL)
        shifted = DDTensor(base_pred.data + 1.0)
        env.policy = lambda x, t, **kw: shifted
        _, cost = env.drift(x, t)
        assert (cost > 0).all()

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_zero_cost_when_policy_equals_base(self, env_class):
        env, x, t, base_pred = make_env(env_class, mode=EnvironmentMode.KL_REGULARIZED_RL)
        env.policy = lambda x, t, **kw: base_pred
        _, cost = env.drift(x, t)
        torch.testing.assert_close(cost, torch.zeros(len(x)), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Test: Adjoint Matching mode validates memoryless schedule
# ---------------------------------------------------------------------------


class TestAdjointMatchingMode:
    """Verify Adjoint Matching mode enforces memoryless schedule."""

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_adjoint_matching_passes_with_memoryless(self, env_class):
        # OT scheduler with memoryless noise schedule should pass
        env, x, t, base_pred = make_env(env_class, mode=EnvironmentMode.ADJOINT_MATCHING)
        env.policy = lambda x, t, **kw: base_pred
        # Should not raise
        env.drift(x, t)


# ---------------------------------------------------------------------------
# Test: reference_relative_control
# ---------------------------------------------------------------------------


class TestReferenceRelativeControl:
    """Verify reference_relative_control computes correct control."""

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_zero_control_when_policy_equals_base(self, env_class):
        env, x, t, base_pred = make_env(env_class)
        env.policy = lambda x, t, **kw: base_pred
        control = env.reference_relative_control(x, t)
        torch.testing.assert_close(control.data, torch.zeros_like(x.data), rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_nonzero_control_when_policy_differs(self, env_class):
        env, x, t, base_pred = make_env(env_class)
        shifted = DDTensor(base_pred.data + 1.0)
        env.policy = lambda x, t, **kw: shifted
        control = env.reference_relative_control(x, t)
        assert control.aggregate("sum").abs().sum() > 0

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_control_policy_added_to_relative_control(self, env_class):
        env, x, t, _base_pred = make_env(env_class)

        torch.manual_seed(99)
        direct_control = DDTensor(torch.randn_like(x.data))
        env.control_policy = lambda x, t, **kw: direct_control

        # No prediction policy, so only direct_control contributes
        control = env.reference_relative_control(x, t)
        torch.testing.assert_close(control.data, direct_control.data, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: stochastic schedule requirement
# ---------------------------------------------------------------------------


class TestStochasticScheduleRequirement:
    """reference_relative_control requires sigma > 0."""

    @pytest.mark.parametrize("env_class", ALL_ENV_CLASSES)
    def test_raises_when_sigma_zero(self, env_class):
        # At t=1 for OT: alpha=1, beta=0, eta=0, sigma=0
        env, x, t, base_pred = make_env(env_class, t_val=1.0)
        shifted = DDTensor(base_pred.data + 1.0)
        env.policy = lambda x, t, **kw: shifted

        with pytest.raises(ValueError, match="sigma == 0"):
            env.reference_relative_control(x, t)


# ---------------------------------------------------------------------------
# Test: diffusion term
# ---------------------------------------------------------------------------


class TestDiffusion:
    """Verify diffusion term equals sigma from the scheduler."""

    def test_diffusion_equals_sigma(self):
        env, x, t, _ = make_env(EndpointEnvironment)
        diffusion = env.diffusion(x, t)
        sigma = env.scheduler.sigma(x, t)
        torch.testing.assert_close(diffusion.data, sigma.data)


