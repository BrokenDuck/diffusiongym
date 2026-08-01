"""Unit tests for core/ mathematical invariants.

Tests that schedule endpoint conditions, interpolation math, prediction conversion,
dynamics properties, and kernel log-prob are all numerically correct.
"""

from __future__ import annotations

import math

import pytest
import torch

from diffusiongym.core import (
    AffineFlowMarginalPreservingSDE,
    AffineGaussianForwardProcess,
    ConstantDiffusionSchedule,
    DefaultEulerGaussianKernelFactory,
    GaussianMarkovKernel,
    MemorylessDiffusionSchedule,
    MemorylessFlowSDE,
    PredictionConverter,
    PredictionKind,
    ProbabilityFlowODE,
    RectifiedFlowSchedule,
    RolloutRequest,
    RolloutStorage,
    TensorGeometry,
)
from diffusiongym.types import DDTensor


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class TestRectifiedFlowSchedule:
    def test_endpoint_conditions(self):
        sched = RectifiedFlowSchedule()
        t0 = torch.tensor(1e-6)
        t1 = torch.tensor(1.0 - 1e-6)

        assert torch.allclose(sched.a(t0), torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(sched.b(t0), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(sched.a(t1), torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(sched.b(t1), torch.tensor(1.0), atol=1e-5)

    def test_derivatives(self):
        sched = RectifiedFlowSchedule()
        t = torch.rand(10)
        assert torch.allclose(sched.da_dt(t), -torch.ones(10))
        assert torch.allclose(sched.db_dt(t), torch.ones(10))

    def test_velocity_target(self):
        """For rectified flow, target velocity = x_data - x_base."""
        sched = RectifiedFlowSchedule()
        t = torch.rand(4)
        x_base = torch.randn(4)
        x_data = torch.randn(4)
        target = sched.da_dt(t) * x_base + sched.db_dt(t) * x_data
        expected = x_data - x_base
        assert torch.allclose(target, expected)


# ---------------------------------------------------------------------------
# DDTensor algebra
# ---------------------------------------------------------------------------

class TestDDTensorAlgebra:
    def _make(self, n, d=3):
        return DDTensor(torch.randn(n, d))

    def test_add_subtract(self):
        x = self._make(4)
        y = self._make(4)
        result = x + y
        assert torch.allclose(result.data, x.data + y.data)
        diff = x - y
        assert torch.allclose(diff.data, x.data - y.data)

    def test_scale_scalar(self):
        x = self._make(4)
        result = x * 2.0
        assert torch.allclose(result.data, x.data * 2.0)

    def test_scale_batched_tensor(self):
        x = self._make(4)
        c = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = x * c
        expected = x.data * c.unsqueeze(-1)
        assert torch.allclose(result.data, expected)

    def test_index_select_int(self):
        x = DDTensor(torch.arange(20, dtype=torch.float).view(4, 5))
        result = x.index_select(1)
        assert result.data.shape == (1, 5)
        assert torch.allclose(result.data[0], x.data[1])

    def test_index_select_tensor(self):
        x = DDTensor(torch.arange(20, dtype=torch.float).view(4, 5))
        idx = torch.tensor([0, 2])
        result = x[idx]
        assert torch.allclose(result.data, x.data[[0, 2]])

    def test_concat(self):
        xs = [DDTensor(torch.randn(3, 2)) for _ in range(4)]
        result = DDTensor.concat(xs)
        assert result.data.shape == (12, 2)

    def test_as_leaf(self):
        x = self._make(4)
        leaf = x.as_leaf()
        assert all(t.requires_grad for t in leaf.state_tensors())
        assert all(t.grad_fn is None for t in leaf.state_tensors())

    def test_assert_compatible_ok(self):
        x = self._make(4)
        y = self._make(4)
        x.assert_compatible(y)  # should not raise

    def test_assert_compatible_fail(self):
        x = self._make(4, 3)
        y = self._make(4, 5)
        with pytest.raises(ValueError):
            x.assert_compatible(y)

    def test_float_dtype_required(self):
        with pytest.raises(TypeError):
            DDTensor(torch.zeros(4, dtype=torch.int32))


# ---------------------------------------------------------------------------
# TensorGeometry
# ---------------------------------------------------------------------------

class TestTensorGeometry:
    def setup_method(self):
        self.geom = TensorGeometry()

    def _make(self, n, d=3):
        return DDTensor(torch.randn(n, d))

    def test_squared_norm_mean(self):
        x = DDTensor(torch.ones(4, 3))
        norm = self.geom.squared_norm(x, reduction="mean")
        assert norm.shape == (4,)
        assert torch.allclose(norm, torch.ones(4))  # mean of [1,1,1] = 1

    def test_squared_norm_sum(self):
        x = DDTensor(torch.ones(4, 3))
        norm = self.geom.squared_norm(x, reduction="sum")
        assert torch.allclose(norm, 3.0 * torch.ones(4))  # sum of [1,1,1] = 3

    def test_active_dimensions(self):
        x = DDTensor(torch.randn(5, 4))
        d = self.geom.active_dimensions(x)
        assert d.shape == (5,)
        assert (d == 4).all()

    def test_project_identity(self):
        x = self._make(4)
        assert torch.allclose(self.geom.project(x).data, x.data)

    def test_standard_normal_like_shape(self):
        x = self._make(4)
        noise = self.geom.standard_normal_like(x)
        assert noise.data.shape == x.data.shape


# ---------------------------------------------------------------------------
# AffineGaussianForwardProcess
# ---------------------------------------------------------------------------

class TestAffineGaussianForwardProcess:
    def setup_method(self):
        self.geom = TensorGeometry()
        self.schedule = RectifiedFlowSchedule()

        class _Sampler:
            def sample_like(self, x_data, *, generator=None):
                return DDTensor(torch.randn_like(x_data.data, generator=generator))

        self.forward_process = AffineGaussianForwardProcess(
            geometry=self.geom,
            base_sampler=_Sampler(),
            schedule=self.schedule,
        )

    def test_interpolation_at_t0(self):
        """x_t ≈ x_base at t≈0."""
        n, d = 8, 3
        x_data = DDTensor(torch.randn(n, d))
        t = torch.full((n,), 1e-6)

        class _FixedSampler:
            def sample_like(self, x_data, *, generator=None):
                return DDTensor(torch.ones_like(x_data.data))

        fp = AffineGaussianForwardProcess(
            geometry=self.geom, base_sampler=_FixedSampler(), schedule=self.schedule
        )
        batch = fp.make_batch(x_data, conditioning={}, t=t)
        # At t≈0: x_t ≈ (1-0)*x_base + 0*x_data = x_base
        assert torch.allclose(batch.x_t.data, torch.ones(n, d), atol=1e-5)

    def test_interpolation_at_t1(self):
        """x_t ≈ x_data at t≈1."""
        n, d = 8, 3
        x_data = DDTensor(torch.randn(n, d))
        t = torch.full((n,), 1.0 - 1e-6)

        class _FixedSampler:
            def sample_like(self, x_data, *, generator=None):
                return DDTensor(torch.zeros_like(x_data.data))

        fp = AffineGaussianForwardProcess(
            geometry=self.geom, base_sampler=_FixedSampler(), schedule=self.schedule
        )
        batch = fp.make_batch(x_data, conditioning={}, t=t)
        assert torch.allclose(batch.x_t.data, x_data.data, atol=1e-5)

    def test_target_velocity_rectified(self):
        """For rectified flow: target_velocity = x_data - x_base."""
        n, d = 8, 3
        x_data = DDTensor(torch.randn(n, d))
        x_base_val = torch.randn(n, d)

        class _FixedSampler:
            def __init__(self, val):
                self.val = val
            def sample_like(self, x_data, *, generator=None):
                return DDTensor(self.val)

        fp = AffineGaussianForwardProcess(
            geometry=self.geom,
            base_sampler=_FixedSampler(x_base_val),
            schedule=self.schedule,
        )
        t = torch.rand(n)
        batch = fp.make_batch(x_data, conditioning={}, t=t)
        expected = x_data.data - x_base_val
        assert torch.allclose(batch.target_velocity.data, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# PredictionConverter
# ---------------------------------------------------------------------------

class TestPredictionConverter:
    def setup_method(self):
        self.geom = TensorGeometry()
        self.schedule = RectifiedFlowSchedule()
        self.converter = PredictionConverter(geometry=self.geom, schedule=self.schedule)

    def test_velocity_identity(self):
        n, d = 4, 3
        v = DDTensor(torch.randn(n, d))
        x_t = DDTensor(torch.randn(n, d))
        t = torch.rand(n)
        result = self.converter.to_velocity(prediction=v, kind=PredictionKind.VELOCITY, x_t=x_t, t=t)
        assert torch.allclose(result.data, v.data)

    def test_endpoint_roundtrip(self):
        """Convert endpoint prediction to velocity and back."""
        n, d = 4, 3
        t = torch.rand(n).clamp(0.05, 0.95)
        x_base = DDTensor(torch.randn(n, d))
        x_data = DDTensor(torch.randn(n, d))

        a = self.schedule.a(t).unsqueeze(-1)
        b = self.schedule.b(t).unsqueeze(-1)
        x_t = DDTensor(a * x_base.data + b * x_data.data)

        # True velocity for rectified flow
        true_v = DDTensor(x_data.data - x_base.data)

        # Use x_data as endpoint prediction
        converted = self.converter.to_velocity(
            prediction=x_data, kind=PredictionKind.ENDPOINT, x_t=x_t, t=t
        )
        assert torch.allclose(converted.data, true_v.data, atol=1e-5)

    def test_endpoint_singular_raises(self):
        """Endpoint conversion at t=1 (a≈0) should raise."""
        n, d = 4, 3
        t = torch.full((n,), 1.0)
        x = DDTensor(torch.randn(n, d))
        pred = DDTensor(torch.randn(n, d))
        with pytest.raises(ValueError, match="a\\(t\\)"):
            self.converter.to_velocity(prediction=pred, kind=PredictionKind.ENDPOINT, x_t=x, t=t)


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

class TestMemorylessDynamics:
    def setup_method(self):
        self.schedule = RectifiedFlowSchedule()
        self.dynamics = MemorylessFlowSDE(affine_schedule=self.schedule)

    def test_sigma_squared_equals_two_eta(self):
        """Memoryless invariant: sigma^2 = 2*eta."""
        mds = MemorylessDiffusionSchedule(self.schedule)
        for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            t = torch.tensor([t_val])
            sigma = mds.value(t)
            eta = mds.eta(t)
            sigma_sq = sigma ** 2
            two_eta = 2.0 * eta
            assert torch.allclose(sigma_sq, two_eta, rtol=1e-4), (
                f"sigma^2 != 2*eta at t={t_val}: {sigma_sq.item()} vs {two_eta.item()}"
            )

    def test_memoryless_flag(self):
        assert self.dynamics.memoryless is True
        assert self.dynamics.stochastic is True

    def test_ode_flags(self):
        ode = ProbabilityFlowODE()
        assert ode.stochastic is False
        assert ode.memoryless is False

    def test_drift_shape(self):
        n, d = 4, 3
        x = DDTensor(torch.randn(n, d))
        t = torch.rand(n).clamp(0.05, 0.95)
        v = DDTensor(torch.randn(n, d))
        coeffs = self.dynamics.coefficients(x=x, t=t, velocity=v)
        assert coeffs.drift.data.shape == (n, d)
        assert coeffs.diffusion.shape == (n,)


# ---------------------------------------------------------------------------
# GaussianMarkovKernel
# ---------------------------------------------------------------------------

class TestGaussianMarkovKernel:
    def setup_method(self):
        self.geom = TensorGeometry()

    def test_log_prob_at_mean(self):
        """log p(mean) = -d/2 * log(2π*var)."""
        n, d = 4, 3
        mean = DDTensor(torch.randn(n, d))
        var = torch.full((n,), 2.0)
        kernel = GaussianMarkovKernel(self.geom, mean, var)
        lp = kernel.log_prob(mean)
        expected = -d / 2.0 * math.log(2 * math.pi * 2.0) * torch.ones(n)
        assert lp.shape == (n,)
        assert torch.allclose(lp, expected, atol=1e-5)

    def test_rsample_shape(self):
        n, d = 8, 5
        mean = DDTensor(torch.zeros(n, d))
        var = torch.ones(n)
        kernel = GaussianMarkovKernel(self.geom, mean, var)
        sample = kernel.rsample()
        assert sample.data.shape == (n, d)

    def test_kl_zero_identical_kernels(self):
        """KL(p || p) = 0."""
        n, d = 4, 3
        mean = DDTensor(torch.randn(n, d))
        var = torch.ones(n)
        k = GaussianMarkovKernel(self.geom, mean, var)
        kl = k.kl_divergence(k)
        assert torch.allclose(kl, torch.zeros(n), atol=1e-6)

    def test_kl_nonnegative(self):
        n, d = 4, 3
        mean1 = DDTensor(torch.randn(n, d))
        mean2 = DDTensor(torch.randn(n, d))
        var = torch.ones(n)
        k1 = GaussianMarkovKernel(self.geom, mean1, var)
        k2 = GaussianMarkovKernel(self.geom, mean2, var)
        kl = k1.kl_divergence(k2)
        assert (kl >= -1e-6).all()


# ---------------------------------------------------------------------------
# EulerODESampler (smoke test)
# ---------------------------------------------------------------------------

class TestEulerODESampler:
    def test_deterministic_with_fixed_seed(self):
        """Same seed → same trajectory."""
        from diffusiongym.core import EulerODESampler, ProbabilityFlowODE
        from diffusiongym.core.environment import FlowEnvironment
        from diffusiongym.core.codec import IdentityCodec
        from diffusiongym.core.reward import RewardBatch
        from diffusiongym.core.rollout import RolloutRequest, RolloutStorage

        geom = TensorGeometry()
        schedule = RectifiedFlowSchedule()

        class _Sampler:
            def sample(self, n, *, conditioning, device, generator=None):
                return DDTensor(torch.randn(n, 1, generator=generator)), conditioning
            def sample_like(self, x, *, generator=None):
                return DDTensor(torch.randn_like(x.data, generator=generator))

        class _Reward:
            def __call__(self, *, sample, latent, conditioning):
                return RewardBatch(rewards=torch.zeros(len(sample)))

        class _NullModel:
            from diffusiongym.core.model import PredictionKind
            prediction_kind = PredictionKind.VELOCITY
            device = torch.device("cpu")
            def __call__(self, x_t, t, *, conditioning):
                return DDTensor(torch.zeros_like(x_t.data))

        converter = PredictionConverter(geometry=geom, schedule=schedule)
        from diffusiongym.core.model import VelocityRegression
        from diffusiongym.core.process import AffineGaussianForwardProcess
        regression = VelocityRegression(geometry=geom, converter=converter)
        fp = AffineGaussianForwardProcess(geometry=geom, base_sampler=_Sampler(), schedule=schedule)
        env = FlowEnvironment(
            geometry=geom,
            base_sampler=_Sampler(),
            forward_process=fp,
            regression=regression,
            codec=IdentityCodec(),
            reward=_Reward(),
        )
        dynamics = ProbabilityFlowODE()
        sampler = EulerODESampler(geom)
        model = _NullModel()
        request = RolloutRequest(
            time_grid=torch.linspace(0, 1, 6),
            storage=RolloutStorage(),
            evaluate_reward=True,
        )

        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        r1 = sampler.rollout(
            environment=env, model=model, dynamics=dynamics, n=4,
            conditioning={}, request=request, generator=gen1
        )
        r2 = sampler.rollout(
            environment=env, model=model, dynamics=dynamics, n=4,
            conditioning={}, request=request, generator=gen2
        )
        assert torch.allclose(r1.terminal_latent.data, r2.terminal_latent.data)

    def test_rejects_stochastic_dynamics(self):
        from diffusiongym.core import EulerODESampler, MemorylessFlowSDE
        geom = TensorGeometry()
        schedule = RectifiedFlowSchedule()
        sampler = EulerODESampler(geom)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        with pytest.raises(ValueError, match="stochastic"):
            sampler.rollout(
                environment=None, model=None, dynamics=dynamics, n=4,
                conditioning={},
                request=__import__("diffusiongym.core.rollout", fromlist=["RolloutRequest"]).RolloutRequest(
                    time_grid=torch.linspace(0, 1, 3)
                ),
            )
