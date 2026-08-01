"""Smoke tests: each trainer runs 1 collect + 1 update without crashing.

These tests verify:
  1. Each trainer can execute end-to-end with a tiny model.
  2. validate() raises for unmet requirements.
  3. Rollout output has the expected structure.
"""

from __future__ import annotations

import copy

import pytest
import torch

from diffusiongym.core import (
    DefaultEulerGaussianKernelFactory,
    EulerMaruyamaSampler,
    EulerODESampler,
    MemorylessFlowSDE,
    ProbabilityFlowODE,
    RectifiedFlowSchedule,
    TensorGeometry,
)
from diffusiongym.trainers import (
    AdjointMatching,
    DiffusionNFT,
    FlowGRPO,
    ORWCFM,
    FineTuningContext,
)
from diffusiongym.trainers.base import PolicyBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RefFlowModel:
    """Frozen reference model for tests."""
    from diffusiongym.core.model import PredictionKind
    prediction_kind = PredictionKind.VELOCITY

    def __init__(self, mlp, device):
        self._mlp = mlp
        self._device = device

    @property
    def device(self):
        return self._device

    def parameters(self):
        return self._mlp.parameters()

    def __call__(self, x_t, t, *, conditioning):
        from diffusiongym.types import DDTensor
        return DDTensor(self._mlp(x_t.data, t))

    def state_dict(self):
        return self._mlp.state_dict()

    def load_state_dict(self, sd):
        return self._mlp.load_state_dict(sd)


def _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=False):
    """Build a FineTuningContext for the given environment and models."""
    ref = None
    if with_ref:
        ref_mlp = copy.deepcopy(tiny_model._mlp)
        ref = _RefFlowModel(ref_mlp, torch.device("cpu"))

    bundle = PolicyBundle(train=tiny_model, rollout=tiny_model_clone, reference=ref)
    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    sde = EulerMaruyamaSampler(geometry, DefaultEulerGaussianKernelFactory(geometry))
    ode = EulerODESampler(geometry)
    return FineTuningContext(
        environment=flow_env,
        policies=bundle,
        optimizer=opt,
        ode_sampler=ode,
        sde_sampler=sde,
    )


# ---------------------------------------------------------------------------
# ORW-CFM
# ---------------------------------------------------------------------------

class TestORWCFMSmoke:
    def test_runs_one_iteration(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry)
        dynamics = ProbabilityFlowODE()
        algo = ORWCFM(steps_per_update=2, batch_size=8)
        algo.validate(context=ctx, dynamics=dynamics)
        time_grid = torch.linspace(0, 1, 4)
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8, time_grid=time_grid, conditioning={}
        )
        metrics = algo.update(context=ctx, experience=exp)
        assert "loss" in metrics
        assert "r_mean" in metrics


# ---------------------------------------------------------------------------
# DiffusionNFT
# ---------------------------------------------------------------------------

class TestDiffusionNFTSmoke:
    def test_runs_one_iteration(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry)
        dynamics = ProbabilityFlowODE()
        algo = DiffusionNFT(inner_epochs=2, batch_size=8)
        algo.validate(context=ctx, dynamics=dynamics)
        time_grid = torch.linspace(0, 1, 4)
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8, time_grid=time_grid, conditioning={}
        )
        metrics = algo.update(context=ctx, experience=exp)
        assert "loss" in metrics

    def test_ema_sync_runs(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry)
        algo = DiffusionNFT()
        algo.synchronize_rollout_policy(context=ctx)


# ---------------------------------------------------------------------------
# FlowGRPO
# ---------------------------------------------------------------------------

class TestFlowGRPOSmoke:
    def test_runs_one_iteration(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(group_size=2, ppo_epochs=1, ppo_batch_size=4)
        algo.validate(context=ctx, dynamics=dynamics)
        time_grid = torch.linspace(0, 1, 4)
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8, time_grid=time_grid, conditioning={}
        )
        metrics = algo.update(context=ctx, experience=exp)
        assert "loss" in metrics

    def test_requires_stochastic_dynamics(self, flow_env, tiny_model, tiny_model_clone, geometry):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = ProbabilityFlowODE()
        algo = FlowGRPO()
        with pytest.raises(ValueError, match="stochastic"):
            algo.validate(context=ctx, dynamics=dynamics)

    def test_requires_reference_policy(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=False)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO()
        with pytest.raises(ValueError, match="reference"):
            algo.validate(context=ctx, dynamics=dynamics)


# ---------------------------------------------------------------------------
# AdjointMatching
# ---------------------------------------------------------------------------

class TestAdjointMatchingSmoke:
    def test_runs_one_iteration(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = AdjointMatching(train_steps_per_iter=2, train_batch_size=4)
        algo.validate(context=ctx, dynamics=dynamics)
        time_grid = torch.linspace(0, 1, 4)
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=4, time_grid=time_grid, conditioning={}
        )
        metrics = algo.update(context=ctx, experience=exp)
        assert "loss" in metrics

    def test_requires_memoryless_dynamics(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        from diffusiongym.core import AffineFlowMarginalPreservingSDE, ConstantDiffusionSchedule
        dynamics = AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ConstantDiffusionSchedule(0.5),
        )
        algo = AdjointMatching()
        with pytest.raises(ValueError, match="memoryless"):
            algo.validate(context=ctx, dynamics=dynamics)

    def test_requires_differentiable_terminal_cost(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule, base_sampler, gaussian_reward
    ):
        """Environment without terminal_cost should raise."""
        from diffusiongym.core import (
            AffineGaussianForwardProcess,
            FlowEnvironment,
            IdentityCodec,
            PredictionConverter,
            VelocityRegression,
        )
        converter = PredictionConverter(geometry=geometry, schedule=schedule)
        regression = VelocityRegression(geometry=geometry, converter=converter)
        fp = AffineGaussianForwardProcess(geometry=geometry, base_sampler=base_sampler, schedule=schedule)
        env_no_cost = FlowEnvironment(
            geometry=geometry,
            base_sampler=base_sampler,
            forward_process=fp,
            regression=regression,
            codec=IdentityCodec(),
            reward=gaussian_reward,
            terminal_cost=None,
        )
        ctx = _make_context(env_no_cost, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = AdjointMatching()
        with pytest.raises(ValueError, match="terminal cost"):
            algo.validate(context=ctx, dynamics=dynamics)
