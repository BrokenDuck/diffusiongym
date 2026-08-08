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

# A time grid that keeps every Euler-Maruyama step contractive under the
# memoryless SDE (|1 - dt/t| <= 1 requires dt <= 2t). Starting at t=0 does not.
_STABLE_GRID = torch.linspace(0.25, 1.0, 4)


class TestFlowGRPOSmoke:
    def test_runs_one_iteration(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(group_size=2, ppo_epochs=1, ppo_batch_size=4)
        algo.validate(context=ctx, dynamics=dynamics)
        time_grid = _STABLE_GRID
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8, time_grid=time_grid, conditioning={}
        )
        metrics = algo.update(context=ctx, experience=exp)
        assert "loss" in metrics

    def test_ratio_is_one_at_identical_parameters(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        """log π_θ(x_{k+1}|x_k) - log π_θ_old(x_{k+1}|x_k) = 0 when θ == θ_old.

        This is the fundamental correctness invariant for importance sampling:
        the ratio must be exactly 1 before any gradient step. If the drift
        used in update() differs from the one used during collection (e.g.
        raw velocity vs. full SDE drift), the log-probs disagree and this
        test fails.
        """
        from diffusiongym.core import DefaultEulerGaussianKernelFactory
        from diffusiongym.types import DDTensor

        # Use identical models for train and rollout so θ == θ_old
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(group_size=2, ppo_epochs=1, ppo_batch_size=32, beta_kl=0.0)
        algo.validate(context=ctx, dynamics=dynamics)

        time_grid = _STABLE_GRID
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8, time_grid=time_grid, conditioning={}
        )

        rollout = exp.rollout
        n = len(rollout.terminal_latent)
        device = tiny_model.device
        kernel_factory = ctx.sde_sampler.kernel_factory

        for step in rollout.steps:
            x_k = step.x.to(device)  # type: ignore[attr-defined]
            x_next = step.x_next.to(device)  # type: ignore[attr-defined]
            t_k = step.t.to(device)
            dt_k = step.dt.to(device).unsqueeze(0).expand(n)
            lp_old = step.log_prob.to(device)

            # Recompute log-prob with the same model (θ == θ_old)
            v = flow_env.predict_velocity(tiny_model, x_t=x_k, t=t_k, conditioning={})
            coeffs = dynamics.coefficients(x=x_k, t=t_k, velocity=v)
            kernel = kernel_factory.build(
                x=x_k, t=t_k, dt=dt_k, drift=coeffs.drift, diffusion=coeffs.diffusion
            )
            lp_new = kernel.log_prob(x_next)

            log_ratio = lp_new - lp_old
            assert torch.allclose(log_ratio, torch.zeros_like(log_ratio), atol=1e-5), (
                f"log-ratio not zero at step: max|log_ratio|={log_ratio.abs().max():.6f}. "
                "Drift in update() does not match drift used during collection."
            )

    def test_kl_zero_when_train_equals_ref(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        """KL(π_θ || π_ref) = 0 when θ == θ_ref.

        The KL term is computed from the kernel means, which depend on the full
        SDE drift. If the dynamics object is not used consistently the means
        differ even for identical models and KL is non-zero.
        """
        from diffusiongym.core import DefaultEulerGaussianKernelFactory

        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        kernel_factory = ctx.sde_sampler.kernel_factory
        device = tiny_model.device

        # Build a single step manually
        n, d = 4, 1
        from diffusiongym.types import DDTensor
        x = DDTensor(torch.randn(n, d))
        t = torch.full((n,), 0.5)
        dt = torch.full((n,), 0.1)

        v = flow_env.predict_velocity(tiny_model, x_t=x, t=t, conditioning={})
        coeffs = dynamics.coefficients(x=x, t=t, velocity=v)

        # Both kernels built from identical drift → KL must be zero
        kernel_a = kernel_factory.build(x=x, t=t, dt=dt, drift=coeffs.drift, diffusion=coeffs.diffusion)
        kernel_b = kernel_factory.build(x=x, t=t, dt=dt, drift=coeffs.drift, diffusion=coeffs.diffusion)
        kl = kernel_a.kl_divergence(kernel_b)

        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6), (
            f"KL not zero for identical kernels: max={kl.abs().max():.8f}"
        )

    def test_advantages_are_group_relative(self):
        """Advantages are normalized within a group, not across the batch."""
        from diffusiongym.trainers.flow_grpo import _grouped_advantages

        rewards = torch.tensor([0.0, 1.0, 100.0, 101.0])
        labels = torch.tensor([0, 0, 1, 1])
        adv = _grouped_advantages(
            rewards, labels=labels, num_groups=2, valid=None, epsilon=1e-4
        )
        # Both groups have the same internal spread, so both must yield the same
        # advantages; the 100x offset between groups must not leak in.
        assert torch.allclose(adv[:2], adv[2:], atol=1e-5)
        assert adv[0] < 0 < adv[1]

    def test_degenerate_group_gives_near_zero_advantage(self):
        """An additive eps_A must not amplify reward noise in a flat group.

        With the previous `std.clamp_min(1e-8)` a group whose rewards differ by
        1e-7 produced unit-magnitude advantages — pure noise promoted to a
        full-strength training signal.
        """
        from diffusiongym.trainers.flow_grpo import _grouped_advantages

        rewards = torch.tensor([1.0, 1.0 + 1e-7, 1.0, 1.0 - 1e-7])
        labels = torch.zeros(4, dtype=torch.long)
        adv = _grouped_advantages(
            rewards, labels=labels, num_groups=1, valid=None, epsilon=1e-4
        )
        assert adv.abs().max() < 1e-2

    def test_invalid_samples_excluded_from_group_statistics(self):
        from diffusiongym.trainers.flow_grpo import _grouped_advantages

        rewards = torch.tensor([0.0, 1.0, 1e6])
        valid = torch.tensor([True, True, False])
        labels = torch.zeros(3, dtype=torch.long)
        adv = _grouped_advantages(
            rewards, labels=labels, num_groups=1, valid=valid, epsilon=1e-4
        )
        assert adv[2] == 0.0
        # Mean/std come from the two valid samples only.
        assert torch.allclose(adv[:2], torch.tensor([-0.7071, 0.7071]), atol=1e-3)

    def test_group_must_not_mix_conditions(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        """Contiguous groups must be homogeneous in the conditioning."""
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(group_size=4, ppo_epochs=1, ppo_batch_size=4)
        conditioning = {"prompt": torch.arange(8).unsqueeze(-1).float()}
        with pytest.raises(ValueError, match="varies inside a group"):
            algo.collect(
                context=ctx, dynamics=dynamics, n=8,
                time_grid=_STABLE_GRID, conditioning=conditioning,
            )

    def test_explicit_group_labels_are_honoured(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        """conditioning[group_key] defines the groups, whatever the batch layout."""
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(group_size=4, ppo_epochs=1, ppo_batch_size=8)
        # Interleaved (not contiguous) groups.
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8, time_grid=_STABLE_GRID,
            conditioning={"group_id": labels},
        )
        rewards = exp.rollout.reward.rewards
        for g in (0, 1):
            mask = labels == g
            # Zero-mean within each group is the defining property.
            assert abs(exp.advantages[mask].mean().item()) < 1e-4
            if rewards[mask].std() > 1e-6:
                assert exp.advantages[mask].std().item() > 0.5

    def test_unstable_time_grid_is_rejected(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        """A grid touching t=0 makes the first EM step expansive, not a discretization."""
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(group_size=2, ppo_epochs=1, ppo_batch_size=4)
        with pytest.raises(ValueError, match="unstable"):
            algo.collect(
                context=ctx, dynamics=dynamics, n=4,
                time_grid=torch.linspace(0.0, 1.0, 4), conditioning={},
            )

    def test_unstable_time_grid_can_be_downgraded_to_a_warning(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = FlowGRPO(
            group_size=2, ppo_epochs=1, ppo_batch_size=4,
            require_stable_time_grid=False,
        )
        with pytest.warns(RuntimeWarning, match="unstable"):
            algo.collect(
                context=ctx, dynamics=dynamics, n=4,
                time_grid=torch.linspace(0.0, 1.0, 4), conditioning={},
            )

    def test_lower_noise_scale_stabilizes_the_grid(self, flow_env, tiny_model, tiny_model_clone, geometry, schedule):
        """sigma = a*sqrt(2*eta) with a < 1 shrinks the 1/t stiffness."""
        from diffusiongym.core import (
            AffineFlowMarginalPreservingSDE,
            ScaledMemorylessDiffusionSchedule,
        )

        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        # dt = 0.293 > 2 * t_0: expansive at a = sqrt(2), contractive at a = 0.5.
        grid = torch.linspace(0.12, 1.0, 4)
        algo = FlowGRPO(group_size=2, ppo_epochs=1, ppo_batch_size=4)
        with pytest.raises(ValueError, match="unstable"):
            algo.collect(
                context=ctx, dynamics=MemorylessFlowSDE(affine_schedule=schedule),
                n=4, time_grid=grid, conditioning={},
            )
        gentle = AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ScaledMemorylessDiffusionSchedule(schedule, 0.5),
        )
        exp = algo.collect(
            context=ctx, dynamics=gentle, n=4, time_grid=grid, conditioning={}
        )
        assert len(exp.rollout.steps) == 3

    def test_minibatches_are_transitions_not_time_slices(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """ppo_batch_size counts transitions drawn from the whole buffer.

        The buffer holds num_steps * n transitions, so a batch size equal to the
        whole buffer must produce exactly one optimizer step per epoch. The
        previous implementation took one step per (timestep, sample-chunk).
        """
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        n, epochs = 8, 2
        algo = FlowGRPO(group_size=2, ppo_epochs=epochs, ppo_batch_size=n * 3)
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=n, time_grid=_STABLE_GRID, conditioning={}
        )
        assert len(exp.rollout.steps) == 3

        calls = []
        real_step = ctx.optimizer.step
        ctx.optimizer.step = lambda *a, **kw: (calls.append(1), real_step(*a, **kw))[1]
        algo.update(context=ctx, experience=exp)
        assert len(calls) == epochs

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
        time_grid = torch.linspace(0, 1, 5)[1:]  # interior: kappa(t) = 1/t
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=4, time_grid=time_grid, conditioning={}
        )
        metrics = algo.update(context=ctx, experience=exp)
        assert "loss" in metrics
        assert "r_mean" in metrics

    def test_rejects_time_grid_touching_zero(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """kappa(t) = 1/t makes the first Euler-Maruyama step expansive at t=0."""
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = AdjointMatching(train_steps_per_iter=1, train_batch_size=4)
        with pytest.raises(ValueError, match="unstable"):
            algo.collect(
                context=ctx,
                dynamics=dynamics,
                n=4,
                time_grid=torch.linspace(0, 1, 4),
                conditioning={},
            )

    def test_terminal_adjoint_sign_increases_reward(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """The regression target must move the velocity *up* the reward gradient.

        With cost g = -r, the optimal control is +lambda * sigma * grad r, so the
        target correction v_target - v_ref must have a positive inner product with
        -grad(cost). A flipped adjoint sign silently minimizes the reward instead.
        """
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        algo = AdjointMatching(train_steps_per_iter=1, train_batch_size=4)
        exp = algo.collect(
            context=ctx,
            dynamics=dynamics,
            n=64,
            time_grid=torch.linspace(0, 1, 5)[1:],
            conditioning={},
        )

        env = ctx.environment
        last = len(exp.rollout.steps) - 1
        step = exp.rollout.steps[last]
        with torch.no_grad():
            v_ref = env.predict_velocity(
                ctx.policies.reference, x_t=step.x, t=step.t, conditioning={}
            )
        correction = exp.velocity_targets[last] - v_ref

        # Descent direction of the terminal cost = ascent direction of the reward.
        x_leaf = step.x.as_leaf(True)
        with torch.enable_grad():
            cost = env.terminal_cost(x_leaf, conditioning={})
            grad_cost = x_leaf.gradient(cost.sum())
        ascent = -grad_cost.detach()

        alignment = sum(
            (c * u).sum() for c, u in zip(
                correction.state_tensors(), ascent.state_tensors(), strict=True
            )
        )
        assert alignment > 0, (
            "Adjoint Matching target moves the policy down the reward gradient — "
            "the terminal adjoint sign is flipped."
        )

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


# ---------------------------------------------------------------------------
# Cross-algorithm invariants
#
# Each of these pins a property the algorithm's own derivation depends on.
# They are cheap and catch the class of bug that end-to-end sample plots hide:
# an update that still "trains" but optimizes the wrong objective.
# ---------------------------------------------------------------------------

class TestAlgorithmInvariants:
    def test_reward_stats_neutralize_a_constant_reward(self):
        """A constant reward carries no preference, so r_norm must be exactly 0.

        ORW-CFM turns r_norm into exp(temperature * r_norm) and DiffusionNFT into
        an optimality probability; if a degenerate reward leaked a non-zero
        r_norm, both would tilt on pure noise.
        """
        from diffusiongym.trainers.orw_cfm import _RewardStats

        stats = _RewardStats(halflife_iters=10.0)
        rewards = torch.full((32,), 3.5)
        stats.update(rewards)
        assert torch.allclose(stats.normalize(rewards), torch.zeros(32), atol=1e-6)

    def test_diffusion_nft_neutral_reward_leaves_policy_unchanged(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """r = 0.5 everywhere => the (2r - 1) factor vanishes => zero gradient.

        The positive and negative branches must cancel exactly when the reward
        expresses no preference; otherwise DiffusionNFT drifts on reward noise.
        """
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry)
        algo = DiffusionNFT(beta=1.0, inner_epochs=3, batch_size=8)
        dynamics = ProbabilityFlowODE()
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8,
            time_grid=torch.linspace(0, 1, 4), conditioning={},
        )
        exp.rewards = torch.full_like(exp.rewards, 2.0)  # constant => r = 0.5

        before = [p.detach().clone() for p in tiny_model.parameters()]
        algo.update(context=ctx, experience=exp)
        for p_before, p_after in zip(before, tiny_model.parameters(), strict=True):
            assert torch.allclose(p_before, p_after, atol=1e-7), (
                "DiffusionNFT moved the policy under a preference-free reward"
            )

    def test_flow_grpo_stored_log_prob_is_reproducible(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """Recomputing the kernel at update time must reproduce collection log-probs.

        Flow-GRPO's importance ratio is only exact if the kernel rebuilt from
        dynamics.coefficients() matches the one that generated the transition —
        so before any gradient step, log rho must be exactly 0.
        """
        from diffusiongym.core import (
            AffineFlowMarginalPreservingSDE,
            ScaledMemorylessDiffusionSchedule,
        )

        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry, with_ref=True)
        dynamics = AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ScaledMemorylessDiffusionSchedule(schedule, 0.75),
        )
        algo = FlowGRPO(group_size=4, ppo_epochs=1, ppo_batch_size=8, beta_kl=1.0)
        time_grid = torch.linspace(0, 1, 6)[1:]
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=8,
            time_grid=time_grid, conditioning={},
        )

        rollout = exp.rollout
        factory = ctx.sde_sampler.kernel_factory
        # Flow-GRPO collects under policies.rollout, so recompute under the same one.
        for step in rollout.steps:
            n = len(step.x)
            with torch.no_grad():
                v = flow_env.predict_velocity(
                    ctx.policies.rollout, x_t=step.x, t=step.t,
                    conditioning=rollout.conditioning,
                )
                coeffs = dynamics.coefficients(x=step.x, t=step.t, velocity=v)
                kernel = factory.build(
                    x=step.x, t=step.t, dt=step.dt.reshape(1).expand(n),
                    drift=coeffs.drift, diffusion=coeffs.diffusion,
                )
                recomputed = kernel.log_prob(step.x_next)
            assert step.log_prob is not None
            assert torch.allclose(recomputed, step.log_prob, atol=1e-5), (
                "recomputed log-prob differs from collection: the Flow-GRPO "
                "importance ratio is not exact at rho = 1"
            )

    def test_orwcfm_refreshes_its_rollout_policy(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """ORW-CFM is online: the rollout policy must track training.

        If it stays at the pretrained weights the method silently degenerates
        into a repeated one-shot importance-weighted refit of p_base·exp(λr) —
        which looks correct near the base model and stops working away from it.
        """
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry)
        algo = ORWCFM(temperature=1.0, rollout_update_interval=1,
                      steps_per_update=3, batch_size=8)
        dynamics = ProbabilityFlowODE()
        before = [p.detach().clone() for p in ctx.policies.rollout.parameters()]

        for _ in range(3):
            exp = algo.collect(
                context=ctx, dynamics=dynamics, n=8,
                time_grid=torch.linspace(0, 1, 4), conditioning={},
            )
            algo.update(context=ctx, experience=exp)
            algo.synchronize_rollout_policy(context=ctx)

        moved = max(
            (a - b).abs().max().item()
            for a, b in zip(before, ctx.policies.rollout.parameters(), strict=True)
        )
        assert moved > 0.0, "ORW-CFM rollout policy never left the pretrained weights"

    def test_orwcfm_w2_requires_a_reference_policy(
        self, flow_env, tiny_model, tiny_model_clone, geometry, schedule
    ):
        """The W2 surrogate regresses against v_ref, so it needs one."""
        ctx = _make_context(flow_env, tiny_model, tiny_model_clone, geometry)
        algo = ORWCFM(temperature=1.0, alpha_w2=0.5)
        with pytest.raises(ValueError, match="reference"):
            algo.validate(context=ctx, dynamics=ProbabilityFlowODE())
