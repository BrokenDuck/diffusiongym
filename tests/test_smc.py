"""Tests for `SMCSampler` (twisted-SMC inference-time guidance).

Reuses the `conftest.py` fixture graph (schedule/geometry/flow_env/tiny_model/
memoryless_sde_dynamics) rather than building parallel scaffolding, per
CLAUDE.md's testing-conventions guidance.
"""

from __future__ import annotations

import pytest
import torch

from diffusiongym.core import (
    DefaultEulerGaussianKernelFactory,
    EulerMaruyamaSampler,
    ProbabilityFlowODE,
    RolloutRequest,
    SMCSampler,
)
from diffusiongym.core.smc import _systematic_resample


def _kernel_factory(geometry):
    return DefaultEulerGaussianKernelFactory(geometry)


class TestSystematicResample:
    def test_uniform_weights_give_the_identity_permutation(self):
        """A single uniform draw fixes n evenly spaced CDF offsets; under
        uniform weights every offset lands in its own bucket. This is what
        makes a constant potential leave the rollout untouched (see
        TestSMCSampler.test_constant_potential_reproduces_plain_sde_rollout)."""
        weights = torch.full((16,), 1.0 / 16)
        gen = torch.Generator().manual_seed(0)
        idx = _systematic_resample(weights, generator=gen)
        assert torch.equal(idx, torch.arange(16))

    def test_every_particle_with_weight_above_one_over_n_survives(self):
        n = 8
        weights = torch.zeros(n)
        weights[0] = 0.5  # far above 1/n = 0.125
        weights[1:] = 0.5 / (n - 1)
        gen = torch.Generator().manual_seed(0)
        idx = _systematic_resample(weights, generator=gen)
        assert (idx == 0).sum() >= 1


class TestSMCSamplerValidation:
    def test_rejects_deterministic_dynamics(self, geometry, flow_env, tiny_model):
        sampler = SMCSampler(geometry, _kernel_factory(geometry))
        request = RolloutRequest(time_grid=torch.linspace(0, 1, 6))
        with pytest.raises(ValueError, match="stochastic"):
            sampler.rollout(
                environment=flow_env,
                model=tiny_model,
                dynamics=ProbabilityFlowODE(),
                n=4,
                conditioning={},
                request=request,
                log_potential=lambda x1, _t: torch.zeros(len(x1)),
            )

    def test_rejects_bad_ess_threshold(self, geometry):
        with pytest.raises(ValueError, match="ess_threshold"):
            SMCSampler(geometry, _kernel_factory(geometry), ess_threshold=0.0)
        with pytest.raises(ValueError, match="ess_threshold"):
            SMCSampler(geometry, _kernel_factory(geometry), ess_threshold=1.5)

    def test_rejects_bad_potential_every(self, geometry):
        with pytest.raises(ValueError, match="potential_every"):
            SMCSampler(geometry, _kernel_factory(geometry), potential_every=0)


class TestSMCSampler:
    def test_constant_potential_reproduces_plain_sde_rollout(
        self, geometry, flow_env, tiny_model, memoryless_sde_dynamics, time_grid_short
    ):
        """Zero potential differentiates no particle, so ESS stays at n the whole
        rollout, no mid-rollout resample fires (do_resample requires ESS below
        threshold*n <= n), and the one unconditional final resample degenerates
        to the identity permutation (uniform weights). The only extra randomness
        SMC consumes relative to a plain SDE rollout is that final resample's
        draw, which happens strictly after the last kernel.rsample call — so it
        cannot perturb the returned particles. Terminal latents should therefore
        match a plain EulerMaruyamaSampler rollout exactly, seed for seed."""
        smc = SMCSampler(geometry, _kernel_factory(geometry), resample="systematic")
        sde = EulerMaruyamaSampler(geometry, _kernel_factory(geometry))
        request = RolloutRequest(time_grid=time_grid_short, evaluate_reward=False)

        gen1 = torch.Generator().manual_seed(7)
        gen2 = torch.Generator().manual_seed(7)

        smc_result = smc.rollout(
            environment=flow_env,
            model=tiny_model,
            dynamics=memoryless_sde_dynamics,
            n=16,
            conditioning={},
            request=request,
            log_potential=lambda x1, _t: torch.zeros(len(x1)),
            generator=gen1,
        )
        sde_result = sde.rollout(
            environment=flow_env,
            model=tiny_model,
            dynamics=memoryless_sde_dynamics,
            n=16,
            conditioning={},
            request=request,
            generator=gen2,
        )

        assert torch.allclose(
            smc_result.terminal_latent.data, sde_result.terminal_latent.data
        )
        assert smc_result.smc is not None
        assert smc_result.smc.num_resamples == 0

    def test_region_favoring_potential_shifts_the_terminal_distribution(
        self, geometry, flow_env, tiny_model, memoryless_sde_dynamics
    ):
        """The whole point of SMC guidance: a potential that scores x>0 highly
        should measurably pull the terminal distribution positive, without any
        change to the model's own velocity field."""
        n = 256
        long_grid = torch.linspace(0.0, 1.0, 21)
        request = RolloutRequest(time_grid=long_grid, evaluate_reward=False)

        smc = SMCSampler(geometry, _kernel_factory(geometry))
        smc_result = smc.rollout(
            environment=flow_env,
            model=tiny_model,
            dynamics=memoryless_sde_dynamics,
            n=n,
            conditioning={},
            request=request,
            log_potential=lambda x1, _t: x1.data.squeeze(-1) * 8.0,
            generator=torch.Generator().manual_seed(0),
        )

        sde = EulerMaruyamaSampler(geometry, _kernel_factory(geometry))
        baseline = sde.rollout(
            environment=flow_env,
            model=tiny_model,
            dynamics=memoryless_sde_dynamics,
            n=n,
            conditioning={},
            request=request,
            generator=torch.Generator().manual_seed(0),
        )

        smc_mean = smc_result.terminal_latent.data.mean().item()
        baseline_mean = baseline.terminal_latent.data.mean().item()
        assert smc_mean > baseline_mean + 0.5, (smc_mean, baseline_mean)

    def test_ess_trace_and_resample_count_are_reported(
        self, geometry, flow_env, tiny_model, memoryless_sde_dynamics
    ):
        n = 64
        grid = torch.linspace(0.0, 1.0, 11)
        request = RolloutRequest(time_grid=grid, evaluate_reward=False)
        # High threshold + a spread-out potential forces at least one resample.
        sampler = SMCSampler(geometry, _kernel_factory(geometry), ess_threshold=0.9)

        result = sampler.rollout(
            environment=flow_env,
            model=tiny_model,
            dynamics=memoryless_sde_dynamics,
            n=n,
            conditioning={},
            request=request,
            log_potential=lambda x1, _t: x1.data.squeeze(-1) * 6.0,
            generator=torch.Generator().manual_seed(3),
        )

        assert result.smc is not None
        assert result.smc.ess_trace.shape == (10,)
        assert result.smc.resampled.shape == (10,)
        assert (result.smc.ess_trace <= n + 1e-3).all()
        assert result.smc.num_resamples == int(result.smc.resampled.sum().item())
        assert result.smc.num_resamples >= 1

    def test_returned_batch_size_matches_n_after_any_resample(
        self, geometry, flow_env, tiny_model, memoryless_sde_dynamics
    ):
        n = 32
        request = RolloutRequest(
            time_grid=torch.linspace(0.0, 1.0, 11), evaluate_reward=False
        )
        sampler = SMCSampler(geometry, _kernel_factory(geometry), ess_threshold=0.9)
        result = sampler.rollout(
            environment=flow_env,
            model=tiny_model,
            dynamics=memoryless_sde_dynamics,
            n=n,
            conditioning={},
            request=request,
            log_potential=lambda x1, _t: x1.data.squeeze(-1) * 6.0,
            generator=torch.Generator().manual_seed(5),
        )
        assert len(result.terminal_latent) == n
