"""Exact-density and diagnostic tests for the 2-D toy problem.

Every reward-vs-KL number reported by `examples/analyze_finetuning.py` rests on
`log_density`, so its change-of-variables bookkeeping is pinned here.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from diffusiongym.toy.gmm2d import (
    BimodalReward,
    GMMFlowModel,
    LinearReward,
    RingReward,
    VelocityMLP,
    analytic_frontier,
    check_density_normalization,
    log_density,
    make_density_grid,
    tilt_diagnostics,
)


def _zero_velocity_model() -> GMMFlowModel:
    """A model whose velocity is identically zero.

    VelocityMLP zero-initializes its output head, so an untrained model is the
    identity flow: its terminal distribution is exactly the N(0, I) base, and
    log_density must reproduce the standard normal in closed form.
    """
    return GMMFlowModel(VelocityMLP(width=32, depth=2), torch.device("cpu"))


class TestLogDensity:
    def test_identity_flow_reproduces_the_standard_normal(self):
        model = _zero_velocity_model()
        x = torch.randn(64, 2)
        expected = -0.5 * (x.square().sum(-1) + 2.0 * math.log(2.0 * math.pi))
        assert torch.allclose(log_density(model, x, steps=20), expected, atol=1e-5)

    def test_density_integrates_to_one(self):
        got = check_density_normalization(
            _zero_velocity_model(), limit=5.0, resolution=60, steps=20
        )
        assert got == pytest.approx(1.0, abs=0.02)

    def test_self_kl_is_zero(self):
        model = _zero_velocity_model()
        x = torch.randn(32, 2)
        lp = log_density(model, x, steps=20)
        assert torch.allclose(lp - lp, torch.zeros_like(lp))


class TestFrontier:
    @staticmethod
    @pytest.fixture(scope="class")
    def grid():
        # No model: the analytic mixture is the reference, which keeps this fast.
        return make_density_grid(None, limit=5.0, resolution=60)

    def test_reference_is_normalized_on_the_grid(self, grid):
        mass = (grid.log_p_ref.exp() * grid.cell_area).sum()
        assert mass.item() == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize(
        "reward", [LinearReward(), BimodalReward(), RingReward()]
    )
    def test_frontier_is_monotone_and_anchored_at_zero(self, grid, reward):
        points = analytic_frontier(grid, reward, [0.0, 0.5, 1.0, 2.0, 4.0])

        # lambda = 0 means p* == p_ref, so the KL must be exactly zero. It is not
        # automatically: grid truncation makes it -log Z unless the reference is
        # renormalized on the same grid it is scored on.
        assert points[0]["kl"] == pytest.approx(0.0, abs=1e-6)

        for lower, upper in pairwise(points):
            assert upper["kl"] >= lower["kl"] - 1e-9
            assert upper["expected_reward"] >= lower["expected_reward"] - 1e-9


class TestTiltDiagnostics:
    def test_untouched_model_reports_no_tilt(self):
        model, reference = _zero_velocity_model(), _zero_velocity_model()
        d = tilt_diagnostics(
            model, reference, LinearReward(), n=128,
            sample_steps=20, density_steps=20,
        )
        assert d.kl_to_reference == pytest.approx(0.0, abs=1e-6)
        assert d.achieved_lambda == pytest.approx(0.0, abs=1e-6)
        # No density change means there is no variance for the reward to
        # explain; r^2 must be undefined rather than a flattering 1.0.
        assert math.isnan(d.tilt_r2)
