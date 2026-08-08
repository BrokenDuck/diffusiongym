"""Tests for registry-driven assembly (`diffusiongym.make`).

The value of `make()` is not saving keystrokes — it is that the choices it makes
are the ones that fail *silently* when made by hand: the SDE profile, the time
grid, whether a reference policy exists, and whether the reward has a
differentiable form. Each of those is pinned here.
"""

from __future__ import annotations

import pytest
import torch

import diffusiongym
from diffusiongym.core import (
    AffineFlowMarginalPreservingSDE,
    MemorylessFlowSDE,
    ProbabilityFlowODE,
)
from diffusiongym.registry import Registry, domain_of

ALGORITHMS = ["orw_cfm", "diffusion_nft", "flow_grpo", "adjoint_matching"]


def _make(algorithm: str, **kwargs):
    return diffusiongym.make(
        modality="toy/gmm2d",
        reward=kwargs.pop("reward", "toy/linear"),
        algorithm=algorithm,
        discretization_steps=kwargs.pop("discretization_steps", 6),
        **kwargs,
    )


class TestRegistry:
    def test_lists_are_populated_after_make(self):
        _make("orw_cfm")
        assert "toy/gmm2d" in diffusiongym.modality_registry.list()
        assert set(diffusiongym.algorithm_registry.list()) == set(ALGORITHMS)

    def test_unknown_id_lists_the_valid_ones(self):
        with pytest.raises(KeyError, match="adjoint_matching"):
            _make("nope")

    def test_re_registering_the_same_class_is_allowed(self):
        class Alpha: ...

        class Beta: ...

        registry = Registry("T")
        registry.register("a", Alpha)
        registry.register("a", Alpha)  # idempotent: module imported twice
        with pytest.raises(ValueError, match="already registered"):
            registry.register("a", Beta)

    def test_domain_of(self):
        assert domain_of("toy/linear") == "toy"
        assert domain_of("linear") is None


class TestRequirementDrivenAssembly:
    """The three choices `make()` derives from `algorithm.requirements`."""

    @pytest.mark.parametrize(
        ("algorithm", "expected"),
        [
            ("orw_cfm", ProbabilityFlowODE),
            ("diffusion_nft", ProbabilityFlowODE),
            ("flow_grpo", AffineFlowMarginalPreservingSDE),
            ("adjoint_matching", MemorylessFlowSDE),
        ],
    )
    def test_dynamics_match_the_algorithm(self, algorithm, expected):
        setup = _make(algorithm)
        assert isinstance(setup.dynamics, expected)
        # MemorylessFlowSDE subclasses the marginal-preserving SDE, so check the
        # flag too rather than relying on the class alone.
        assert setup.dynamics.memoryless == (algorithm == "adjoint_matching")

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_time_grid_is_interior_exactly_when_stochastic(self, algorithm):
        setup = _make(algorithm)
        if setup.dynamics.stochastic:
            assert setup.time_grid[0] > 0.0, (
                "a stochastic rollout on a grid touching t=0 is expansive"
            )
        else:
            assert setup.time_grid[0] == 0.0
        assert setup.time_grid[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_reference_policy_exists_only_where_required(self, algorithm):
        setup = _make(algorithm)
        needs = setup.algorithm.requirements.needs_reference_policy
        assert (setup.context.policies.reference is not None) == needs

    def test_policies_start_from_identical_weights(self):
        setup = _make("flow_grpo")
        train = setup.context.policies.train.state_dict()
        for other in (setup.context.policies.rollout, setup.context.policies.reference):
            for key, value in other.state_dict().items():
                assert torch.equal(value, train[key]), (
                    f"{key} differs between policies at initialization"
                )


class TestRejections:
    def test_adjoint_matching_refuses_a_non_differentiable_reward(self):
        with pytest.raises(ValueError, match="differentiable terminal cost"):
            _make("adjoint_matching", reward="toy/box")

    def test_black_box_algorithms_accept_the_same_reward(self):
        setup = _make("orw_cfm", reward="toy/box")
        assert setup.environment.terminal_cost is None

    def test_mismatched_domains_are_rejected(self):
        with pytest.raises(ValueError, match="Incompatible domains"):
            diffusiongym.make(
                modality="toy/gmm2d", reward="molecules/dipole", algorithm="orw_cfm"
            )

    def test_zero_discretization_steps_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            _make("orw_cfm", discretization_steps=0)


class TestSetupRuns:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_one_iteration_end_to_end(self, algorithm):
        """A setup from `make()` must run without any further wiring."""
        setup = _make(algorithm, algorithm_kwargs=_small(algorithm))
        experience = setup.algorithm.collect(
            context=setup.context,
            dynamics=setup.dynamics,
            n=8,
            time_grid=setup.time_grid,
            conditioning={},
        )
        metrics = setup.algorithm.update(
            context=setup.context, experience=experience
        )
        setup.algorithm.synchronize_rollout_policy(context=setup.context)
        assert "loss" in metrics

    def test_kwargs_reach_the_constructors(self):
        setup = _make(
            "adjoint_matching",
            algorithm_kwargs={"lambda_reward": 3.0, "train_steps_per_iter": 1},
            reward_kwargs={"direction": (0.0, 1.0)},
        )
        assert setup.algorithm.lambda_reward == 3.0
        # reward_kwargs must reach both the reward and its differentiable cost
        assert torch.equal(
            setup.environment.reward.c, torch.tensor([0.0, 1.0])
        )
        assert torch.equal(
            setup.environment.terminal_cost.c, torch.tensor([0.0, 1.0])
        )

    def test_optimizer_factory_is_honoured(self):
        setup = _make(
            "orw_cfm",
            optimizer_factory=lambda model: torch.optim.SGD(
                model.parameters(), lr=0.01
            ),
        )
        assert isinstance(setup.context.optimizer, torch.optim.SGD)


def _small(algorithm: str) -> dict:
    """Minimal inner-loop sizes so the smoke test stays fast."""
    match algorithm:
        case "orw_cfm":
            return {"steps_per_update": 1, "batch_size": 4}
        case "diffusion_nft":
            return {"inner_epochs": 1, "batch_size": 4}
        case "flow_grpo":
            return {"group_size": 2, "ppo_epochs": 1, "ppo_batch_size": 8}
        case _:
            return {"train_steps_per_iter": 1, "train_batch_size": 4}
