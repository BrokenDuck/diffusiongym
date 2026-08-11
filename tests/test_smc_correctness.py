"""Does `SMCSampler` actually sample the distribution it claims to?

`test_smc.py` checks the sampler's plumbing — shapes, validation, the ESS trace,
and that a positive potential moves the mean in the positive direction. None of
that distinguishes a correct twisted-SMC implementation from a plausible one:
"the mean moved the right way" is satisfied by any monotone reweighting,
including several that target the wrong distribution entirely.

These tests answer the quantitative question instead. `toy/analytic1d.py`
supplies a flow whose velocity field is exact, so the terminal law of an SDE
rollout is a known Gaussian mixture, and whose exponential tilt
``p(x) * exp(alpha*x) / Z`` is another known mixture. That makes the
distribution `SMCSampler` is *supposed* to produce available in closed form —
means, variances and per-mode masses — so the tests can assert agreement rather
than direction.

Two references are used, and the distinction matters when one fails:

  * the **closed-form tilt** — agreement means the whole pipeline (dynamics,
    discretisation, twisting, resampling) is right;
  * **importance-reweighting a plain `EulerMaruyamaSampler` rollout on the same
    time grid** — this shares the discretisation error, so a mismatch here
    isolates the sampler.

`TestKnownDefects` at the bottom is marked `xfail(strict=True)`: those tests
assert the *correct* behaviour and currently fail. When a fix lands they flip to
unexpected passes, which is the signal to drop the marker.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from diffusiongym.core import (
    AffineFlowMarginalPreservingSDE,
    DefaultEulerGaussianKernelFactory,
    EulerMaruyamaSampler,
    RectifiedFlowSchedule,
    RolloutRequest,
    RolloutStorage,
    ScaledMemorylessDiffusionSchedule,
    SMCSampler,
    SMCStats,
    TensorGeometry,
)
from diffusiongym.toy.analytic1d import (
    ExactVelocityModel,
    Mixture1D,
    make_environment,
    weighted_moments,
)

# A well-separated symmetric pair: mode masses are then a sharp, unambiguous
# readout of how much probability the tilt moved, which a unimodal target
# cannot provide.
TWO_MODE = Mixture1D(
    weights=torch.tensor([0.5, 0.5]), means=torch.tensor([-2.0, 2.0]), sigma=0.6
)
# Asymmetric and three-way: the tilt must reproduce a *ratio* between three
# unequal masses, not just "move right".
THREE_MODE = Mixture1D(
    weights=torch.tensor([0.6, 0.3, 0.1]),
    means=torch.tensor([-3.0, 0.0, 3.0]),
    sigma=0.5,
)

STEPS = 64
NOISE_SCALE = 0.7
N_PARTICLES = 4096
N_SEEDS = 4


@pytest.fixture(scope="module")
def geometry():
    return TensorGeometry()


@pytest.fixture(scope="module")
def schedule():
    return RectifiedFlowSchedule()


@pytest.fixture(scope="module")
def kernel_factory(geometry):
    return DefaultEulerGaussianKernelFactory(geometry)


@pytest.fixture(scope="module")
def dynamics(schedule):
    return AffineFlowMarginalPreservingSDE(
        affine_schedule=schedule,
        diffusion_schedule=ScaledMemorylessDiffusionSchedule(schedule, NOISE_SCALE),
    )


@pytest.fixture(scope="module")
def time_grid():
    # Interior grid: the marginal-preserving drift carries kappa(t) = 1/t.
    return torch.linspace(0.0, 1.0, STEPS + 1)[1:]


def linear_potential(alpha: float):
    """log phi(x) = alpha * x, the tilt `Mixture1D.tilt` inverts in closed form."""

    def log_potential(x1, _t):
        return alpha * x1.data.squeeze(-1)

    return log_potential


def smc_samples(
    mixture: Mixture1D,
    alpha: float,
    *,
    geometry,
    kernel_factory,
    dynamics,
    time_grid,
    n: int = N_PARTICLES,
    seeds: int = N_SEEDS,
    **sampler_kwargs,
) -> Tensor:
    """Pooled terminal samples over several independent SMC runs.

    Pooling across seeds rather than raising `n` in one run is deliberate: it
    keeps per-run particle interaction (and hence any degeneracy) at the scale a
    caller would actually use, while still giving enough draws to resolve the
    target to a few parts in a thousand.
    """
    env = make_environment(mixture)
    model = ExactVelocityModel(mixture)
    sampler = SMCSampler(geometry, kernel_factory, **sampler_kwargs)
    out = []
    for seed in range(seeds):
        result = sampler.rollout(
            environment=env,
            model=model,
            dynamics=dynamics,
            n=n,
            conditioning={},
            request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
            log_potential=linear_potential(alpha),
            generator=torch.Generator().manual_seed(seed),
        )
        out.append(result.terminal_latent.data.squeeze(-1))
    return torch.cat(out)


def sde_samples(
    mixture: Mixture1D, *, geometry, kernel_factory, dynamics, time_grid, n, seed
) -> Tensor:
    sampler = EulerMaruyamaSampler(geometry, kernel_factory)
    result = sampler.rollout(
        environment=make_environment(mixture),
        model=ExactVelocityModel(mixture),
        dynamics=dynamics,
        n=n,
        conditioning={},
        request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
        generator=torch.Generator().manual_seed(seed),
    )
    return result.terminal_latent.data.squeeze(-1)


def stats_of(result) -> SMCStats:
    """`Rollout.smc` is optional in general; `SMCSampler` always populates it."""
    assert result.smc is not None
    return result.smc


def mixture_cdf(mixture: Mixture1D, x: Tensor) -> Tensor:
    z = (x.unsqueeze(-1) - mixture.means) / (mixture.sigma * math.sqrt(2.0))
    return (mixture.weights * 0.5 * (1.0 + torch.erf(z))).sum(-1)


def ks_distance(samples: Tensor, mixture: Mixture1D) -> float:
    """sup |F_empirical - F_target|."""
    x = samples.sort().values
    n = len(x)
    empirical = torch.arange(1, n + 1, dtype=x.dtype) / n
    return float((empirical - mixture_cdf(mixture, x)).abs().max())


class TestAnalyticGroundTruth:
    """Validate the yardstick before measuring anything with it.

    If these fail, every other failure in this file is uninterpretable.
    """

    def test_sde_rollout_reproduces_the_data_mixture(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """The exact velocity plus a marginal-preserving SDE has the data
        mixture as its exact terminal law, so an untwisted rollout must land on
        it — this is what makes the model's contribution to any later
        discrepancy zero by construction."""
        x = sde_samples(
            TWO_MODE,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
            n=40000,
            seed=11,
        )
        assert x.mean().item() == pytest.approx(TWO_MODE.mean(), abs=0.05)
        assert x.var().item() == pytest.approx(TWO_MODE.var(), rel=0.03)
        assert ks_distance(x, TWO_MODE) < 0.02

    def test_closed_form_tilt_matches_numerical_reweighting(self):
        """`Mixture1D.tilt` against brute-force quadrature on a fine grid."""
        alpha = 1.3
        grid = torch.linspace(-12.0, 12.0, 40001)
        log_p = TWO_MODE.log_density(grid) + alpha * grid
        w = torch.softmax(log_p, dim=0)
        mean = float((w * grid).sum())
        var = float((w * (grid - mean) ** 2).sum())

        tilted = TWO_MODE.tilt(alpha)
        assert mean == pytest.approx(tilted.mean(), abs=1e-3)
        assert var == pytest.approx(tilted.var(), abs=1e-3)

    def test_marginal_matches_a_forward_interpolation(self):
        """`Mixture1D.marginal(t)` against sampling x_t = (1-t) z + t x1."""
        t = 0.4
        gen = torch.Generator().manual_seed(0)
        x1 = TWO_MODE.sample(200000, generator=gen)
        z = torch.randn(200000, generator=gen)
        x_t = (1.0 - t) * z + t * x1
        marginal = TWO_MODE.marginal(t)
        assert x_t.mean().item() == pytest.approx(marginal.mean(), abs=0.02)
        assert x_t.var().item() == pytest.approx(marginal.var(), rel=0.02)


class TestSMCTargetsTheTiltedLaw:
    """The central question: is the output distribution the right one?"""

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_mean_and_variance_match_the_closed_form_tilt(
        self, alpha, geometry, kernel_factory, dynamics, time_grid
    ):
        """The variance is the discriminating half of this test. A sampler that
        merely concentrates particles on high-potential regions reproduces the
        mean while collapsing the spread; the tilt of a Gaussian mixture leaves
        each component's variance untouched, so the target variance must be
        matched from above as well as below."""
        target = TWO_MODE.tilt(alpha)
        x = smc_samples(
            TWO_MODE,
            alpha,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
        )
        assert x.mean().item() == pytest.approx(target.mean(), abs=0.05)
        assert x.var().item() == pytest.approx(target.var(), rel=0.10)

    @pytest.mark.parametrize("alpha", [0.5, 1.0])
    def test_mode_masses_match_the_closed_form_tilt(
        self, alpha, geometry, kernel_factory, dynamics, time_grid
    ):
        """On the asymmetric three-mode mixture the tilt must turn masses
        (0.6, 0.3, 0.1) into a specific new triple — at alpha=1 the smallest
        mode has to grow to 0.86. Reproducing a three-way ratio is not something
        a directionally-correct-but-wrong sampler achieves by luck."""
        target = THREE_MODE.tilt(alpha)
        x = smc_samples(
            THREE_MODE,
            alpha,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
        )
        observed = THREE_MODE.mode_masses(x)
        assert torch.allclose(observed, target.weights, atol=0.02), (
            observed.tolist(),
            target.weights.tolist(),
        )

    @pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0])
    def test_full_distribution_matches_the_closed_form_tilt(
        self, alpha, geometry, kernel_factory, dynamics, time_grid
    ):
        """Moments can agree while the shape does not; compare the CDFs.

        The threshold is loose relative to the iid Kolmogorov-Smirnov critical
        value (1.36/sqrt(n) ~= 0.011 here) because resampling makes particles
        within a run positively correlated, which inflates KS above its iid
        null. Measured values sit around 0.005-0.015, so 0.04 still fails a
        sampler that targets a materially different law.
        """
        x = smc_samples(
            TWO_MODE,
            alpha,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
        )
        assert ks_distance(x, TWO_MODE.tilt(alpha)) < 0.04

    def test_matches_importance_reweighting_of_the_same_rollout(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """Compare against self-normalised importance sampling on a plain SDE
        rollout over the same time grid.

        That reference carries the identical discretisation error, so it is the
        exact law SMC should produce — agreement here isolates the sampler from
        every other source of error in the pipeline.
        """
        alpha = 1.5
        x_smc = smc_samples(
            TWO_MODE,
            alpha,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
        )
        x_sde = sde_samples(
            TWO_MODE,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
            n=60000,
            seed=77,
        )
        is_mean, is_var = weighted_moments(x_sde, alpha * x_sde)
        assert x_smc.mean().item() == pytest.approx(is_mean, abs=0.05)
        assert x_smc.var().item() == pytest.approx(is_var, rel=0.10)

    def test_zero_potential_leaves_the_law_untouched(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """alpha = 0 must recover the base mixture exactly, not approximately:
        a sampler that leaks bias through its resampling step shows it here even
        with nothing to guide toward."""
        x = smc_samples(
            TWO_MODE,
            0.0,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
        )
        assert x.mean().item() == pytest.approx(TWO_MODE.mean(), abs=0.05)
        assert x.var().item() == pytest.approx(TWO_MODE.var(), rel=0.05)

    def test_potential_is_invariant_to_an_additive_constant(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """A log-potential is only defined up to a constant — exp(a+c)/Z is the
        same distribution as exp(a)/Z. Because the weight update telescopes
        (`logphi_k - logphi_prev`) and ESS is shift-invariant, the constant must
        cancel *bitwise*, not merely in distribution. An implementation that
        anchored weights absolutely rather than incrementally would fail this
        even though its samples still looked reasonable."""
        common = {
            "geometry": geometry,
            "kernel_factory": kernel_factory,
            "dynamics": dynamics,
            "time_grid": time_grid,
            "n": 512,
            "seeds": 1,
        }
        base = smc_samples(TWO_MODE, 1.0, **common)

        env = make_environment(TWO_MODE)
        shifted = SMCSampler(geometry, kernel_factory).rollout(
            environment=env,
            model=ExactVelocityModel(TWO_MODE),
            dynamics=dynamics,
            n=512,
            conditioning={},
            request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
            log_potential=lambda x1, _t: 1.0 * x1.data.squeeze(-1) + 37.5,
            generator=torch.Generator().manual_seed(0),
        )
        assert torch.equal(base, shifted.terminal_latent.data.squeeze(-1))


class TestResamplingMechanics:
    def test_lower_ess_threshold_resamples_less(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        counts = {}
        for threshold in (0.1, 0.5, 1.0):
            result = SMCSampler(
                geometry, kernel_factory, ess_threshold=threshold
            ).rollout(
                environment=make_environment(TWO_MODE),
                model=ExactVelocityModel(TWO_MODE),
                dynamics=dynamics,
                n=1024,
                conditioning={},
                request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
                log_potential=linear_potential(1.5),
                generator=torch.Generator().manual_seed(0),
            )
            counts[threshold] = stats_of(result).num_resamples
        assert counts[0.1] < counts[0.5] < counts[1.0]
        assert counts[1.0] == len(time_grid) - 1  # every step

    def test_intermediate_resampling_preserves_particle_diversity(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """This is what SMC buys over plain importance sampling.

        With resampling effectively disabled the weight accumulates over the
        whole rollout and the single unconditional final resample duplicates a
        small subset; with ESS-triggered resampling the weight is reset before
        it degenerates. Both target the same law, so only the diversity — not
        the mean — separates them.
        """
        def unique_count(threshold: float) -> int:
            result = SMCSampler(
                geometry, kernel_factory, ess_threshold=threshold
            ).rollout(
                environment=make_environment(TWO_MODE),
                model=ExactVelocityModel(TWO_MODE),
                dynamics=dynamics,
                n=2048,
                conditioning={},
                request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
                log_potential=linear_potential(1.5),
                generator=torch.Generator().manual_seed(0),
            )
            return len(torch.unique(result.terminal_latent.data.squeeze(-1)))

        assert unique_count(0.1) < 0.75 * unique_count(0.9)

    def test_resampling_methods_agree_in_distribution(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """Systematic resampling has lower variance than multinomial, but both
        are unbiased — they must target the same law."""
        kwargs = {
            "geometry": geometry,
            "kernel_factory": kernel_factory,
            "dynamics": dynamics,
            "time_grid": time_grid,
        }
        systematic = smc_samples(TWO_MODE, 1.5, resample="systematic", **kwargs)
        multinomial = smc_samples(TWO_MODE, 1.5, resample="multinomial", **kwargs)
        assert systematic.mean().item() == pytest.approx(
            multinomial.mean().item(), abs=0.06
        )
        target = TWO_MODE.tilt(1.5)
        assert ks_distance(multinomial, target) < 0.04

    def test_conditioning_follows_the_resampled_particles(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """A resample duplicates particles, so per-particle conditioning must be
        duplicated with them. Tensor and list entries are re-indexed by separate
        code paths in `_index_conditioning`; both are checked here because a
        divergence between them would pair each particle with another's prompt.
        """
        n = 64
        result = SMCSampler(geometry, kernel_factory, ess_threshold=0.9).rollout(
            environment=make_environment(TWO_MODE),
            model=ExactVelocityModel(TWO_MODE),
            dynamics=dynamics,
            n=n,
            conditioning={
                "tag": torch.arange(n, dtype=torch.float32),
                "note": [f"p{i}" for i in range(n)],
            },
            request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
            log_potential=linear_potential(1.5),
            generator=torch.Generator().manual_seed(0),
        )
        assert stats_of(result).num_resamples > 0
        tags = [int(v) for v in result.conditioning["tag"].tolist()]
        notes = [int(s[1:]) for s in result.conditioning["note"]]
        assert len(tags) == n
        assert tags == notes
        assert len(set(tags)) < n  # a resample really did duplicate particles

    def test_ess_trace_is_bounded_and_aligned_with_the_grid(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        result = SMCSampler(geometry, kernel_factory, ess_threshold=0.6).rollout(
            environment=make_environment(TWO_MODE),
            model=ExactVelocityModel(TWO_MODE),
            dynamics=dynamics,
            n=512,
            conditioning={},
            request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
            log_potential=linear_potential(1.5),
            generator=torch.Generator().manual_seed(0),
        )
        stats = stats_of(result)
        num_steps = len(time_grid) - 1
        assert stats.ess_trace.shape == (num_steps,)
        assert torch.isfinite(stats.ess_trace).all()
        assert (stats.ess_trace > 0).all()
        assert (stats.ess_trace <= 512 + 1e-3).all()
        # A resample resets the weights, so the step after one must be back at n.
        fired = torch.nonzero(stats.resampled).squeeze(-1)
        for k in fired.tolist():
            if k + 1 < num_steps:
                assert stats.ess_trace[k + 1] > 0.5 * 512


class TestKnownDefects:
    """Behaviour that is currently wrong. Each asserts the *correct* outcome.

    Marked `strict`, so a fix turns them into unexpected passes rather than
    silently passing unnoticed.
    """

    def test_potential_every_changes_only_efficiency_not_the_target(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """A terminal-anchored SMC degenerates, at `potential_every =
        num_steps`, to plain importance sampling: one uninformative
        intermediate weight plus an exact terminal correction. That is
        high-variance but still targets the tilt.

        This is the test for the terminal increment in `SMCSampler.rollout`.
        Before it existed the terminal correction was missing and the tilt was
        lost altogether here, while `potential_every=1` on a fine grid still
        looked correct — which is exactly why the check has to vary
        `potential_every` rather than only assert the default configuration.
        """
        alpha = 1.5
        target = TWO_MODE.tilt(alpha)
        x = smc_samples(
            TWO_MODE,
            alpha,
            geometry=geometry,
            kernel_factory=kernel_factory,
            dynamics=dynamics,
            time_grid=time_grid,
            n=8192,
            seeds=2,
            potential_every=len(time_grid) - 1,
        )
        assert x.mean().item() == pytest.approx(target.mean(), abs=0.15)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "-inf log-potentials produce NaN weights. logw starts at 0 and the "
            "first increment is logphi_0 - 0 = -inf for every particle whose "
            "endpoint estimate violates the constraint; the next increment is "
            "then -inf - (-inf) = NaN. ESS becomes NaN, `NaN < threshold` is "
            "False so no resample ever fires, and the run returns unguided "
            "samples with no error raised."
        ),
    )
    def test_hard_constraint_potential_is_respected(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        """Constraint satisfaction is a headline use of inference-time guidance,
        and the natural way to express it is a log-potential of 0 inside the
        feasible set and -inf outside."""

        def constraint(x1, _t):
            inside = x1.data.squeeze(-1) > 1.0
            return torch.where(
                inside, torch.zeros_like(inside, dtype=torch.float32),
                torch.full(inside.shape, -float("inf")),
            )

        result = SMCSampler(geometry, kernel_factory).rollout(
            environment=make_environment(TWO_MODE),
            model=ExactVelocityModel(TWO_MODE),
            dynamics=dynamics,
            n=2048,
            conditioning={},
            request=RolloutRequest(time_grid=time_grid, evaluate_reward=False),
            log_potential=constraint,
            generator=torch.Generator().manual_seed(0),
        )
        assert torch.isfinite(stats_of(result).ess_trace).all()
        satisfied = (result.terminal_latent.data.squeeze(-1) > 1.0).float().mean()
        assert satisfied > 0.9, float(satisfied)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Resampling re-indexes the live particles but not the already-stored "
            "`steps`, so after the first resample `steps[k].x_next` and "
            "`steps[k+1].x` describe different particles, and `terminal_latent` "
            "does not match `steps[-1].x_next`. The rollout is advertised as "
            "interchangeable with the other two samplers' output, but any "
            "consumer of trajectories (Flow-GRPO's per-step log-probs, Adjoint "
            "Matching's backward pass) would silently pair mismatched states."
        ),
    )
    def test_stored_steps_form_a_contiguous_trajectory(
        self, geometry, kernel_factory, dynamics, time_grid
    ):
        result = SMCSampler(geometry, kernel_factory, ess_threshold=0.9).rollout(
            environment=make_environment(TWO_MODE),
            model=ExactVelocityModel(TWO_MODE),
            dynamics=dynamics,
            n=64,
            conditioning={},
            request=RolloutRequest(
                time_grid=time_grid,
                evaluate_reward=False,
                storage=RolloutStorage(states=True),
            ),
            log_potential=linear_potential(1.5),
            generator=torch.Generator().manual_seed(0),
        )
        assert stats_of(result).num_resamples > 0
        for k in range(len(result.steps) - 1):
            assert torch.allclose(
                result.steps[k].x_next.data, result.steps[k + 1].x.data
            ), f"steps[{k}].x_next does not match steps[{k + 1}].x"
        assert torch.allclose(
            result.terminal_latent.data, result.steps[-1].x_next.data
        )
