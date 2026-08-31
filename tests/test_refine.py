"""Tests for `local_proposals` and `tail_time_grid`.

`local_proposals` targets no particular distribution — it is a mutation
operator, and which mutations survive is the caller's decision. So there is no
closed form to check it against and `test_smc_correctness.py`'s approach does
not transfer. What can be pinned exactly, and is what these tests do, is the
behaviour at both ends of the one knob that matters (`noise_frac`, the mutation
radius) plus the two structural properties a caller depends on: proposals of one
seed must genuinely differ, and seeds must not mix.
"""

import pytest
import torch

import diffusiongym
from diffusiongym import local_proposals, make_time_grid, tail_time_grid
from diffusiongym.core import ProbabilityFlowODE
from diffusiongym.types import DDTensor

DEVICE = torch.device("cpu")


def _setup():
    return diffusiongym.make(
        modality="toy/gmm2d",
        reward="toy/linear",
        algorithm="flow_grpo",
        discretization_steps=8,
        device=DEVICE,
    )


def _propose(setup, *, seeds, num_proposals=4, noise_frac=0.5, seed=0, dynamics=None):
    return local_proposals(
        environment=setup.environment,
        model=setup.context.policies.rollout,
        dynamics=dynamics if dynamics is not None else setup.dynamics,
        kernel_factory=setup.context.sde_sampler.kernel_factory,
        seeds=seeds,
        conditioning={},
        time_grid=tail_time_grid(setup.time_grid, noise_frac=noise_frac),
        num_proposals=num_proposals,
        generator=torch.Generator().manual_seed(seed),
    )


# ---------------------------------------------------------------------------
# tail_time_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "noise_frac,expected_steps,expected_start",
    [(1.0, 8, 1 / 9), (0.5, 4, 5 / 9), (0.3, 2, 7 / 9), (0.01, 1, 8 / 9)],
)
def test_tail_grid_rounds_onto_the_grid(noise_frac, expected_steps, expected_start):
    """`noise_frac` is rounded onto the discretisation, never interpolated —
    the model is evaluated at grid points, so a `t_start` between two of them
    would put the noising time and the first evaluation time on either side of a
    step."""
    grid = tail_time_grid(make_time_grid(8, stochastic=True), noise_frac=noise_frac)
    assert grid.numel() - 1 == expected_steps
    assert float(grid[0]) == pytest.approx(expected_start)
    assert float(grid[-1]) == pytest.approx(1.0)


def test_tail_grid_always_leaves_one_step():
    """A `noise_frac` smaller than a single step must still mutate once rather
    than return a zero-step grid `local_proposals` would reject."""
    grid = tail_time_grid(make_time_grid(4, stochastic=True), noise_frac=1e-9)
    assert grid.numel() == 2


@pytest.mark.parametrize("noise_frac", [0.0, -0.5, 1.5])
def test_tail_grid_validates_noise_frac(noise_frac):
    with pytest.raises(ValueError, match="noise_frac"):
        tail_time_grid(make_time_grid(4, stochastic=True), noise_frac=noise_frac)


# ---------------------------------------------------------------------------
# local_proposals
# ---------------------------------------------------------------------------


def test_smaller_noise_frac_stays_closer_to_the_seed():
    """`noise_frac` is the mutation radius, and this is the only claim the
    operator makes about *where* its output lands. Monotone in the knob, and the
    two ends have to be far apart or the knob is decorative."""
    setup = _setup()
    seeds = DDTensor(torch.randn(32, 2, generator=torch.Generator().manual_seed(3)))

    distances = []
    for noise_frac in (0.125, 0.5, 1.0):
        _, x1, _ = _propose(setup, seeds=seeds, num_proposals=2, noise_frac=noise_frac)
        # Row `l * n + j` is proposal l of seed j, so seeds tile with `repeat`.
        tiled = seeds.data.repeat(2, 1)
        distances.append(float((x1.data - tiled).norm(dim=-1).mean().item()))

    assert distances[0] < distances[1] < distances[2]
    assert distances[2] > 3 * distances[0]


def test_the_proposals_of_one_seed_differ():
    """Each of the `n * L` rows draws its own base noise and its own transition
    noise. Sharing either would make `L` copies of the same trajectory and the
    caller's selection over them a no-op — silently, since the shapes are
    identical either way."""
    setup = _setup()
    seeds = DDTensor(torch.zeros(4, 2))
    z, x1, _ = _propose(setup, seeds=seeds, num_proposals=8)

    # Column j across the 8 proposals of seed j.
    for j in range(4):
        rows = torch.arange(8) * 4 + j
        assert z.data[rows].std(dim=0).min() > 1e-3
        assert x1.data[rows].std(dim=0).min() > 1e-3


def test_seeds_do_not_mix():
    """Two well-separated seeds must produce two well-separated clouds. Nothing
    in this operator resamples across the population — that is exactly what
    distinguishes it from `SMCSampler`, whose resample can collapse every
    particle onto one seed's neighbourhood."""
    setup = _setup()
    seeds = DDTensor(torch.tensor([[-6.0, -6.0], [6.0, 6.0]]))
    _, x1, _ = _propose(setup, seeds=seeds, num_proposals=8, noise_frac=0.125)

    left = x1.data[torch.arange(8) * 2 + 0]
    right = x1.data[torch.arange(8) * 2 + 1]
    assert left[:, 0].max() < right[:, 0].min()


def test_z_lives_at_the_grids_first_point():
    """`z` is the representation the caller measures locality in, so it has to
    be at the level the tail grid starts from, not one step into the rollout."""
    setup = _setup()
    seeds = DDTensor(torch.full((64, 2), 4.0))
    grid = tail_time_grid(setup.time_grid, noise_frac=0.5)
    z, _, _ = _propose(setup, seeds=seeds, num_proposals=4, noise_frac=0.5)

    # x_t = a(t) * base + b(t) * x_1 with a Gaussian base, so E[z] = b(t) * 4.
    schedule = setup.environment.forward_process.schedule
    expected = float(schedule.b(grid[0].unsqueeze(0)).item()) * 4.0
    assert float(z.data.mean().item()) == pytest.approx(expected, abs=0.15)


def test_deterministic_dynamics_are_rejected():
    """Under the ODE the kernel variance is zero, so the proposals of a seed
    differ only through their initial noising and never decorrelate — the
    operator would be far less local than asked for, with no error."""
    setup = _setup()
    with pytest.raises(ValueError, match="stochastic"):
        _propose(
            setup,
            seeds=DDTensor(torch.zeros(2, 2)),
            dynamics=ProbabilityFlowODE(),
        )


def test_arguments_are_validated():
    setup = _setup()
    seeds = DDTensor(torch.zeros(2, 2))
    common = {
        "environment": setup.environment,
        "model": setup.context.policies.rollout,
        "dynamics": setup.dynamics,
        "kernel_factory": setup.context.sde_sampler.kernel_factory,
        "seeds": seeds,
        "conditioning": {},
    }
    with pytest.raises(ValueError, match="num_proposals"):
        local_proposals(**common, time_grid=setup.time_grid, num_proposals=0)
    with pytest.raises(ValueError, match="time_grid"):
        local_proposals(**common, time_grid=setup.time_grid[:1], num_proposals=2)
