"""Tests for `ForwardKLDistillation`, driven directly by hand-built
`ReferenceSource`s rather than through any loop.

The load-bearing test here is `test_the_untilted_teacher_is_a_fixed_point`: it
is the one property of this loss that can be checked without a ground-truth
target, and it fails for a whole family of plausible implementations (see its
docstring).
"""

import dataclasses

import pytest
import torch

import diffusiongym
from diffusiongym.trainers import ForwardKLDistillation, ReferenceSource
from diffusiongym.types import DDTensor

DEVICE = torch.device("cpu")

_FLAT = lambda x: torch.zeros(len(x))


def _setup(**algorithm_kwargs):
    kwargs = {
        "roll_in_size": 32,
        "inner_epochs": 2,
        "batch_size": 8,
        "num_teacher_proposals": 4,
    } | algorithm_kwargs
    return diffusiongym.make(
        modality="toy/gmm2d",
        reward="toy/linear",
        algorithm="forward_kl_distillation",
        discretization_steps=8,
        device=DEVICE,
        algorithm_kwargs=kwargs,
    )


def _experience(setup, *, pool=None, log_weight=_FLAT, n=16):
    experience = setup.algorithm.collect(
        context=setup.context,
        dynamics=setup.dynamics,
        n=n,
        time_grid=setup.time_grid,
        conditioning={},
    )
    if log_weight is None:
        return experience
    endpoints = experience.latent if pool is None else pool
    return dataclasses.replace(
        experience,
        reference=ReferenceSource(
            endpoints=endpoints,
            conditioning={},
            probs=torch.ones(len(endpoints)),
        ),
        log_weight=log_weight,
    )


def _grad_norm(setup, experience):
    setup.context.optimizer.zero_grad()
    setup.algorithm.update(context=setup.context, experience=experience)
    grads = [
        p.grad.flatten()
        for p in setup.context.policies.train.parameters()
        if p.grad is not None
    ]
    return float(torch.cat(grads).norm())


# ---------------------------------------------------------------------------
# The fixed point
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [1024, 8192])
def test_the_untilted_teacher_is_a_fixed_point(size):
    """With a flat tilt, `rho` equal to the rollout policy's own samples, and
    no roll-in candidate, the teacher *is* the lazy policy — so a student
    already equal to it must see zero expected gradient.

    This is the test that catches the natural-but-wrong implementation. Mapping
    each proposal to an endpoint at `t_next` directly leaves an O(dt) offset:
    the Euler mean is `x + drift*dt` and the drift is not the velocity (it
    carries the `kappa*x` and `c(t)` corrections), so the candidate average
    lands slightly away from the student's own prediction and an *untilted*
    teacher still drags the model. `_build_teacher` centres the candidates on
    the kernel mean to remove it.

    The residual is Monte-Carlo noise around zero, so it must shrink with the
    roll-in size — a bias would not. Parametrised for exactly that reason:
    checking one size against a fixed threshold cannot tell the two apart."""
    torch.manual_seed(0)
    setup = _setup(
        roll_in_size=size,
        batch_size=size,
        inner_epochs=1,
        num_teacher_proposals=8,
        seed_candidate_weight=0.0,
    )
    experience = _experience(setup, n=64)
    assert _grad_norm(setup, experience) < 8.0 / size**0.5


def test_the_gradient_survives_a_tilt():
    """The mirror of the fixed-point test: the same setup with a real tilt must
    produce a gradient far above the noise floor, or the test above would be
    passing for the trivial reason that nothing is connected."""
    torch.manual_seed(0)
    setup = _setup(
        roll_in_size=2048,
        batch_size=2048,
        inner_epochs=1,
        num_teacher_proposals=8,
        seed_candidate_weight=0.0,
    )
    flat = _grad_norm(setup, _experience(setup, n=64))
    torch.manual_seed(0)
    setup = _setup(
        roll_in_size=2048,
        batch_size=2048,
        inner_epochs=1,
        num_teacher_proposals=8,
        seed_candidate_weight=0.0,
    )
    tilted = _grad_norm(
        setup, _experience(setup, n=64, log_weight=lambda x: 5.0 * x.data[:, 0])
    )
    assert tilted > 10 * flat


# ---------------------------------------------------------------------------
# The teacher
# ---------------------------------------------------------------------------


def test_a_flat_tilt_weights_every_candidate_equally():
    """M+1 candidates, no tilt -> ESS is exactly M+1 and the roll-in endpoint
    holds exactly 1/(M+1). Both numbers pin the `(M+1, size)` reshape: a
    transposed layout would mix candidates across roll-ins and neither would
    come out round."""
    setup = _setup(num_teacher_proposals=4)
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(setup)
    )
    assert metrics["teacher_ess"] == pytest.approx(5.0, abs=1e-4)
    assert metrics["teacher_seed_weight"] == pytest.approx(0.2, abs=1e-4)


def test_seed_candidate_weight_sets_the_reference_share():
    """`rho`'s contribution to the conditional target would otherwise be pinned
    at 1/(M+1) — an artefact of how many proposals happen to be drawn, not a
    choice. Doubling the weight must double the share."""
    setup = _setup(num_teacher_proposals=3, seed_candidate_weight=3.0)
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(setup)
    )
    # 3 / (3 + 3 proposals) = 0.5
    assert metrics["teacher_seed_weight"] == pytest.approx(0.5, abs=1e-4)


def test_zero_proposals_reduces_to_flow_matching_on_the_reference():
    """M=0 leaves the roll-in endpoint as the only candidate, so the target is
    that endpoint and the loss is plain flow matching on `rho^+`. It is the
    documented degenerate case and must not divide by zero on the way."""
    setup = _setup(num_teacher_proposals=0)
    metrics = setup.algorithm.update(
        context=setup.context, experience=_experience(setup)
    )
    assert metrics["teacher_ess"] == pytest.approx(1.0)
    assert metrics["teacher_seed_weight"] == pytest.approx(1.0)


def test_the_reference_pool_is_what_gets_sampled():
    """`rho` is the caller's, not the policy's. A pool concentrated far from
    anything the model generates must still be regressed toward — that is the
    entire point of an *expanding* reference, and the property that separates
    this from a geometric tilt of `p_theta`."""
    setup = _setup(roll_in_size=256, batch_size=64, inner_epochs=30)
    far = DDTensor(torch.full((64, 2), 6.0))
    before = setup.context.policies.train(
        DDTensor(torch.zeros(4, 2)), torch.full((4,), 0.5), conditioning={}
    ).data.mean()
    setup.algorithm.update(
        context=setup.context, experience=_experience(setup, pool=far)
    )
    after = setup.context.policies.train(
        DDTensor(torch.zeros(4, 2)), torch.full((4,), 0.5), conditioning={}
    ).data.mean()
    assert after > before, "the velocity field did not move toward the pool"


def test_probs_select_within_the_pool():
    """A pool whose probability mass is entirely on one half must never train
    on the other. Checked through the loss rather than by inspection: the
    zero-probability half sits far away, so including it would be loud."""
    setup = _setup(roll_in_size=512, batch_size=64, inner_epochs=1)
    pool = DDTensor(torch.cat([torch.zeros(64, 2), torch.full((64, 2), 50.0)]))
    probs = torch.cat([torch.ones(64), torch.zeros(64)])
    experience = dataclasses.replace(
        _experience(setup),
        reference=ReferenceSource(endpoints=pool, conditioning={}, probs=probs),
    )
    metrics = setup.algorithm.update(context=setup.context, experience=experience)
    assert metrics["loss"] < 10.0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_the_rollout_policy_is_refreshed_only_every_lazy_interval():
    """Without laziness the teacher moves every round and the whole timescale
    separation the method rests on disappears; without the refresh ever firing,
    the teacher stays at theta_0 forever and the method is not online."""
    setup = _setup(lazy_interval=3)
    train = setup.context.policies.train
    rollout = setup.context.policies.rollout
    with torch.no_grad():
        for p in train.parameters():
            p.add_(1.0)

    def synced():
        return all(
            torch.equal(a, b)
            for a, b in zip(rollout.state_dict().values(), train.state_dict().values())
        )

    for _ in range(2):
        setup.algorithm.synchronize_rollout_policy(context=setup.context)
        assert not synced()
    setup.algorithm.synchronize_rollout_policy(context=setup.context)
    assert synced()


def test_update_without_a_reference_says_who_owns_it():
    """A bare diffusiongym loop cannot drive this algorithm, and the error has
    to say so rather than fail somewhere inside the teacher."""
    setup = _setup()
    with pytest.raises(ValueError, match="ReferenceSource"):
        setup.algorithm.update(
            context=setup.context, experience=_experience(setup, log_weight=None)
        )


def test_it_never_asks_for_a_reference_policy():
    """`rho` arrives as endpoint samples, never as a density, so a third copy
    of the network would be built for nothing."""
    assert ForwardKLDistillation().requirements.needs_reference_policy is False
    setup = _setup()
    assert setup.context.policies.reference is None


def test_it_requires_stochastic_dynamics():
    """Under an ODE the kernel variance is zero, every teacher proposal is the
    same point, and the tilt silently does nothing."""
    assert ForwardKLDistillation().requirements.needs_stochastic_rollout is True
    assert _setup().dynamics.stochastic


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"num_teacher_proposals": -1}, "num_teacher_proposals"),
        ({"seed_candidate_weight": -0.5}, "seed_candidate_weight"),
        ({"t_min": 0.9, "t_max": 0.5}, "t_min"),
        ({"roll_in_size": 0}, "roll_in_size"),
        ({"inner_epochs": 0}, "inner_epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"lazy_interval": 0}, "lazy_interval"),
    ],
)
def test_constructor_validates_its_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ForwardKLDistillation(**kwargs)
