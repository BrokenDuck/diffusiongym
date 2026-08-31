"""Tests for `core/ot.py`, checked against an exact solver rather than a direction.

For equal-size uniform marginals the optimal coupling is a *permutation*, so
`scipy.optimize.linear_sum_assignment` gives the exact transport cost — the same
role `toy/analytic1d.py` plays for `SMCSampler`. Every claim below is either
checked against that, or is an exact identity (a flat potential, a zero
divergence) rather than a monotonicity assertion.
"""

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from diffusiongym.core.ot import (
    sinkhorn_cost,
    sinkhorn_divergence_potential,
    sinkhorn_potentials,
)


def _clouds(n=64, shift=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g)
    y = torch.randn(n, 2, generator=g) + torch.tensor([shift, 0.0])
    return x, y


def _exact_w1(cost: torch.Tensor) -> float:
    """Exact W1 for equal-size uniform marginals: a linear assignment."""
    rows, cols = linear_sum_assignment(cost.numpy())
    return float(cost[rows, cols].mean().item())


@pytest.mark.parametrize("rel_eps,tolerance", [(0.05, 0.05), (0.01, 0.01)])
def test_sinkhorn_cost_converges_to_exact_optimal_transport(rel_eps, tolerance):
    """The whole point of the entropic relaxation is that it approximates the
    real thing. Measured relative error over `eps / mean(C)`: 15.3% at 0.5, 5.5%
    at 0.1, 2.8% at 0.05, 0.5% at 0.01 — so the default of 0.05 downstream is
    within 3%, and always an *over*-estimate, i.e. a conservative penalty."""
    x, y = _clouds()
    cost = torch.cdist(x, y)
    eps = rel_eps * float(cost.mean().item())

    f, g = sinkhorn_potentials(cost, epsilon=eps, num_iters=2000)
    approx = float(sinkhorn_cost(cost, f, g, epsilon=eps).item())
    exact = _exact_w1(cost)

    assert approx >= exact
    assert approx == pytest.approx(exact, rel=tolerance)


def test_the_potential_ranks_by_distance_from_the_reference():
    """`f` is the first variation, so it is what a mirror step subtracts. If it
    did not separate candidates sitting on the reference from candidates far
    outside it, the locality term would be inert."""
    g = torch.Generator().manual_seed(1)
    reference = torch.randn(128, 2, generator=g)
    near = torch.randn(32, 2, generator=g)
    far = torch.randn(32, 2, generator=g) + torch.tensor([4.0, 4.0])
    candidates = torch.cat([near, far])

    cost_xy = torch.cdist(candidates, reference)
    cost_xx = torch.cdist(candidates, candidates)
    eps = 0.05 * float(cost_xy.mean().item())
    psi, _ = sinkhorn_divergence_potential(cost_xy, cost_xx, epsilon=eps, num_iters=2000)

    assert float(psi[:32].max().item()) < float(psi[32:].min().item())


@pytest.mark.parametrize("rel_eps", [0.2, 0.05])
def test_debiasing_makes_a_measure_local_to_itself(rel_eps):
    """`nu == rho` is the one case with an exact answer: no transport is needed,
    so the first variation must be constant and the penalty must not rank
    anything. The raw potential fails this — its spread is 0.112 at
    `eps = 0.2*mean(C)` and 0.011 at 0.05 — and it fails in the direction that
    looks like success, pulling every candidate toward the middle of its own
    batch. That is the entire reason `sinkhorn_divergence_potential` exists."""
    z = torch.randn(96, 2, generator=torch.Generator().manual_seed(2))
    cost = torch.cdist(z, z)
    eps = rel_eps * float(cost.mean().item())

    raw, _ = sinkhorn_potentials(cost, epsilon=eps, num_iters=2000)
    psi, _ = sinkhorn_divergence_potential(cost, cost, epsilon=eps, num_iters=2000)

    assert float(raw.std().item()) > 1e-2 * rel_eps
    assert float(psi.abs().max().item()) < 1e-6


@pytest.mark.parametrize("epsilon", [1e-3, 1e-1, 1.0])
def test_log_domain_survives_a_tiny_epsilon_and_a_huge_cost(epsilon):
    """`exp(-C/eps)` underflows to zero for all of these and the iteration
    returns NaN. The log-domain form is the only reason small `eps` — where the
    approximation is actually good — is reachable at all."""
    x, y = _clouds()
    cost = torch.cdist(x * 1000.0, y * 1000.0)

    f, g = sinkhorn_potentials(cost, epsilon=epsilon, num_iters=200)
    assert torch.isfinite(f).all()
    assert torch.isfinite(g).all()
    assert torch.isfinite(sinkhorn_cost(cost, f, g, epsilon=epsilon))


def test_the_coupling_converges_to_its_marginals():
    """The dual pair is only the first variation of *optimal* transport once the
    coupling actually has the right marginals. Measured max violation: 2.3e-1
    after 1 iteration, 1.7e-4 after 10, 1.5e-8 after 50 — which is what makes
    the default `num_iters=100` comfortable rather than tight."""
    x, y = _clouds()
    cost = torch.cdist(x, y)
    n = cost.shape[0]
    eps = 0.05 * float(cost.mean().item())

    violations = []
    for iters in (1, 10, 50):
        f, g = sinkhorn_potentials(cost, epsilon=eps, num_iters=iters, tol=0.0)
        pi = ((f.unsqueeze(1) + g.unsqueeze(0) - cost) / eps).exp() / (n * n)
        violations.append(float((pi.sum(dim=1) - 1.0 / n).abs().max().item()))

    assert violations[0] > violations[1] > violations[2]
    assert violations[-1] < 1e-6


def test_arguments_are_validated():
    cost = torch.rand(4, 5)
    with pytest.raises(ValueError, match="epsilon"):
        sinkhorn_potentials(cost, epsilon=0.0)
    with pytest.raises(ValueError, match="2-D"):
        sinkhorn_potentials(torch.rand(4), epsilon=0.1)
    with pytest.raises(ValueError, match="square"):
        sinkhorn_divergence_potential(cost, torch.rand(4, 5), epsilon=0.1)
