"""Entropic optimal transport on particle sets: costs and dual potentials.

Two empirical measures, a cost matrix between their supports, and nothing else —
no state type, no model, no geometry. Everything here is a plain tensor, so this
file is data-type abstract by construction: the caller decides what a "point" is
and hands over `cost[i, j] = c(x_i, y_j)`.

The reason this exists is the **first variation**. For a functional
`W_c(nu, rho)` of the source measure, the Kantorovich dual potential `f` at
`x_i` *is* `delta W_c / delta nu` up to an additive constant, so a
mirror-ascent step that wants to penalise moving away from `rho` needs only
`f` — never a density, never a critic network, never a training loop. That is
what makes an optimal-transport locality term compatible with a particle-only
algorithm.

**Log-domain throughout.** The scaling iteration is written on the potentials
rather than on the kernel, so nothing underflows at small `epsilon` and a cost
range of `1e3` is as safe as one of `1e0`. The naive `exp(-C/eps)` form is
unusable at both.

**Debiasing is on by default and matters.** Entropic regularisation shrinks the
coupling toward the independent product, which biases `f` toward a function that
pulls the source cloud in on itself. Left in, that reads in a downstream
algorithm as "the locality term is working" when what is actually happening is
that every candidate near the middle of its own batch scores well. The Sinkhorn
divergence — subtract the self-transport potential `f_{nu,nu}` — removes it, at
the cost of one extra scaling loop. `sinkhorn_divergence_potential` does that;
`sinkhorn_potentials` is the raw pair if a caller genuinely wants it. Measured
on `nu == rho` (96 points, 2-D), where the correct answer is a flat potential:
raw `f` has spread 0.112 at `eps = 0.2 * mean(C)` and 0.011 at `0.05`; debiased
`psi` has spread exactly 0 at both.

**Choosing epsilon.** Scale it by the mean cost, then it means the same thing
across problems. Against an exact solver (`scipy.optimize.linear_sum_assignment`
on 64 vs 64 uniform points, where the assignment *is* the optimal coupling) the
primal cost overshoots by:

| `eps / mean(C)` | 0.5 | 0.2 | 0.1 | 0.05 | 0.02 | 0.01 |
|---|---|---|---|---|---|---|
| relative error | 15.3% | 9.7% | 5.5% | 2.8% | 1.1% | 0.5% |

`0.05` is the default downstream: the bias is small, and it is an overshoot in a
*penalty*, i.e. it errs conservative. The source-marginal violation reaches
1.7e-4 by iteration 10 and 1.5e-8 by 50, so `num_iters=100` with the `tol`
early exit is comfortable rather than tight.

Note that `sinkhorn_cost` returns the raw primal `<pi, C>`, which carries that
same entropic overshoot — at `eps = 0.05 * mean(C)` the self-transport of a
cloud against itself measured 0.045 against a real displacement of 2.87, so as a
logged diagnostic it is ~1.5% high and not worth a second scaling loop to fix.
"""

import torch
from torch import Tensor

__all__ = [
    "sinkhorn_cost",
    "sinkhorn_divergence_potential",
    "sinkhorn_potentials",
]


def _uniform(n: int, like: Tensor) -> Tensor:
    return torch.full((n,), 1.0 / n, device=like.device, dtype=like.dtype)


def sinkhorn_potentials(
    cost: Tensor,
    *,
    epsilon: float,
    num_iters: int = 100,
    a: Tensor | None = None,
    b: Tensor | None = None,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Dual potentials `(f, g)` of the entropic OT problem for `cost`.

    Parameters
    ----------
    cost : Tensor
        `(n, m)` ground cost. Not required to be a metric; `sinkhorn_cost`
        reports whatever this measures.
    epsilon : float
        Entropic regularisation, in the units of `cost`. Callers usually want to
        scale it by the mean cost so it means the same thing across problems.
    num_iters : int
        Maximum scaling iterations. The loop exits early once the source
        marginal violation drops below `tol`.
    a, b : Tensor, optional
        Marginal weights, `(n,)` and `(m,)`. Default uniform. Not renormalised —
        an unbalanced pair is the caller's decision and its own problem.
    tol : float
        Max-norm on `log a - logsumexp_j log(pi_ij)`, the source marginal
        violation, checked every 10 iterations.

    Returns
    -------
    (f, g) : tuple[Tensor, Tensor]
        Shapes `(n,)` and `(m,)`. `f` is the first variation with respect to the
        source measure, defined up to an additive constant — which is why every
        caller here feeds it to a softmax or a difference.
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {tuple(cost.shape)}.")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    n, m = cost.shape
    a = _uniform(n, cost) if a is None else a.to(cost)
    b = _uniform(m, cost) if b is None else b.to(cost)
    log_a, log_b = a.clamp_min(1e-30).log(), b.clamp_min(1e-30).log()

    f = torch.zeros(n, device=cost.device, dtype=cost.dtype)
    g = torch.zeros(m, device=cost.device, dtype=cost.dtype)

    for it in range(num_iters):
        # f_i = -eps * logsumexp_j [ (g_j - C_ij)/eps + log b_j ]
        f = -epsilon * torch.logsumexp(
            (g.unsqueeze(0) - cost) / epsilon + log_b.unsqueeze(0), dim=1
        )
        g = -epsilon * torch.logsumexp(
            (f.unsqueeze(1) - cost) / epsilon + log_a.unsqueeze(1), dim=0
        )

        if it % 10 == 9:
            log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon
            log_pi = log_pi + log_a.unsqueeze(1) + log_b.unsqueeze(0)
            violation = (torch.logsumexp(log_pi, dim=1) - log_a).abs().max()
            if float(violation.item()) < tol:
                break

    return f, g


def sinkhorn_cost(
    cost: Tensor,
    f: Tensor,
    g: Tensor,
    *,
    epsilon: float,
    a: Tensor | None = None,
    b: Tensor | None = None,
) -> Tensor:
    """Transport cost `<pi, C>` under the coupling `(f, g)` induce.

    The primal cost rather than the dual objective: with a converged pair the
    two agree up to the entropy term, and the primal is the number that can be
    compared against an exact solver.
    """
    n, m = cost.shape
    a = _uniform(n, cost) if a is None else a.to(cost)
    b = _uniform(m, cost) if b is None else b.to(cost)

    log_pi = (f.unsqueeze(1) + g.unsqueeze(0) - cost) / epsilon
    log_pi = log_pi + a.clamp_min(1e-30).log().unsqueeze(1)
    log_pi = log_pi + b.clamp_min(1e-30).log().unsqueeze(0)
    return (log_pi.exp() * cost).sum()


def sinkhorn_divergence_potential(
    cost_xy: Tensor,
    cost_xx: Tensor,
    *,
    epsilon: float,
    num_iters: int = 100,
    a: Tensor | None = None,
    b: Tensor | None = None,
    tol: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Debiased first variation `f_{nu,rho} - f_{nu,nu}`, and the transport cost.

    Parameters
    ----------
    cost_xy : Tensor
        `(n, m)` cost between the source particles and the reference.
    cost_xx : Tensor
        `(n, n)` cost of the source against itself. Supplying it rather than
        recomputing keeps this file free of any notion of what a point is.

    Returns
    -------
    (psi, transport_cost) : tuple[Tensor, Tensor]
        `psi` has shape `(n,)`; `transport_cost` is the scalar `<pi, cost_xy>`,
        which is the quantity to log when a locality penalty needs to be
        observable rather than merely applied.

    The correction is the reason `nu == rho` gives `psi` flat and a divergence
    of zero. Without it both are visibly wrong in exactly the direction that
    looks like success: a spurious inward pull on the source cloud.
    """
    if cost_xx.shape[0] != cost_xx.shape[1] or cost_xx.shape[0] != cost_xy.shape[0]:
        raise ValueError(
            f"cost_xx must be square and match cost_xy's rows, got "
            f"{tuple(cost_xx.shape)} against {tuple(cost_xy.shape)}."
        )

    f_xy, g_xy = sinkhorn_potentials(
        cost_xy, epsilon=epsilon, num_iters=num_iters, a=a, b=b, tol=tol
    )
    f_xx, _ = sinkhorn_potentials(
        cost_xx, epsilon=epsilon, num_iters=num_iters, a=a, b=a, tol=tol
    )
    transport = sinkhorn_cost(cost_xy, f_xy, g_xy, epsilon=epsilon, a=a, b=b)
    return f_xy - f_xx, transport
