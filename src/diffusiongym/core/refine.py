"""Local exploration in the noised representation: partial noising, then denoising.

Given clean seeds `x_1`, forward-noise them to an intermediate time `t_start < 1`
and integrate back to `t = 1` under the model's own SDE. `t_start` is the
mutation radius: `t_start -> 1` barely perturbs the seed, `t_start -> 0` erases
it and this degenerates into an unconditional rollout. `tail_time_grid` turns
that dial.

`local_proposals` draws `L` **independent** such mutations per seed and returns
both ends of each: the noised representation `z` at level `s`, and the endpoint
`x_1` it denoises to. Both are needed by a caller that scores candidates in a
representation space and experiments on endpoints.

**No selection happens here.** An earlier version of this file tilted the
denoising itself, choosing among `L` proposals at *every* step in proportion to
`exp(log_potential(x_hat1))`. That was a greedy local search whose stationary
law was not any specified distribution, and it forced a temperature knob on a
sampler that has no business having an opinion. Selection now belongs to the
caller, applied once to the whole candidate set — which is what makes it an
optimisation step in probability space rather than a per-step heuristic.
"""

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.kernel import EulerGaussianKernelFactory, GaussianMarkovKernel
from diffusiongym.types import DDBatch
from diffusiongym.utils import index_conditioning

type Conditioning = Mapping[str, Any]

__all__ = ["local_proposals", "tail_time_grid"]


def tail_time_grid(time_grid: Tensor, *, noise_frac: float) -> Tensor:
    """The sub-grid a refinement of strength `noise_frac` integrates over.

    `noise_frac` is `K/T` in the usual "noise K of T steps" phrasing: the
    returned grid starts at the first grid point at or after `1 - noise_frac`
    and runs to the end. The requested level is therefore **rounded onto the
    grid**, and `1 - grid[0]` is the level actually applied — log that rather
    than `noise_frac` if the two need to agree.

    `noise_frac = 1` returns the whole grid (a full rollout, seed erased);
    the returned grid always has at least one step, so a `noise_frac` smaller
    than one step still performs a single refinement step.
    """
    if not 0.0 < noise_frac <= 1.0:
        raise ValueError(f"noise_frac must be in (0, 1], got {noise_frac}.")
    target = torch.tensor(
        1.0 - noise_frac, dtype=time_grid.dtype, device=time_grid.device
    )
    index = min(
        int(torch.searchsorted(time_grid, target).item()), time_grid.numel() - 2
    )
    return time_grid[index:]


@torch.no_grad()
def local_proposals[StateT: DDBatch](
    *,
    environment: Any,  # FlowEnvironment (avoid circular import)
    model: Any,  # FlowModel
    dynamics: FlowDynamics[StateT],
    kernel_factory: EulerGaussianKernelFactory[StateT],
    seeds: StateT,
    conditioning: Conditioning,
    time_grid: Tensor,
    num_proposals: int,
    generator: Generator | None = None,
) -> tuple[StateT, StateT, Conditioning]:
    """`L` independent partial-noise/denoise mutations per seed.

    Parameters
    ----------
    seeds : StateT
        `n` clean `x_1` states. They are forward-noised here rather than by the
        caller, so the noising time and the first evaluation time cannot
        disagree.
    time_grid : Tensor
        A tail grid from `tail_time_grid`. Its first point is the noise level
        `s`; the returned `z` lives there.
    num_proposals : int
        `L`. Each of the `n * L` rows draws its own base noise, so the `L`
        copies of one seed genuinely diverge rather than sharing a trajectory.

    Returns
    -------
    (z, x1, conditioning) : tuple[StateT, StateT, Conditioning]
        `z` the noised representation at `time_grid[0]`, `x1` the endpoint it
        reached, and the conditioning tiled to match. **Row `l * n + j` is
        proposal `l` of seed `j`** — `repeat`, never `repeat_interleave`; a
        caller reshaping to `(L, n)` reads that layout and only that one.

    Requires stochastic dynamics. Under `ProbabilityFlowODE` the kernel variance
    is `diffusion**2 * dt` with `diffusion = 0`, so every proposal of a seed
    would follow the identical trajectory from its (differing) `z` and the
    operator would be far less local than the caller asked for — quietly, with
    no error.
    """
    if not dynamics.stochastic:
        raise ValueError(
            "local_proposals requires stochastic dynamics (dynamics.stochastic "
            "must be True). Without transition noise the L proposals of a seed "
            "differ only through their initial noising and never decorrelate. "
            "Use AffineFlowMarginalPreservingSDE."
        )
    if num_proposals < 1:
        raise ValueError(f"num_proposals must be >= 1, got {num_proposals}.")
    if time_grid.numel() < 2:
        raise ValueError(
            f"time_grid needs at least two points, got {time_grid.numel()}."
        )

    device = model.device
    time_grid = time_grid.to(device)
    n = len(seeds)
    total = n * num_proposals

    tile = torch.arange(n, device=device).repeat(num_proposals)
    tiled_seeds = seeds.to(device).index_select(tile)
    cond = index_conditioning(dict(conditioning), tile)

    t_start = time_grid[0].clamp_min(1e-2)
    z = environment.forward_process.make_batch(
        tiled_seeds,
        conditioning=cond,
        t=t_start.expand(total),
        generator=generator,
    ).x_t

    x = z
    for t0, t1 in pairwise(time_grid):
        dt = t1 - t0
        t_eval = t0.clamp_min(1e-2).expand(total)

        velocity = environment.predict_velocity(
            model, x_t=x, t=t_eval, conditioning=cond
        )
        coeffs = dynamics.coefficients(x=x, t=t_eval, velocity=velocity)
        kernel: GaussianMarkovKernel[StateT] = kernel_factory.build(
            x=x,
            t=t_eval,
            dt=dt.unsqueeze(0).expand(total),
            drift=coeffs.drift,
            diffusion=coeffs.diffusion,
        )
        x = kernel.rsample(generator=generator)

    return z.detach(), x.detach(), cond
