"""Visual correctness check for `core/smc.py` against a closed-form ground truth.

Run:

    uv run python examples/check_smc.py              # both figures
    uv run python examples/check_smc.py --figure correctness
    uv run python examples/check_smc.py --figure diagnostics

`toy/analytic1d.py` supplies a flow with an exact velocity field, so an untwisted
SDE rollout lands on a known Gaussian mixture and the exponentially tilted law
``p(x) exp(alpha x) / Z`` is another known mixture. Every dashed reference curve
below is therefore *analytic*, not a second simulation — where SMC and the dashed
line disagree, SMC is wrong.

Two figures, answering two different questions:

  smc_correctness.png  — does the sampler produce the right distribution?
  smc_diagnostics.png  — how do its knobs behave, and where does it break?

Numbers for every panel are also printed to stdout, so the figures can be checked
against exact values rather than read by eye.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from diffusiongym.core import (
    AffineFlowMarginalPreservingSDE,
    DefaultEulerGaussianKernelFactory,
    RectifiedFlowSchedule,
    RolloutRequest,
    ScaledMemorylessDiffusionSchedule,
    SMCSampler,
    TensorGeometry,
)
from diffusiongym.toy.analytic1d import (
    ExactVelocityModel,
    Mixture1D,
    make_environment,
)

FIGURES = Path(__file__).parent / "figures"

# Categorical slots 1-3 of the validated default palette (all-pairs safe in both
# CVD and normal vision at three series); gray carries context, never identity.
SMC = "#2a78d6"  # blue   — what the sampler produced
TARGET = "#eb6834"  # orange — the closed-form answer
THIRD = "#1baf7a"  # aqua   — third series where one is needed
CONTEXT = "#8c8b85"  # the untilted base law, and grids/axes
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"

TWO_MODE = Mixture1D(
    weights=torch.tensor([0.5, 0.5]), means=torch.tensor([-2.0, 2.0]), sigma=0.6
)
THREE_MODE = Mixture1D(
    weights=torch.tensor([0.6, 0.3, 0.1]),
    means=torch.tensor([-3.0, 0.0, 3.0]),
    sigma=0.5,
)


def style_axes(ax, *, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Recessive grid and axes; the marks carry the chart, not the frame."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(CONTEXT)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=CONTEXT, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, pad=9, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def build(mixture: Mixture1D, *, noise_scale: float = 0.7):
    schedule = RectifiedFlowSchedule()
    geometry = TensorGeometry()
    return {
        "environment": make_environment(mixture),
        "model": ExactVelocityModel(mixture),
        "geometry": geometry,
        "kernel_factory": DefaultEulerGaussianKernelFactory(geometry),
        "dynamics": AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ScaledMemorylessDiffusionSchedule(
                schedule, noise_scale
            ),
        ),
    }


def interior_grid(steps: int) -> Tensor:
    """The marginal-preserving drift carries kappa(t) = 1/t, so skip t = 0."""
    return torch.linspace(0.0, 1.0, steps + 1)[1:]


def run_smc(parts, *, alpha, n, steps, seed, potential=None, **sampler_kwargs):
    sampler = SMCSampler(parts["geometry"], parts["kernel_factory"], **sampler_kwargs)
    return sampler.rollout(
        environment=parts["environment"],
        model=parts["model"],
        dynamics=parts["dynamics"],
        n=n,
        conditioning={},
        request=RolloutRequest(time_grid=interior_grid(steps), evaluate_reward=False),
        log_potential=potential or (lambda x1, _t: alpha * x1.data.squeeze(-1)),
        generator=torch.Generator().manual_seed(seed),
    )


def smc_pool(parts, *, alpha, n, steps, seeds, **kw) -> Tensor:
    return torch.cat(
        [
            run_smc(parts, alpha=alpha, n=n, steps=steps, seed=s, **kw)
            .terminal_latent.data.squeeze(-1)
            for s in range(seeds)
        ]
    )


def mixture_cdf(mixture: Mixture1D, x: Tensor) -> Tensor:
    z = (x.unsqueeze(-1) - mixture.means) / (mixture.sigma * math.sqrt(2.0))
    return (mixture.weights * 0.5 * (1.0 + torch.erf(z))).sum(-1)


def ks_distance(samples: Tensor, mixture: Mixture1D) -> float:
    x = samples.sort().values
    n = len(x)
    return float(
        (torch.arange(1, n + 1, dtype=x.dtype) / n - mixture_cdf(mixture, x))
        .abs()
        .max()
    )


def analytic_quantiles(mixture: Mixture1D, probs: Tensor) -> Tensor:
    """Invert the mixture CDF by bisection — no closed form, but exact to 1e-6."""
    lo = torch.full_like(probs, -20.0)
    hi = torch.full_like(probs, 20.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_low = mixture_cdf(mixture, mid) < probs
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Figure 1 — correctness
# ---------------------------------------------------------------------------


def figure_correctness(args) -> Path:
    parts = build(TWO_MODE)
    parts3 = build(THREE_MODE)
    alphas = [0.5, 1.0, 2.0]

    fig = plt.figure(figsize=(13.5, 7.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.30)

    # --- Row 1: terminal density against the closed-form tilt, per alpha ------
    print("\n[terminal law vs closed-form tilt]  two-mode mixture")
    print(f"{'alpha':>6} {'SMC mean':>10} {'exact':>10} {'SMC var':>10} "
          f"{'exact':>10} {'KS':>8}")
    x_plot = torch.linspace(-5.0, 6.0, 900)
    for col, alpha in enumerate(alphas):
        ax = fig.add_subplot(grid[0, col])
        samples = smc_pool(
            parts, alpha=alpha, n=args.n, steps=args.steps, seeds=args.seeds
        )
        target = TWO_MODE.tilt(alpha)

        ax.hist(
            samples.numpy(), bins=110, range=(-5.0, 6.0), density=True,
            color=SMC, alpha=0.5, edgecolor="none", label="SMC samples",
        )
        ax.plot(
            x_plot.numpy(), TWO_MODE.log_density(x_plot).exp().numpy(),
            color=CONTEXT, linewidth=1.6, linestyle=":", label="base law (no tilt)",
        )
        ax.plot(
            x_plot.numpy(), target.log_density(x_plot).exp().numpy(),
            color=TARGET, linewidth=2.0, linestyle="--",
            label="exact tilted target",
        )
        style_axes(
            ax,
            title=f"α = {alpha}",
            xlabel="x" if col == 1 else "",
            ylabel="density" if col == 0 else "",
        )
        # Headroom so the tallest bin never touches the frame or the legend.
        ax.set_ylim(0, 1.22 * float(target.log_density(x_plot).exp().max()))
        if col == 0:
            ax.legend(
                frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left"
            )
        print(f"{alpha:>6} {samples.mean():>10.4f} {target.mean():>10.4f} "
              f"{samples.var():>10.4f} {target.var():>10.4f} "
              f"{ks_distance(samples, target):>8.4f}")

    # --- Quantile residuals --------------------------------------------------
    # A plain Q-Q plot here would be three lines lying on the diagonal, which
    # reads as "correct" at a resolution far coarser than the actual agreement.
    # Plotting the deviation from the diagonal instead spends the whole vertical
    # axis on the error, and separates the three series that would otherwise
    # overlap.
    ax = fig.add_subplot(grid[1, 0])
    probs = torch.linspace(0.01, 0.99, 300)
    print("\n[quantile residuals]  max |SMC quantile - exact quantile| over 1-99%")
    # Stagger the direct labels in x, and push the aqua one below its own
    # curve so it does not land on top of the blue series.
    label_at = {0.5: (0.28, -14), 1.0: (0.55, 10), 2.0: (0.82, 10)}
    for alpha, color in zip(alphas, (THIRD, SMC, TARGET), strict=True):
        samples = smc_pool(
            parts, alpha=alpha, n=args.n, steps=args.steps, seeds=args.seeds
        )
        exact = analytic_quantiles(TWO_MODE.tilt(alpha), probs)
        residual = torch.quantile(samples, probs) - exact
        ax.plot(probs.numpy(), residual.numpy(), color=color, linewidth=2.0)
        # Direct labels rather than a legend alone: aqua sits below 3:1 contrast
        # on this surface, so identity must not rest on the swatch.
        position, dy = label_at[alpha]
        idx = int(position * (len(probs) - 1))
        ax.annotate(
            f"α = {alpha}",
            xy=(float(probs[idx]), float(residual[idx])),
            xytext=(0, dy), textcoords="offset points", ha="center",
            color=color, fontsize=8.5, fontweight="bold",
        )
        print(f"  alpha={alpha}: max |Δ| = {float(residual.abs().max()):.4f}")
    ax.axhline(0.0, color=CONTEXT, linewidth=1.4, linestyle=":")
    ax.set_ylim(-0.2, 0.2)
    ax.text(
        0.5, -0.178, "the dotted line is exact agreement",
        color=INK_2, fontsize=8, ha="center",
    )
    style_axes(
        ax,
        title="Quantile error against the exact target",
        xlabel="quantile level",
        ylabel="SMC quantile − exact quantile",
    )

    # --- Mode masses on the asymmetric three-mode mixture --------------------
    ax = fig.add_subplot(grid[1, 1])
    mass_alphas = [0.0, 0.5, 1.0]
    width = 0.36
    print("\n[mode masses]  three-mode mixture, weights (0.6, 0.3, 0.1)")
    for i, alpha in enumerate(mass_alphas):
        samples = smc_pool(
            parts3, alpha=alpha, n=args.n, steps=args.steps, seeds=args.seeds
        )
        observed = THREE_MODE.mode_masses(samples)
        exact = THREE_MODE.tilt(alpha).weights
        base = torch.arange(3) + i * 4.0
        ax.bar(
            base - width / 2, observed.numpy(), width, color=SMC,
            edgecolor=SURFACE, linewidth=1.0,
            label="SMC" if i == 0 else None,
        )
        ax.bar(
            base + width / 2, exact.numpy(), width, color=TARGET,
            edgecolor=SURFACE, linewidth=1.0,
            label="exact" if i == 0 else None,
        )
        ax.text(
            float(base.float().mean()), -0.115, f"α = {alpha}",
            ha="center", color=INK_2, fontsize=8.5,
        )
        print(f"  alpha={alpha}: SMC   {[round(v, 4) for v in observed.tolist()]}")
        print(f"           exact {[round(v, 4) for v in exact.tolist()]}")
    ax.set_xticks([0, 1, 2, 4, 5, 6, 8, 9, 10])
    ax.set_xticklabels(["-3", "0", "+3"] * 3)
    ax.set_ylim(0, 1.0)
    style_axes(
        ax,
        title="Mode mass reproduced (three-mode mixture)",
        ylabel="share of probability",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # --- KS vs particle count: does the error shrink like Monte Carlo? -------
    ax = fig.add_subplot(grid[1, 2])
    counts = [128, 512, 2048, 8192]
    ks_values = []
    target_15 = TWO_MODE.tilt(1.5)
    print("\n[convergence]  KS distance to the exact tilt, alpha = 1.5")
    for n in counts:
        # Average the per-run KS over independent runs: a single run's KS is
        # itself noisy, and the point of the panel is the rate, not one draw.
        runs = [
            ks_distance(
                run_smc(parts, alpha=1.5, n=n, steps=args.steps, seed=s)
                .terminal_latent.data.squeeze(-1),
                target_15,
            )
            for s in range(8)
        ]
        ks = float(torch.tensor(runs).mean())
        ks_values.append(ks)
        print(f"  n={n:>6}: KS={ks:.4f}   iid reference 1.36/sqrt(n)={1.36/math.sqrt(n):.4f}")
    ax.plot(counts, ks_values, color=SMC, linewidth=2.0, marker="o", markersize=6,
            label="SMC")
    ax.plot(
        counts, [1.36 / math.sqrt(n) for n in counts], color=CONTEXT,
        linewidth=1.4, linestyle=":", label="1.36/√n (iid reference)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(
        ax,
        title="Error shrinks at the Monte-Carlo rate",
        xlabel="particles per run",
        ylabel="KS distance",
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower left")

    fig.suptitle(
        "SMCSampler reproduces the exact exponentially tilted law",
        color=INK, fontsize=14, x=0.008, ha="left", y=0.985,
    )
    fig.text(
        0.008, 0.938,
        "Dashed lines are closed-form, not simulated — the analytic 1-D flow has "
        "an exact velocity field, so any gap is sampler error.",
        color=INK_2, fontsize=9.5, ha="left",
    )
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "smc_correctness.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2 — mechanics and defects
# ---------------------------------------------------------------------------


def figure_diagnostics(args) -> Path:
    parts = build(TWO_MODE)
    alpha = 1.5
    target = TWO_MODE.tilt(alpha)

    fig = plt.figure(figsize=(13.5, 7.6), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.22)

    # --- ESS trace ------------------------------------------------------------
    ax = fig.add_subplot(grid[0, 0])
    print("\n[ESS and resampling]  alpha = 1.5, n = 1024")
    for threshold, color in zip((0.1, 0.5, 0.9), (THIRD, SMC, TARGET), strict=True):
        result = run_smc(
            parts, alpha=alpha, n=1024, steps=args.steps, seed=0,
            ess_threshold=threshold,
        )
        stats = result.smc
        ess = stats.ess_trace / 1024
        steps_axis = torch.arange(len(ess))
        ax.plot(steps_axis.numpy(), ess.numpy(), color=color, linewidth=1.8)
        fired = torch.nonzero(stats.resampled).squeeze(-1)
        ax.scatter(
            fired.numpy(), ess[fired].numpy(), color=color, s=26, zorder=3,
            edgecolor=SURFACE, linewidth=1.0,
        )
        ax.annotate(
            f"threshold {threshold}",
            xy=(len(ess) - 1, float(ess[-1])), xytext=(5, 0),
            textcoords="offset points", color=color, fontsize=8.5,
            fontweight="bold", va="center",
        )
        print(f"  threshold={threshold}: {stats.num_resamples} resamples, "
              f"final ESS/n = {float(ess[-1]):.3f}")
    ax.set_xlim(0, args.steps + 12)
    ax.set_ylim(0, 1.18)
    style_axes(
        ax,
        title="Effective sample size along the rollout (dots = resample fired)",
        xlabel="integration step",
        ylabel="ESS / n",
    )

    # --- Particle diversity ---------------------------------------------------
    ax = fig.add_subplot(grid[0, 1])
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    uniques = []
    print("\n[particle diversity]  distinct terminal particles out of 2048")
    for threshold in thresholds:
        result = run_smc(
            parts, alpha=alpha, n=2048, steps=args.steps, seed=0,
            ess_threshold=threshold,
        )
        u = len(torch.unique(result.terminal_latent.data.squeeze(-1)))
        uniques.append(u)
        print(f"  threshold={threshold}: {u}/2048 distinct")
    bars = ax.bar(
        range(len(thresholds)), uniques, 0.62, color=SMC,
        edgecolor=SURFACE, linewidth=1.5,
    )
    for rect, value in zip(bars, uniques, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2, value + 30, str(value),
            ha="center", color=INK_2, fontsize=8.5,
        )
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([str(t) for t in thresholds])
    ax.set_ylim(0, 2300)
    ax.axhline(2048, color=CONTEXT, linewidth=1.2, linestyle=":")
    ax.text(
        -0.42, 2085, "n = 2048 (no duplication)", color=INK_2, fontsize=8,
    )
    style_axes(
        ax,
        title="Resampling more often keeps particles distinct",
        xlabel="ess_threshold",
        ylabel="distinct terminal particles",
    )

    # --- DEFECT 1: potential_every changes the target ------------------------
    ax = fig.add_subplot(grid[1, 0])
    every_values = [1, 2, 4, 8, 16, 32, 64]
    means, spreads = [], []
    print("\n[DEFECT] terminal mean vs potential_every (target 2.5301, base 0.0000)")
    for every in every_values:
        run_means = [
            float(
                run_smc(
                    parts, alpha=alpha, n=4096, steps=args.steps, seed=s,
                    potential_every=every,
                ).terminal_latent.data.squeeze(-1).mean()
            )
            for s in range(3)
        ]
        t = torch.tensor(run_means)
        means.append(float(t.mean()))
        spreads.append(float(t.std()))
        print(f"  potential_every={every:>3}: mean {means[-1]:>8.4f}")
    ax.axhline(target.mean(), color=TARGET, linewidth=1.8, linestyle="--")
    ax.text(
        1.05, target.mean() + 0.09, "exact tilted target", color=TARGET,
        fontsize=8.5, fontweight="bold",
    )
    ax.axhline(TWO_MODE.mean(), color=CONTEXT, linewidth=1.4, linestyle=":")
    ax.text(
        1.05, TWO_MODE.mean() + 0.09, "untilted base law — no guidance at all",
        color=INK_2, fontsize=8.5,
    )
    ax.errorbar(
        every_values, means, yerr=spreads, color=SMC, linewidth=2.0,
        marker="o", markersize=6, capsize=3, label="SMC",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(every_values)
    ax.set_xticklabels([str(v) for v in every_values])
    ax.set_ylim(-0.55, 3.1)
    style_axes(
        ax,
        title="DEFECT — potential_every silently changes the target law",
        xlabel="potential_every  (64 = one evaluation, at step 0)",
        ylabel="terminal mean",
    )

    # --- DEFECT 2: hard-constraint potential ---------------------------------
    ax = fig.add_subplot(grid[1, 1])

    def constraint(x1, _t):
        inside = x1.data.squeeze(-1) > 1.0
        return torch.where(
            inside,
            torch.zeros_like(inside, dtype=torch.float32),
            torch.full(inside.shape, -float("inf")),
        )

    result = run_smc(
        parts, alpha=alpha, n=8192, steps=args.steps, seed=0, potential=constraint
    )
    samples = result.terminal_latent.data.squeeze(-1)
    satisfied = float((samples > 1.0).float().mean())
    nan_frac = float(torch.isnan(result.smc.ess_trace).float().mean())
    distinct = len(torch.unique(samples))
    print("\n[DEFECT] hard-constraint potential  log phi = 0 if x > 1 else -inf")
    print(f"  fraction of terminal samples satisfying x > 1: {satisfied:.4f}")
    print(f"  fraction of the ESS trace that is NaN:         {nan_frac:.4f}")
    print(f"  distinct terminal particles:                   {distinct}/8192")

    x_plot = torch.linspace(-5.0, 6.0, 900)
    base_curve = TWO_MODE.log_density(x_plot).exp()
    # NaN weights collapse the final resample onto a handful of particles, so
    # the sample histogram carries a spike orders of magnitude off-scale. Clamp
    # to the base law's own height and say so, rather than let one bin flatten
    # the panel into an unreadable baseline.
    ceiling = 1.5 * float(base_curve.max())
    counts, edges = torch.histogram(
        samples, bins=110, range=(-5.0, 6.0), density=True
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(
        centers.numpy(), counts.clamp_max(ceiling).numpy(),
        width=float(edges[1] - edges[0]), color=SMC, alpha=0.5,
        edgecolor="none", label="SMC samples (clipped)",
    )
    ax.plot(
        x_plot.numpy(), base_curve.numpy(),
        color=CONTEXT, linewidth=1.6, linestyle=":", label="untilted base law",
    )
    ax.axvspan(1.0, 6.0, color=TARGET, alpha=0.12, zorder=0)
    ax.axvline(1.0, color=TARGET, linewidth=1.8, linestyle="--")
    ax.set_ylim(0, ceiling * 1.35)
    ax.text(
        5.85, ceiling * 1.29, "feasible region  x > 1", color=TARGET,
        fontsize=8.5, fontweight="bold", ha="right",
    )
    ax.text(
        -4.7, ceiling * 1.18,
        f"{satisfied:.1%} of samples satisfy the constraint\n"
        f"{nan_frac:.0%} of the ESS trace is NaN\n"
        f"only {distinct} of 8192 particles are distinct\nno error raised",
        color=INK, fontsize=9, va="top", linespacing=1.5,
    )
    style_axes(
        ax,
        title="DEFECT — a -inf log-potential is ignored, silently",
        xlabel="x",
        ylabel="density",
    )
    ax.legend(
        frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right",
        bbox_to_anchor=(1.0, 0.72),
    )

    fig.suptitle(
        "SMCSampler mechanics, and two ways it fails without saying so",
        color=INK, fontsize=14, x=0.008, ha="left", y=0.985,
    )
    fig.text(
        0.008, 0.938,
        "Top row: the knobs behave as designed. Bottom row: two configurations "
        "that return plausible samples for the wrong distribution.",
        color=INK_2, fontsize=9.5, ha="left",
    )
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "smc_diagnostics.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure", choices=["correctness", "diagnostics", "both"], default="both"
    )
    parser.add_argument("--n", type=int, default=4096, help="particles per run")
    parser.add_argument("--seeds", type=int, default=4, help="runs pooled per point")
    parser.add_argument("--steps", type=int, default=64, help="integration steps")
    args = parser.parse_args()

    written = []
    if args.figure in ("correctness", "both"):
        written.append(figure_correctness(args))
    if args.figure in ("diagnostics", "both"):
        written.append(figure_diagnostics(args))
    print("\nwrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
