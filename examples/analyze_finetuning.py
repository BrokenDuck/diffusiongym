"""Behaviour analysis for the four fine-tuning algorithms on the 2-D GMM toy.

Where `finetune_2d_gmm.py` produces qualitative sample plots, this script
measures the trade-off the algorithms are actually making. Three modes:

    frontier  E[r] against KL(p_theta || p_ref), multi-seed, plotted against the
              best achievable trade-off. The analytic frontier is exact here, so
              a run sitting below/right of it is spending KL it did not need to.

    maps      Where the probability mass actually moved: the achieved
              log p_theta/p_ref field next to the ideal lambda*r - log Z, plus
              the tilt scatter whose slope is the lambda the method really
              applied and whose r^2 says how much of the density change the
              reward explains at all.

    sweep     One algorithm, one hyperparameter, several values — the effect on
              E[r], KL, achieved lambda and localization.

Usage:
    uv run python examples/analyze_finetuning.py frontier --reward linear
    uv run python examples/analyze_finetuning.py maps --reward bimodal --lam 4
    uv run python examples/analyze_finetuning.py sweep --algo flow_grpo \\
        --param beta_kl --values 0.25 0.5 1.0 2.0

The pretrained model is cached in examples/figures/pretrained.pt and reused.
"""

from __future__ import annotations

import argparse
import copy
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from diffusiongym.core import (
    AffineFlowMarginalPreservingSDE,
    AffineGaussianForwardProcess,
    DefaultEulerGaussianKernelFactory,
    EulerMaruyamaSampler,
    EulerODESampler,
    FlowEnvironment,
    IdentityCodec,
    MemorylessFlowSDE,
    PolicyBundle,
    PredictionConverter,
    ProbabilityFlowODE,
    RectifiedFlowSchedule,
    ScaledMemorylessDiffusionSchedule,
    TensorGeometry,
    VelocityRegression,
)
from diffusiongym.toy.gmm2d import (
    BimodalDifferentiableCost,
    BimodalReward,
    GMMBaseSampler,
    GMMFlowModel,
    LinearDifferentiableCost,
    LinearReward,
    QuadraticDifferentiableCost,
    QuadraticReward,
    RingDifferentiableCost,
    RingReward,
    VelocityMLP,
    analytic_frontier,
    grid_rewards,
    log_density,
    make_density_grid,
    pretrain_velocity_model,
    sample_model,
    tilt_diagnostics,
    tilted_log_density,
)
from diffusiongym.trainers import (
    ORWCFM,
    AdjointMatching,
    DiffusionNFT,
    FineTuningContext,
    FlowGRPO,
)
from diffusiongym.types import DDTensor

DEVICE = torch.device("cpu")
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)
CHECKPOINT = FIGURES / "pretrained.pt"

# Categorical slots 1-4 of the validated palette (see dataviz/references/palette.md).
# Fixed assignment per algorithm: colour follows the entity, never its rank.
ALGO_COLOR = {
    "orwcfm": "#2a78d6",
    "diffusion_nft": "#eb6834",
    "flow_grpo": "#1baf7a",
    "adjoint_matching": "#eda100",
}
ALGO_LABEL = {
    "orwcfm": "ORW-CFM-W2",
    "diffusion_nft": "DiffusionNFT",
    "flow_grpo": "Flow-GRPO",
    "adjoint_matching": "Adjoint Matching",
}
ALGORITHMS = list(ALGO_COLOR)
# Diverging: blue (mass removed) - neutral gray - red (mass added). Never a
# rainbow, and the midpoint must read as "nothing changed".
MASS_CMAP = LinearSegmentedColormap.from_list("mass", ["#2a78d6", "#f0efec", "#d03b3b"])
INK, MUTED, GRIDLINE = "#1a1a19", "#77756f", "#e5e3de"

REWARDS = {
    "linear": (LinearReward, LinearDifferentiableCost),
    "quadratic": (QuadraticReward, QuadraticDifferentiableCost),
    "bimodal": (BimodalReward, BimodalDifferentiableCost),
    "ring": (RingReward, RingDifferentiableCost),
}
# Chosen so the frontier spans a useful KL range for each reward: the bimodal and
# ring rewards are bounded, so lambda=1 barely moves them (KL 0.07 and 0.04).
DEFAULT_LAMBDAS = {
    "linear": [0.5, 1.0, 2.0, 4.0],
    "quadratic": [0.25, 0.5, 1.0, 2.0],
    "bimodal": [2.0, 4.0, 8.0, 16.0],
    "ring": [2.0, 4.0, 8.0, 16.0],
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def load_or_pretrain(steps: int = 20_000) -> torch.nn.Module:
    mlp = VelocityMLP(width=128, depth=3).to(DEVICE)
    if CHECKPOINT.exists():
        mlp.load_state_dict(torch.load(CHECKPOINT))
        return mlp
    print(f"pretraining ({steps} steps), caching to {CHECKPOINT}")
    pretrain_velocity_model(
        GMMFlowModel(mlp, DEVICE), steps=steps, batch_size=512, device=DEVICE
    )
    torch.save(mlp.state_dict(), CHECKPOINT)
    return mlp


def build_algorithm(
    name: str,
    *,
    lam: float,
    std_r: float,
    schedule,
    overrides: dict | None = None,
):
    """Build one algorithm calibrated to effective lambda = `lam`.

    Every algorithm except Adjoint Matching normalizes the reward by std_r
    before applying its knob, so the knob has to be rescaled or the runs target
    different distributions. See CLAUDE.md for the mapping.
    """
    overrides = dict(overrides or {})
    if name == "orwcfm":
        kwargs = {
            "temperature": lam * std_r,
            "alpha_w2": 0.5,
            "rollout_update_interval": 1,
            "steps_per_update": 10,
            "batch_size": 64,
        }
        return ORWCFM(**(kwargs | overrides)), ProbabilityFlowODE(), False, True
    if name == "diffusion_nft":
        kwargs = {"beta": 1.0, "ema_decay": 0.995, "inner_epochs": 5,
                  "batch_size": 64}
        return DiffusionNFT(**(kwargs | overrides)), ProbabilityFlowODE(), False, False
    if name == "flow_grpo":
        kwargs = {
            "group_size": 8,
            "ppo_epochs": 2,
            "ppo_batch_size": 64,
            "beta_kl": 1.0 / (lam * std_r),
        }
        dynamics = AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ScaledMemorylessDiffusionSchedule(schedule, 0.75),
        )
        return FlowGRPO(**(kwargs | overrides)), dynamics, True, True
    if name == "adjoint_matching":
        kwargs = {
            "lambda_reward": lam,
            "train_steps_per_iter": 20,
            "train_batch_size": 64,
        }
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        return AdjointMatching(**(kwargs | overrides)), dynamics, True, True
    raise ValueError(name)


@dataclass
class RunResult:
    model: GMMFlowModel
    ref_model: GMMFlowModel
    reward_fn: object


def run_once(
    algo_name: str,
    reward_name: str,
    *,
    lam: float,
    seed: int,
    iters: int,
    n_rollout: int,
    pretrained: torch.nn.Module,
    time_steps: int = 10,
    overrides: dict | None = None,
) -> RunResult:
    torch.manual_seed(seed)

    def fresh() -> GMMFlowModel:
        return GMMFlowModel(copy.deepcopy(pretrained).to(DEVICE), DEVICE)

    train_model, rollout_model, ref_model = fresh(), fresh(), fresh()
    geometry, schedule = TensorGeometry(), RectifiedFlowSchedule()
    base_sampler = GMMBaseSampler()
    converter = PredictionConverter(geometry=geometry, schedule=schedule)

    reward_cls, cost_cls = REWARDS[reward_name]
    reward_fn, terminal_cost = reward_cls(), cost_cls()

    env = FlowEnvironment(
        geometry=geometry,
        base_sampler=base_sampler,
        forward_process=AffineGaussianForwardProcess(
            geometry=geometry, base_sampler=base_sampler, schedule=schedule
        ),
        regression=VelocityRegression(geometry=geometry, converter=converter),
        codec=IdentityCodec(),
        reward=reward_fn,
        terminal_cost=terminal_cost,
    )

    samples = sample_model(ref_model, n=4000, steps=50, device=DEVICE)
    batch = DDTensor(samples)
    std_r = max(
        reward_fn(sample=batch, latent=batch, conditioning={}).rewards.std().item(),
        1e-6,
    )

    algo, dynamics, interior_grid, needs_ref = build_algorithm(
        algo_name, lam=lam, std_r=std_r, schedule=schedule, overrides=overrides
    )
    ctx = FineTuningContext(
        environment=env,
        policies=PolicyBundle(
            train=train_model,
            rollout=rollout_model,
            reference=ref_model if needs_ref else None,
        ),
        optimizer=torch.optim.Adam(train_model.parameters(), lr=3e-4),
        ode_sampler=EulerODESampler(geometry),
        sde_sampler=EulerMaruyamaSampler(
            geometry, DefaultEulerGaussianKernelFactory(geometry)
        ),
    )
    algo.validate(context=ctx, dynamics=dynamics)

    grid = (
        torch.linspace(0.0, 1.0, time_steps + 2)[1:]
        if interior_grid
        else torch.linspace(0.0, 1.0, time_steps + 1)
    )
    for _ in range(iters):
        exp = algo.collect(
            context=ctx, dynamics=dynamics, n=n_rollout,
            time_grid=grid, conditioning={},
        )
        algo.update(context=ctx, experience=exp)
        algo.synchronize_rollout_policy(context=ctx)

    return RunResult(model=train_model, ref_model=ref_model, reward_fn=reward_fn)


def _style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDLINE)
    ax.tick_params(colors=MUTED, labelsize=9)


# ---------------------------------------------------------------------------
# Mode: frontier
# ---------------------------------------------------------------------------


def mode_frontier(args, pretrained) -> None:
    lambdas = args.lam or DEFAULT_LAMBDAS[args.reward]
    reward_fn = REWARDS[args.reward][0]()
    ref_model = GMMFlowModel(copy.deepcopy(pretrained).to(DEVICE), DEVICE)
    grid = make_density_grid(ref_model, resolution=args.grid_res, steps=args.grid_steps)

    # Extend well past the requested lambdas: a method can overshoot its target
    # tilt and land at a KL far beyond the one it was asked for, and a frontier
    # that stops short would leave those points with nothing to compare against.
    dense = [x / 20.0 for x in range(int(max(lambdas) * 6 * 20) + 1)]
    frontier = analytic_frontier(grid, reward_fn, dense)

    results: dict[str, list[dict]] = {a: [] for a in args.algos}
    for algo_name in args.algos:
        for lam in lambdas:
            per_seed = []
            for seed in range(args.seeds):
                out = run_once(
                    algo_name, args.reward, lam=lam, seed=seed, iters=args.iters,
                    n_rollout=args.rollout_n, pretrained=pretrained,
                    time_steps=args.time_steps,
                )
                d = tilt_diagnostics(
                    out.model, out.ref_model, out.reward_fn,
                    n=args.eval_n, density_steps=args.density_steps,
                )
                per_seed.append(d)
                print(
                    f"  {algo_name:<17} lam={lam:<5g} seed={seed}  "
                    f"E[r]={d.expected_reward:+.3f}  KL={d.kl_to_reference:.3f}  "
                    f"achieved_lam={d.achieved_lambda:+.2f}  r2={d.tilt_r2:.3f}"
                )
            results[algo_name].append(
                {
                    "lam": lam,
                    "r_mean": statistics.mean(d.expected_reward for d in per_seed),
                    "r_sd": _sd([d.expected_reward for d in per_seed]),
                    "kl_mean": statistics.mean(d.kl_to_reference for d in per_seed),
                    "kl_sd": _sd([d.kl_to_reference for d in per_seed]),
                    "lam_hat": statistics.mean(d.achieved_lambda for d in per_seed),
                    "r2": statistics.mean(d.tilt_r2 for d in per_seed),
                }
            )

    max_kl = max(
        r["kl_mean"] + r["kl_sd"] for rows in results.values() for r in rows
    )
    visible = [p for p in frontier if p["kl"] <= max_kl * 1.08]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    _style(ax)
    ax.plot(
        [p["kl"] for p in visible], [p["expected_reward"] for p in visible],
        color=MUTED, linewidth=2, zorder=2, label="best achievable",
    )
    for algo_name in args.algos:
        rows = results[algo_name]
        ax.errorbar(
            [r["kl_mean"] for r in rows], [r["r_mean"] for r in rows],
            xerr=[r["kl_sd"] for r in rows], yerr=[r["r_sd"] for r in rows],
            color=ALGO_COLOR[algo_name], marker="o", markersize=8, linewidth=2,
            capsize=3, elinewidth=1.2, zorder=3, label=ALGO_LABEL[algo_name],
        )
    ax.set_xlabel("KL(p$_\\theta$ ‖ p$_{ref}$)   — cost", color=INK)
    ax.set_ylabel("E[r]   — benefit", color=INK)
    ax.set_title(
        f"Reward–divergence frontier · {args.reward} reward\n"
        f"{args.seeds} seeds, {args.iters} iterations, λ ∈ "
        f"{{{', '.join(str(x) for x in lambdas)}}}",
        color=INK, fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax = axes[1]
    _style(ax)
    lo, hi = min(lambdas) * 0.6, max(lambdas) * 1.6
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=2, zorder=2,
            label="requested = achieved")
    for algo_name in args.algos:
        rows = results[algo_name]
        ax.plot(
            [r["lam"] for r in rows], [r["lam_hat"] for r in rows],
            color=ALGO_COLOR[algo_name], marker="o", markersize=8, linewidth=2,
            zorder=3, label=ALGO_LABEL[algo_name],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{v:g}" for v in lambdas])
    ax.set_xlim(lo, hi)
    ax.minorticks_off()
    ax.set_xlabel("λ requested", color=INK)
    ax.set_ylabel("λ achieved  (slope of log p$_\\theta$/p$_{ref}$ vs r)", color=INK)
    ax.set_title(
        "Did the method apply the tilt it was asked for?\n"
        "off the diagonal = mis-calibrated; only KL-anchored methods can sit on it",
        color=INK, fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout()
    path = FIGURES / f"frontier_{args.reward}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"\nsaved -> {path}")

    print(f"\n{'algorithm':<18}{'λ':>6}{'E[r]':>18}{'KL':>16}{'λ̂':>8}{'r²':>7}")
    for algo_name in args.algos:
        for r in results[algo_name]:
            print(
                f"{ALGO_LABEL[algo_name]:<18}{r['lam']:>6g}"
                f"{r['r_mean']:>11.3f} ±{r['r_sd']:<5.3f}"
                f"{r['kl_mean']:>10.3f} ±{r['kl_sd']:<5.3f}"
                f"{r['lam_hat']:>8.2f}{r['r2']:>7.3f}"
            )


def _sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


# ---------------------------------------------------------------------------
# Mode: maps — where did the probability mass move?
# ---------------------------------------------------------------------------


def mode_maps(args, pretrained) -> None:
    lam = (args.lam or [DEFAULT_LAMBDAS[args.reward][1]])[0]
    reward_fn = REWARDS[args.reward][0]()
    ref_model = GMMFlowModel(copy.deepcopy(pretrained).to(DEVICE), DEVICE)
    grid = make_density_grid(ref_model, resolution=args.grid_res, steps=args.grid_steps)
    rewards_on_grid = grid_rewards(grid, reward_fn)
    ideal = tilted_log_density(grid, rewards_on_grid, lam) - grid.log_p_ref

    # A log-ratio is meaningless where there is no mass to move: outside the
    # support both densities are ~0 and their ratio is numerical noise, which
    # renders as large spurious blobs. Restrict every map to the region that
    # actually carries reference probability.
    support = grid.log_p_ref > (grid.log_p_ref.max() - args.support_nats)
    ideal_masked = torch.where(support, ideal, torch.nan)

    runs = []
    for algo_name in args.algos:
        out = run_once(
            algo_name, args.reward, lam=lam, seed=0, iters=args.iters,
            n_rollout=args.rollout_n, pretrained=pretrained,
            time_steps=args.time_steps,
        )
        d = tilt_diagnostics(
            out.model, out.ref_model, out.reward_fn,
            n=args.eval_n, density_steps=args.density_steps,
        )
        achieved = log_density(
            out.model, grid.points, steps=args.grid_steps
        ) - grid.log_p_ref
        runs.append((algo_name, d, torch.where(support, achieved, torch.nan)))
        print(
            f"  {algo_name:<17} E[r]={d.expected_reward:+.3f}  "
            f"KL={d.kl_to_reference:.3f}  lam_hat={d.achieved_lambda:+.2f}  "
            f"r2={d.tilt_r2:.3f}"
        )

    # Scale every panel to the *ideal* range and let the achieved maps clip.
    # Sharing the achieved range instead would let one extreme cell in the
    # low-density centre wash out every other row, and clipping is itself the
    # useful signal here: saturated colour means the method moved more mass
    # than the reward ever asked for.
    span = float(ideal_masked[support].abs().max().item())
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    MASS_CMAP.set_bad("#f7f6f4")
    ref_image = grid.as_image(grid.log_p_ref).detach().numpy()
    xs_img = grid.as_image(grid.points[:, 0]).detach().numpy()
    ys_img = grid.as_image(grid.points[:, 1]).detach().numpy()
    extent = [-grid.limit, grid.limit, -grid.limit, grid.limit]

    n_algo = len(runs)
    fig, axes = plt.subplots(n_algo, 3, figsize=(13.5, 4.1 * n_algo), squeeze=False)

    for row, (algo_name, d, achieved_masked) in enumerate(runs):
        for col, (field, title) in enumerate(
            (
                (achieved_masked, "achieved  log p$_\\theta$/p$_{ref}$"),
                (ideal_masked, f"ideal  $\\lambda$r $-$ log Z   ($\\lambda$={lam:g})"),
            )
        ):
            ax = axes[row][col]
            im = ax.imshow(
                grid.as_image(field).detach().numpy(), origin="lower",
                extent=extent, cmap=MASS_CMAP, norm=norm,
            )
            # Reference density outline, so "where the mass is" stays visible.
            ax.contour(
                xs_img, ys_img, ref_image, levels=4,
                colors=MUTED, linewidths=0.6, alpha=0.5,
            )
            ax.tick_params(colors=MUTED, labelsize=9)
            if col == 0:
                ax.set_ylabel(ALGO_LABEL[algo_name], color=INK, fontsize=11)
            if row == 0:
                ax.set_title(title, color=INK, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[row][2]
        _style(ax)
        ax.scatter(
            d.rewards.numpy(), d.log_ratio.numpy(), s=9, alpha=0.3,
            color=ALGO_COLOR[algo_name], edgecolors="none",
        )
        # Both lines drawn through the same centroid, so only the slopes differ.
        xs = torch.linspace(d.rewards.min(), d.rewards.max(), 2)
        mean_r, mean_l = d.rewards.mean(), d.log_ratio.mean()
        for slope, colour, label in (
            (lam, MUTED, f"ideal slope $\\lambda$={lam:g}"),
            (d.achieved_lambda, ALGO_COLOR[algo_name],
             f"fit $\\hat\\lambda$={d.achieved_lambda:.2f}, r$^2$={d.tilt_r2:.2f}"),
        ):
            ax.plot(xs.numpy(), (mean_l + slope * (xs - mean_r)).numpy(),
                    color=colour, linewidth=2, label=label)
        ax.set_xlabel("r(x)", color=INK)
        if row == 0:
            ax.set_title("tilt is reward-shaped?", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="best")

    fig.suptitle(
        f"Where fine-tuning moved probability mass · {args.reward} reward, "
        f"$\\lambda$={lam:g}\n"
        "red = mass added, blue = mass removed, gray = unchanged; blank = outside "
        "the reference support; colour clipped to the ideal range",
        color=INK, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = FIGURES / f"mass_maps_{args.reward}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"\nsaved -> {path}")



# ---------------------------------------------------------------------------
# Mode: sweep — effect of one hyperparameter
# ---------------------------------------------------------------------------


def mode_sweep(args, pretrained) -> None:
    lam = (args.lam or [DEFAULT_LAMBDAS[args.reward][1]])[0]
    values = [_coerce(v) for v in args.values]
    rows = []
    for value in values:
        per_seed = []
        for seed in range(args.seeds):
            # `time_steps` is a rollout property, not a constructor argument, so
            # it is swept through run_once rather than through overrides.
            time_steps = value if args.param == "time_steps" else args.time_steps
            overrides = {} if args.param == "time_steps" else {args.param: value}
            out = run_once(
                args.algo, args.reward, lam=lam, seed=seed, iters=args.iters,
                n_rollout=args.rollout_n, pretrained=pretrained,
                time_steps=time_steps, overrides=overrides,
            )
            per_seed.append(
                tilt_diagnostics(
                    out.model, out.ref_model, out.reward_fn,
                    n=args.eval_n, density_steps=args.density_steps,
                )
            )
        row = {
            "value": value,
            "r_mean": statistics.mean(d.expected_reward for d in per_seed),
            "r_sd": _sd([d.expected_reward for d in per_seed]),
            "kl_mean": statistics.mean(d.kl_to_reference for d in per_seed),
            "kl_sd": _sd([d.kl_to_reference for d in per_seed]),
            "lam_hat": statistics.mean(d.achieved_lambda for d in per_seed),
            "r2": statistics.mean(d.tilt_r2 for d in per_seed),
        }
        rows.append(row)
        print(
            f"  {args.param}={value!s:<8} E[r]={row['r_mean']:+.3f}  "
            f"KL={row['kl_mean']:.3f}  λ̂={row['lam_hat']:+.2f}  r²={row['r2']:.3f}"
        )

    # Small multiples, one panel per measure — never two y-scales on one axis.
    panels = [
        ("E[r]", "r_mean", "r_sd"),
        ("KL(p$_\\theta$ ‖ p$_{ref}$)", "kl_mean", "kl_sd"),
        ("λ achieved", "lam_hat", None),
        ("r²  (tilt explained by reward)", "r2", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    xs = list(range(len(values)))
    for ax, (label, key, err) in zip(axes, panels, strict=True):
        _style(ax)
        ax.errorbar(
            xs, [r[key] for r in rows],
            yerr=[r[err] for r in rows] if err else None,
            color=ALGO_COLOR[args.algo], marker="o", markersize=8, linewidth=2,
            capsize=3, elinewidth=1.2,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in values])
        ax.set_xlabel(args.param, color=INK)
        ax.set_title(label, color=INK, fontsize=11)
    fig.suptitle(
        f"{ALGO_LABEL[args.algo]} · effect of {args.param} · {args.reward} reward, "
        f"λ={lam:g}, {args.seeds} seeds",
        color=INK, fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = FIGURES / f"sweep_{args.algo}_{args.param}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"\nsaved -> {path}")


def _coerce(text: str):
    for cast in (int, float):
        try:
            value = cast(text)
        except ValueError:
            continue
        if cast is int and text.strip().lstrip("-").isdigit():
            return value
        if cast is float:
            return value
    return text


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["frontier", "maps", "sweep"])
    parser.add_argument("--reward", choices=list(REWARDS), default="linear")
    parser.add_argument("--algos", nargs="+", default=ALGORITHMS)
    parser.add_argument("--algo", default="flow_grpo", help="sweep mode only")
    parser.add_argument("--param", default="beta_kl", help="sweep mode only")
    parser.add_argument("--values", nargs="+", default=[], help="sweep mode only")
    parser.add_argument("--lam", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--rollout-n", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--eval-n", type=int, default=1500)
    parser.add_argument("--density-steps", type=int, default=120)
    parser.add_argument("--grid-res", type=int, default=90)
    parser.add_argument("--grid-steps", type=int, default=80)
    parser.add_argument("--support-nats", type=float, default=10.0,
                        help="mask map cells whose reference density is\n"
                             "more than this many nats below the peak")
    parser.add_argument("--pretrain-steps", type=int, default=20_000)
    args = parser.parse_args()

    pretrained = load_or_pretrain(args.pretrain_steps)
    print(f"\n=== {args.mode} · {args.reward} reward ===")
    if args.mode == "frontier":
        mode_frontier(args, pretrained)
    elif args.mode == "maps":
        mode_maps(args, pretrained)
    else:
        if not args.values:
            parser.error("sweep mode needs --values")
        mode_sweep(args, pretrained)


if __name__ == "__main__":
    main()
