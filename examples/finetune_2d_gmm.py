"""Fine-tuning demo on the 2-D four-Gaussian mixture toy problem.

Runs all four algorithms (ORW-CFM, DiffusionNFT, Flow-GRPO, Adjoint Matching)
on the three reward families (linear, quadratic, box) and produces a grid of
visualisation figures.

Usage:
    uv run python examples/finetune_2d_gmm.py
    uv run python examples/finetune_2d_gmm.py --reward linear --algo orwcfm --iters 30
    uv run python examples/finetune_2d_gmm.py --pretrain-steps 5000 --no-show

Outputs:  examples/figures/<reward>_<algo>_samples.png
"""

import argparse
import copy
import math
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    SIGMA_DATA,
    BoxReward,
    GMMBaseSampler,
    GMMFlowModel,
    LinearDifferentiableCost,
    LinearReward,
    QuadraticDifferentiableCost,
    QuadraticReward,
    TiltedMixture,
    VelocityMLP,
    compute_metrics,
    exact_velocity,
    pretrain_velocity_model,
    sample_model,
    tilted_target_linear,
    tilted_target_quadratic,
)
from diffusiongym.trainers import (
    ORWCFM,
    AdjointMatching,
    DiffusionNFT,
    FineTuningContext,
    FlowGRPO,
)
from diffusiongym.types import DDTensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------


def build_env(
    model: GMMFlowModel,
    reward_fn,
    terminal_cost=None,
    geometry=None,
    schedule=None,
    base_sampler=None,
) -> FlowEnvironment:
    if geometry is None:
        geometry = TensorGeometry()
    if schedule is None:
        schedule = RectifiedFlowSchedule()
    if base_sampler is None:
        base_sampler = GMMBaseSampler()

    converter = PredictionConverter(geometry=geometry, schedule=schedule)
    regression = VelocityRegression(geometry=geometry, converter=converter)
    forward_process = AffineGaussianForwardProcess(
        geometry=geometry, base_sampler=base_sampler, schedule=schedule
    )
    return FlowEnvironment(
        geometry=geometry,
        base_sampler=base_sampler,
        forward_process=forward_process,
        regression=regression,
        codec=IdentityCodec(),
        reward=reward_fn,
        terminal_cost=terminal_cost,
    )


def build_context(
    env: FlowEnvironment,
    train_model: GMMFlowModel,
    rollout_model: GMMFlowModel,
    ref_model: GMMFlowModel | None = None,
    lr: float = 3e-4,
    geometry=None,
) -> FineTuningContext:
    if geometry is None:
        geometry = env.geometry
    bundle = PolicyBundle(
        train=train_model,
        rollout=rollout_model,
        reference=ref_model,
    )
    opt = torch.optim.Adam(train_model.parameters(), lr=lr)
    kernel_factory = DefaultEulerGaussianKernelFactory(geometry)
    ode = EulerODESampler(geometry)
    sde = EulerMaruyamaSampler(geometry, kernel_factory)
    return FineTuningContext(
        environment=env,
        policies=bundle,
        optimizer=opt,
        ode_sampler=ode,
        sde_sampler=sde,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def estimate_reward_std(
    model: GMMFlowModel,
    reward_fn,
    *,
    n: int = 4000,
) -> float:
    """Standard deviation of the reward under the base policy.

    Three of the four algorithms normalize the reward by its own spread before
    applying their knob, so their effective tilt is knob / std_r (ORW-CFM) or
    1 / (beta_kl * std_r) (Flow-GRPO). Adjoint Matching does not normalize at
    all. Comparing the four at a fixed knob therefore compares different target
    distributions unless std_r is divided back out — and std_r differs by 3x
    between the linear and quadratic rewards on this problem.
    """
    samples = sample_model(model, n=n, steps=50, device=DEVICE)
    batch = DDTensor(samples)
    rewards = reward_fn(sample=batch, latent=batch, conditioning={}).rewards
    return max(rewards.std().item(), 1e-6)


def run_finetune(
    algo_name: str,
    reward_name: str,
    pretrained_mlp: torch.nn.Module,
    *,
    n_iter: int = 40,
    n_rollout: int = 64,
    lam: float = 1.0,
    time_steps: int = 10,
    verbose: bool = True,
) -> tuple[GMMFlowModel, list[dict]]:
    """Run fine-tuning for one (algo, reward) pair.

    Returns the fine-tuned model and a list of per-iteration metrics.
    """
    device = DEVICE

    # Fresh copy of the pretrained MLP for every run
    def _fresh() -> GMMFlowModel:
        mlp = copy.deepcopy(pretrained_mlp).to(device)
        return GMMFlowModel(mlp, device)

    train_model = _fresh()
    rollout_model = _fresh()
    ref_model = _fresh()

    geometry = TensorGeometry()
    schedule = RectifiedFlowSchedule()
    base_sampler = GMMBaseSampler()

    # Reward
    if reward_name == "linear":
        reward_fn = LinearReward()
        terminal_cost = LinearDifferentiableCost()
    elif reward_name == "quadratic":
        reward_fn = QuadraticReward()
        terminal_cost = QuadraticDifferentiableCost()
    elif reward_name == "box":
        reward_fn = BoxReward()
        terminal_cost = None  # not differentiable
    else:
        raise ValueError(reward_name)

    env = build_env(
        train_model,
        reward_fn,
        terminal_cost=terminal_cost,
        geometry=geometry,
        schedule=schedule,
        base_sampler=base_sampler,
    )

    # Only Adjoint Matching applies the reward on its own scale; every other
    # algorithm normalizes it by std_r first, so their knobs must be rescaled or
    # the four runs silently target different tilts (see calibrate_knobs).
    std_r = estimate_reward_std(ref_model, reward_fn)

    # Algorithm
    lam_eff = "unknown"
    if algo_name == "orwcfm":
        # weights = exp(temperature * r_norm) => lambda_eff = temperature / std_r
        # alpha_w2 is a W2 *fidelity* knob, not an inverse temperature: without
        # it the online iteration reweights its own samples every round and
        # diverges (measured: E[r] runs to 11.9 against an analytic 2.05, with
        # all mass on one mode). It also means ORW-CFM-W2 targets a
        # W2-regularized optimum, not the KL-tilted p* the other methods aim at.
        algo = ORWCFM(
            temperature=lam * std_r,
            alpha_w2=0.5,
            rollout_update_interval=1,
            steps_per_update=10,
            batch_size=64,
        )
        lam_eff = f"{lam:.2f} on the reward term (W2-regularized, not KL)"
        dynamics = ProbabilityFlowODE()
        ctx = build_context(
            env, train_model, rollout_model, ref_model=ref_model, geometry=geometry
        )

    elif algo_name == "diffusion_nft":
        # No calibration is possible: the optimality probability is
        # r = 0.5 + 0.5 * clamp(r_norm, -1, 1), a *linear* (and saturating)
        # function of the reward, so beta is a step size in velocity space and
        # not an inverse temperature. There is no lam it can be matched to.
        algo = DiffusionNFT(beta=1.0, ema_decay=0.995, inner_epochs=5, batch_size=64)
        lam_eff = "n/a (saturating linear tilt, no inverse temperature)"
        dynamics = ProbabilityFlowODE()
        ctx = build_context(env, train_model, rollout_model, geometry=geometry)

    elif algo_name == "flow_grpo":
        # beta_kl sets the target tilt: the stationary policy is
        # p* ∝ p_ref · exp(Â / beta_kl) with Â = (r - mean) / std_r, so the
        # effective tilt is lambda_eff = 1 / (beta_kl * std_r). Calibrating off
        # std_r rather than hard-coding beta_kl matters: std_r is 2.0 for the
        # linear reward but 6.4 for the quadratic one, so a fixed beta_kl = 0.5
        # would target lambda = 1 on the former and lambda = 0.31 on the latter.
        # Note std_r keeps shrinking as the policy concentrates, so this pins the
        # tilt at iteration 0 only.
        algo = FlowGRPO(
            group_size=8,
            ppo_epochs=2,
            ppo_batch_size=64,
            beta_kl=1.0 / (lam * std_r),
        )
        lam_eff = f"{lam:.2f} at iteration 0, rising as std_r shrinks"
        # Flow-GRPO does not need memorylessness (that is an Adjoint Matching
        # requirement); a = sqrt(2)*0.75 keeps the 1/t stiffness of the drift
        # manageable while leaving enough exploration noise.
        dynamics = AffineFlowMarginalPreservingSDE(
            affine_schedule=schedule,
            diffusion_schedule=ScaledMemorylessDiffusionSchedule(schedule, 0.75),
        )
        ctx = build_context(
            env, train_model, rollout_model, ref_model=ref_model, geometry=geometry
        )

    elif algo_name == "adjoint_matching":
        if terminal_cost is None:
            print(
                f"  Skipping adjoint_matching for {reward_name} (needs differentiable cost)"
            )
            return train_model, []
        # Adjoint Matching carries no advantage normalization: the tilt is set
        # directly by the reward scale, g = -lambda_reward * r, so lam is the
        # same lambda the analytic target and ORW-CFM's temperature use.
        algo = AdjointMatching(
            lambda_reward=lam, train_steps_per_iter=20, train_batch_size=64
        )
        lam_eff = f"{lam:.2f} (exact; no reward normalization)"
        dynamics = MemorylessFlowSDE(affine_schedule=schedule)
        ctx = build_context(
            env, train_model, rollout_model, ref_model=ref_model, geometry=geometry
        )

    else:
        raise ValueError(algo_name)

    algo.validate(context=ctx, dynamics=dynamics)

    if verbose:
        print(f"  std_r={std_r:.3f}  ->  effective lambda = {lam_eff}")

    if algo_name in ("flow_grpo", "adjoint_matching"):
        # Drop the singular t=0 node: the marginal-preserving drift carries a
        # kappa(t) = 1/t term, so an Euler-Maruyama step at t≈0 with dt=1/T is
        # expansive (the mean map becomes x ↦ -9x for T=10) and the trajectory
        # leaves the data manifold entirely. Steps of size 1/(T+1) starting at
        # t=1/(T+1) give dt/t <= 1 everywhere. Both algorithms check this in
        # collect(). Adjoint Matching is the more sensitive of the two: it is
        # pinned to the memoryless schedule, the noisiest member of the family.
        # Adjoint Matching integrates the lean adjoint backward with explicit
        # Euler, so its error accumulates over the trajectory and shows up as an
        # under-applied tilt rather than as instability: at lambda = 2 the
        # achieved tilt was 0.45 at 10 steps and 1.84 at 40. Give it a finer grid
        # than the algorithms that only need the rollout to be stable.
        steps = time_steps * 4 if algo_name == "adjoint_matching" else time_steps
        time_grid = torch.linspace(0.0, 1.0, steps + 2, device=device)[1:]
    else:
        time_grid = torch.linspace(0.0, 1.0, time_steps + 1, device=device)
    history = []

    for i in range(n_iter):
        exp = algo.collect(
            context=ctx,
            dynamics=dynamics,
            n=n_rollout,
            time_grid=time_grid,
            conditioning={},
        )
        metrics = algo.update(context=ctx, experience=exp)
        algo.synchronize_rollout_policy(context=ctx)

        if verbose and (i + 1) % max(1, n_iter // 10) == 0:
            parts = [f"iter {i + 1:3d}/{n_iter}"]
            for k, v in metrics.items():
                parts.append(f"{k}={v:.4f}")
            print("  " + "  ".join(parts))

        history.append(metrics)

    return train_model, history


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _grid_density(
    means: torch.Tensor,
    weights: torch.Tensor,
    covs: torch.Tensor | None = None,
    *,
    xlim: tuple[float, float] = (-5, 5),
    ylim: tuple[float, float] = (-5, 5),
    res: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(*xlim, res)
    ys = np.linspace(*ylim, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = torch.from_numpy(np.stack([XX.ravel(), YY.ravel()], axis=1)).float()

    Z = torch.zeros(pts.shape[0])
    for k in range(means.shape[0]):
        mu = means[k]
        if covs is not None:
            cov_k = covs[k]
        else:
            cov_k = (SIGMA_DATA**2) * torch.eye(2)
        diff = pts - mu.unsqueeze(0)
        cov_inv = torch.linalg.inv(cov_k)
        maha = (diff @ cov_inv * diff).sum(-1)
        log_det = torch.linalg.slogdet(cov_k)[1]
        log_p = -0.5 * (maha + log_det + 2 * math.log(2 * math.pi))
        Z += weights[k] * torch.exp(log_p)

    return XX, YY, Z.reshape(res, res).numpy()


def visualize(
    base_samples: torch.Tensor,
    finetuned_samples: torch.Tensor,
    target: TiltedMixture | None,
    *,
    title: str = "",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xlim = ylim = (-5.5, 5.5)
    kw = dict(alpha=0.35, s=8)

    # Col 1: base model samples
    ax = axes[0]
    ax.scatter(base_samples[:, 0], base_samples[:, 1], c="steelblue", **kw)
    ax.set_title("Base model samples")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    # Col 2: analytic target density
    ax = axes[1]
    if target is not None:
        XX, YY, Z = _grid_density(target.means, target.weights, target.covs)
        ax.contourf(XX, YY, Z, levels=20, cmap="Reds", alpha=0.7)
        ax.contour(XX, YY, Z, levels=10, colors="darkred", linewidths=0.5)
    ax.set_title("Analytic target p*")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    # Col 3: fine-tuned model samples
    ax = axes[2]
    ax.scatter(finetuned_samples[:, 0], finetuned_samples[:, 1], c="tomato", **kw)
    if target is not None:
        XX, YY, Z = _grid_density(target.means, target.weights, target.covs)
        ax.contour(XX, YY, Z, levels=8, colors="darkred", linewidths=0.8, alpha=0.6)
    ax.set_title("Fine-tuned samples")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def visualize_velocity_field(
    model: GMMFlowModel,
    t_values: list[float] = [0.25, 0.5, 0.75],
    *,
    title_prefix: str = "",
    save_path: str | None = None,
    show: bool = False,
    weights: torch.Tensor | None = None,
    means: torch.Tensor | None = None,
) -> None:
    """Compare analytic vs. learned velocity fields at fixed times."""
    fig, axes = plt.subplots(len(t_values), 3, figsize=(15, 5 * len(t_values)))
    if len(t_values) == 1:
        axes = axes[None, :]

    xs = torch.linspace(-4, 4, 15)
    ys = torch.linspace(-4, 4, 15)
    XX, YY = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1).to(DEVICE)

    for row, t_val in enumerate(t_values):
        t_tensor = torch.full((pts.shape[0],), t_val, device=DEVICE)

        # Analytic velocity
        with torch.no_grad():
            u_true = exact_velocity(pts, t_tensor, weights=weights, means=means)
            u_pred = model._mlp(pts, t_tensor)

        err = (u_pred - u_true).norm(dim=-1).reshape(15, 15)

        def _quiver(ax, U, color, ttl):
            Ux = U[:, 0].reshape(15, 15).cpu().numpy()
            Uy = U[:, 1].reshape(15, 15).cpu().numpy()
            ax.quiver(XX.numpy(), YY.numpy(), Ux, Uy, alpha=0.8, color=color, scale=30)
            ax.set_title(ttl)
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_aspect("equal")

        _quiver(axes[row, 0], u_true, "steelblue", f"Analytic  t={t_val}")
        _quiver(axes[row, 1], u_pred.detach(), "tomato", f"Learned  t={t_val}")

        # Error heatmap
        ax = axes[row, 2]
        im = ax.imshow(
            err.detach().cpu().numpy(),
            origin="lower",
            extent=[-4, 4, -4, 4],
            cmap="hot_r",
            vmin=0,
        )
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax.set_title(f"||u_θ - u*||  t={t_val}")

    fig.suptitle(f"{title_prefix} velocity field", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_training_curves(
    history: list[dict],
    *,
    title: str = "",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
        vals = [m.get(key, float("nan")) for m in history]
        ax.plot(vals)
        ax.set_xlabel("iteration")
        ax.set_ylabel(key)
        ax.set_title(key)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="2-D GMM fine-tuning demo")
    parser.add_argument(
        "--reward", choices=["linear", "quadratic", "box", "all"], default="linear"
    )
    parser.add_argument(
        "--algo",
        choices=["orwcfm", "diffusion_nft", "flow_grpo", "adjoint_matching", "all"],
        default="orwcfm",
    )
    parser.add_argument("--pretrain-steps", type=int, default=20_000)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--rollout-n", type=int, default=64)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    show = not args.no_show
    rewards = ["linear", "quadratic", "box"] if args.reward == "all" else [args.reward]
    algos = (
        ["orwcfm", "diffusion_nft", "flow_grpo", "adjoint_matching"]
        if args.algo == "all"
        else [args.algo]
    )

    # --- Pretrain ---
    print(f"\n=== Pretraining flow model ({args.pretrain_steps} steps) ===")
    mlp = VelocityMLP(width=128, depth=3).to(DEVICE)
    base_model = GMMFlowModel(mlp, DEVICE)
    pretrain_velocity_model(
        base_model,
        steps=args.pretrain_steps,
        batch_size=512,
        device=DEVICE,
        verbose=True,
    )
    pretrained_mlp = copy.deepcopy(mlp)  # frozen reference

    # Baseline samples + velocity check
    print("\n=== Verifying base model ===")
    base_samples = sample_model(base_model, n=2000, steps=50, device=DEVICE)
    print(
        f"  Base model sample range  x1: [{base_samples[:, 0].min():.2f}, {base_samples[:, 0].max():.2f}]"
    )

    visualize_velocity_field(
        base_model,
        t_values=[0.25, 0.5, 0.75],
        title_prefix="Base model",
        save_path=str(FIGURES_DIR / "base_velocity_field.png"),
        show=show,
    )

    # --- Fine-tune ---
    for reward_name in rewards:
        print(f"\n{'=' * 60}")
        print(f"  Reward: {reward_name}  (λ={args.lam})")
        print(f"{'=' * 60}")

        # Analytic target (if available)
        if reward_name == "linear":
            target = tilted_target_linear(args.lam)
        elif reward_name == "quadratic":
            target = tilted_target_quadratic(args.lam)
        else:
            target = None

        for algo_name in algos:
            print(f"\n--- Algorithm: {algo_name} ---")
            ft_model, history = run_finetune(
                algo_name,
                reward_name,
                pretrained_mlp,
                n_iter=args.iters,
                n_rollout=args.rollout_n,
                lam=args.lam,
                time_steps=args.time_steps,
            )
            if not history:
                continue

            ft_samples = sample_model(ft_model, n=2000, steps=50, device=DEVICE)

            # Metrics
            if reward_name == "linear":
                reward_fn = LinearReward()
            elif reward_name == "quadratic":
                reward_fn = QuadraticReward()
            else:
                reward_fn = BoxReward()

            met = compute_metrics(ft_samples, target, reward_fn=reward_fn)
            print(f"  Metrics: {', '.join(f'{k}={v:.4f}' for k, v in met.items())}")

            tag = f"{reward_name}_{algo_name}"

            visualize(
                base_samples,
                ft_samples,
                target,
                title=f"{reward_name.capitalize()} reward  |  {algo_name}  |  λ={args.lam}",
                save_path=str(FIGURES_DIR / f"{tag}_samples.png"),
                show=show,
            )

            plot_training_curves(
                history,
                title=f"{reward_name} / {algo_name}",
                save_path=str(FIGURES_DIR / f"{tag}_curves.png"),
                show=show,
            )

            if reward_name != "box":
                # Fine-tuned velocity field (target weights/means available)
                if target is not None:
                    visualize_velocity_field(
                        ft_model,
                        t_values=[0.5],
                        title_prefix=f"{tag}",
                        save_path=str(FIGURES_DIR / f"{tag}_velocity.png"),
                        show=show,
                        weights=target.weights,
                        means=target.means,
                    )

    print(f"\nDone. Figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
