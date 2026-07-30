"""Trajectory-level Reward-Weighted MLE fine-tuning."""

import logging
from pathlib import Path

import torch
from torch import nn

from diffusiongym.environments import Environment
from diffusiongym.schedulers.base import MemorylessNoiseSchedule
from diffusiongym.types import DDBatch


def reward_weighted_mle[D: DDBatch](
    env: Environment[D],
    samples_per_iter: int,
    num_iterations: int = 100,
    lr: float = 1e-5,
    temperature: float = 1.0,
    noise_schedule_override: MemorylessNoiseSchedule[D] | None = None,
    log_every: int | None = None,
    exp_dir: Path | None = None,
):
    """Trajectory-level reward-weighted MLE fine-tuning.

    Trains directly on the SDE trajectory by regressing each Euler step with
    per-sample importance weights derived from normalized rewards.  Callers
    may supply a ``noise_schedule_override`` to replace the base model's
    noise schedule before training (e.g. a domain-specific memoryless schedule).
    """
    if noise_schedule_override is not None:
        env.base_model.scheduler.noise_schedule = noise_schedule_override

    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)

    r_ema_m1 = None
    r_ema_m2 = None

    halflife = 0.1 * num_iterations
    beta = 1 - 2 ** (-1 / halflife)

    for it in range(1, num_iterations + 1):
        s = env.sample(samples_per_iter, pbar=False)
        r = s.rewards.clone()
        v = s.valids.clone()

        with torch.no_grad():
            if r_ema_m1 is None or r_ema_m2 is None:
                r_ema_m1 = r[v].mean()
                r_ema_m2 = (r[v] ** 2).mean()
            else:
                r_ema_m1 = (1 - beta) * r_ema_m1 + beta * r[v].mean()
                r_ema_m2 = (1 - beta) * r_ema_m2 + beta * (r[v] ** 2).mean()

            r_ema_var = (r_ema_m2 - r_ema_m1**2).clamp_min(1e-6)

        r[v] = (r[v] - r_ema_m1) / (r_ema_var.sqrt() + 1e-8)
        r[v] = r[v].clamp(-5, 5)
        r[~v] = -20

        r = r.to(env.device)

        env.base_model.train()
        total_loss = 0.0
        opt.zero_grad()

        for x_t, x_t_next, t0, t1, diffusion_t in zip(
            s.trajectory[:-1],
            s.trajectory[1:],
            s.timesteps[:-1],
            s.timesteps[1:],
            s.diffusions,
        ):
            x_t = x_t.to(env.device)
            x_t_next = x_t_next.to(env.device)
            diffusion_t = diffusion_t.to(env.device)

            dt = t1 - t0
            t = t0 * torch.ones(len(x_t), device=x_t.device)
            new_drift_t, _ = env.drift(x_t, t, **s.kwargs)
            new_mean_t_next = x_t + dt * new_drift_t

            weight = torch.exp(temperature * r)
            loss = weight * (
                ((x_t_next - new_mean_t_next) / diffusion_t) ** 2
            ).aggregate("mean")
            loss = loss.mean() / s.num_steps

            if loss.isnan().any() or loss.isinf().any():
                raise ValueError("Loss is NaN or Inf")

            loss.backward()
            total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(env.base_model.parameters(), 1.0)
        opt.step()

        if it % log_every == 0:
            metrics = {
                "loss": total_loss,
                "grad_norm": grad_norm,
                "r_mean": s.rewards[s.valids].mean(),
                "r_std": s.rewards[s.valids].std(),
                "r_min": s.rewards[s.valids].min(),
                "r_max": s.rewards[s.valids].max(),
                "valid": s.valids.float().mean(),
            }
            logging.info(
                f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
            )

        if exp_dir is not None:
            torch.save(env.base_model.state_dict(), exp_dir / "last.pt")
