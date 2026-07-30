"""Flow-GRPO: Group Relative Policy Optimization for Flow Models."""

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diffusiongym.environments import Environment
from diffusiongym.types import DDBatch
from diffusiongym.utils import dict_to_device, index_dict


@dataclass
class _Transition[D: DDBatch]:
    x_t: D
    x_next: D
    t: torch.Tensor      # shape (batch,)
    old_mean: D
    sigma_t: float
    dt: float
    advantage: torch.Tensor  # shape (batch,)
    kwargs: dict[str, Any]


def flow_grpo[D: DDBatch](
    env: Environment[D],
    samples_per_iter: int,
    group_size: int = 4,
    num_iterations: int = 100,
    ppo_epochs: int = 4,
    ppo_batch_size: int = 64,
    clip_epsilon: float = 0.2,
    beta_kl: float = 0.01,
    sigma_scale: float = 1.0,
    lr: float = 1e-5,
    log_every: int | None = None,
    exp_dir: Path | None = None,
):
    """Flow-GRPO fine-tuning.

    Implements Group Relative Policy Optimization for continuous-time flow
    models. Each iteration:

      1. Collects grouped stochastic SDE trajectories using a frozen old policy.
      2. Computes per-group normalized advantages from terminal rewards.
      3. Runs PPO updates with a clipped importance-ratio objective and a
         closed-form KL penalty against a fixed reference model.

    The stochastic SDE uses ``sigma_t = sigma_scale * sqrt(t / (1-t))``,
    clamped to avoid divergence near t=1. This differs from the environment's
    memoryless schedule; env.sample is NOT used here.

    Parameters
    ----------
    env:
        Environment. env.base_model is fine-tuned in-place.
    samples_per_iter:
        Total trajectories per iteration. Must be divisible by group_size;
        any remainder is truncated.
    group_size:
        Number of trajectories per condition group for advantage estimation.
    num_iterations:
        Total outer iterations.
    ppo_epochs:
        PPO gradient epochs over each collected replay buffer.
    ppo_batch_size:
        Mini-batch size for PPO updates.
    clip_epsilon:
        PPO probability-ratio clipping range.
    beta_kl:
        Coefficient for the KL penalty against the reference model.
    sigma_scale:
        Noise amplitude ``a`` in ``sigma_t = a * sqrt(t / (1-t))``.
    lr:
        AdamW learning rate.
    log_every:
        Log every this many iterations (default: 1% of total).
    exp_dir:
        If provided, checkpoints saved here as last.pt.
    """
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    # Truncate to a multiple of group_size.
    num_groups = samples_per_iter // group_size
    n = num_groups * group_size

    policy = env.base_model

    # Frozen KL anchor — never updated.
    reference_model = copy.deepcopy(policy)
    reference_model.eval()
    reference_model.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    device = env.device

    timesteps = torch.linspace(0.0, 1.0, env.discretization_steps + 1, device=device)

    for it in range(1, num_iterations + 1):
        policy.eval()

        # Freeze a snapshot for this iteration's rollouts.
        old_model = copy.deepcopy(policy)
        old_model.eval()
        old_model.requires_grad_(False)

        # ------------------------------------------------------------------
        # 1. Collect stochastic SDE trajectories under old_model.
        # ------------------------------------------------------------------
        buffer: list[_Transition[D]] = []

        with torch.no_grad():
            latent, kwargs = policy.sample_p0(n)
            latent, kwargs = policy.preprocess(latent, **kwargs)
            kwargs_dev = dict_to_device(kwargs, device)

            x = latent

            for t0, t1 in zip(timesteps[:-1], timesteps[1:]):
                dt = (t1 - t0).item()
                t_val = t0.clamp_min(2e-2).item()
                t_curr = torch.full((n,), t_val, device=device)

                # sigma_t = sigma_scale * sqrt(t / (1-t)), clamped below t=0.98
                t_clamped = min(t_val, 0.98)
                sigma_t = sigma_scale * math.sqrt(t_clamped / max(1.0 - t_clamped, 1e-6))

                old_pred = old_model(x, t_curr, **kwargs_dev)
                old_drift = env.drift_from_prediction(x, t_curr, old_pred)
                old_mean = x + dt * old_drift

                epsilon = x.randn_like()
                x_next = old_mean + math.sqrt(dt) * sigma_t * epsilon

                buffer.append(_Transition(
                    x_t=x.cpu(),
                    x_next=x_next.cpu(),
                    t=t_curr.cpu(),
                    old_mean=old_mean.cpu(),
                    sigma_t=sigma_t,
                    dt=dt,
                    advantage=torch.zeros(n),  # filled below
                    kwargs=dict_to_device(kwargs_dev, "cpu"),
                ))
                x = x_next

            # Terminal reward.
            final_sample = policy.postprocess(x)
            rewards, valids = env.reward(final_sample, x, **kwargs_dev)

        # ------------------------------------------------------------------
        # 2. Per-group normalized advantages.
        # ------------------------------------------------------------------
        rewards_grouped = rewards.view(num_groups, group_size)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
        advantages = ((rewards_grouped - group_mean) / group_std).view(n)
        advantages[~valids] = 0.0

        for tr in buffer:
            tr.advantage = advantages.clone()

        # ------------------------------------------------------------------
        # 3. PPO update epochs.
        # ------------------------------------------------------------------
        policy.train()

        num_steps = len(buffer)

        for _epoch in range(ppo_epochs):
            step_order = torch.randperm(num_steps).tolist()

            for k in step_order:
                tr = buffer[k]
                sigma_t = tr.sigma_t
                dt = tr.dt
                variance = sigma_t ** 2 * dt

                x_t_dev = tr.x_t.to(device)
                x_next_dev = tr.x_next.to(device)
                old_mean_dev = tr.old_mean.to(device)
                t_dev = tr.t.to(device)
                adv_dev = tr.advantage.to(device)
                kw_dev = dict_to_device(tr.kwargs, device)

                for start in range(0, n, ppo_batch_size):
                    end = min(start + ppo_batch_size, n)
                    sl = slice(start, end)

                    x_b = x_t_dev[sl].to(device)
                    x_next_b = x_next_dev[sl].to(device)
                    old_mean_b = old_mean_dev[sl].to(device)
                    t_b = t_dev[sl]
                    adv_b = adv_dev[sl]
                    kw_b = index_dict(kw_dev, start, end)

                    # New drift and mean.
                    new_pred = policy(x_b, t_b, **kw_b)
                    new_drift = env.drift_from_prediction(x_b, t_b, new_pred)
                    new_mean_b = x_b + dt * new_drift

                    # Log-probabilities of the stored transition under new and old policy.
                    # logp ∝ -||x_next - mean||² / (2 * sigma² * dt)
                    sq_new = ((x_next_b - new_mean_b) ** 2 / variance).aggregate("sum")
                    sq_old = ((x_next_b - old_mean_b) ** 2 / variance).aggregate("sum")
                    log_ratio = 0.5 * (sq_old - sq_new)  # logp_new - logp_old
                    ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))

                    # Clipped PPO objective.
                    obj_unclipped = ratio * adv_b
                    obj_clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_b
                    clipped_obj = torch.min(obj_unclipped, obj_clipped)

                    # Closed-form KL against reference (prediction-space).
                    # KL ≈ (dt/2) * c(t)² * |v_new - v_ref|²
                    # where c(t) = sigma*(1-t)/(2t) + 1/sigma
                    with torch.no_grad():
                        ref_pred = reference_model(x_b, t_b, **kw_b)

                    t_val_b = t_b.mean().item()
                    c_t = (
                        sigma_t * (1.0 - t_val_b) / (2.0 * t_val_b + 1e-8)
                        + 1.0 / (sigma_t + 1e-8)
                    )
                    kl = 0.5 * dt * c_t ** 2 * ((new_pred - ref_pred) ** 2).aggregate("mean")

                    loss = -(clipped_obj - beta_kl * kl).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()

        if it % log_every == 0:
            metrics = {
                "r_mean": rewards[valids].mean(),
                "r_std": rewards[valids].std(),
                "r_min": rewards[valids].min(),
                "r_max": rewards[valids].max(),
                "valid": valids.float().mean(),
            }
            logging.info(
                f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
            )

        if exp_dir is not None:
            torch.save(policy.state_dict(), exp_dir / "last.pt")

    policy.eval()
