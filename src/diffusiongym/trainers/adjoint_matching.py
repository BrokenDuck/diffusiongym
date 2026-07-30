"""Adjoint Matching fine-tuning (https://arxiv.org/abs/2409.08861)."""

import copy
import logging
from pathlib import Path

import torch
from torch import nn

from diffusiongym.environments import Environment
from diffusiongym.environments.base import EnvironmentMode
from diffusiongym.types import DDBatch
from diffusiongym.utils import dict_to_device


def adjoint_matching[D: DDBatch](
    env: Environment[D],
    samples_per_iter: int,
    train_steps_per_iter: int,
    train_batch_size: int,
    num_iterations: int = 100,
    lr: float = 1e-4,
    log_every: int | None = None,
    exp_dir: Path | None = None,
):
    """Adjoint Matching fine-tuning.

    Rolls out the policy under the memoryless SDE, integrates the lean adjoint
    backward through the base drift using VJPs, and regresses the policy drift
    toward the per-step adjoint targets.

    Requires a differentiable reward: the terminal adjoint condition is
    a_K = -reward_scale * ∇_{x_K} r(x_K), computed via autograd. A ValueError
    is raised if the reward gradient is zero everywhere.

    Parameters
    ----------
    env:
        Environment. After training, env.base_model and env.policy both point
        to the trained policy; the frozen reference is discarded.
    samples_per_iter:
        Trajectories per outer iteration.
    train_steps_per_iter:
        Gradient steps per inner training loop.
    train_batch_size:
        Mini-batch size for the inner loop.
    num_iterations:
        Total outer iterations.
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

    # Freeze reference; trainable policy starts from same weights.
    reference_model = copy.deepcopy(env.base_model)
    reference_model.eval()
    reference_model.requires_grad_(False)

    policy = env.base_model
    env.base_model = reference_model
    env.policy = policy
    env.mode = EnvironmentMode.ADJOINT_MATCHING

    opt = torch.optim.AdamW(policy.parameters(), lr=lr)

    for it in range(1, num_iterations + 1):
        policy.eval()

        # ------------------------------------------------------------------
        # 1. Forward rollout under the memoryless SDE.
        # ------------------------------------------------------------------
        sample = env.sample(samples_per_iter, pbar=False)
        # sample.trajectory[k] is x at timesteps[k], stored on CPU.

        n = samples_per_iter
        device = env.device
        kwargs_dev = dict_to_device(sample.kwargs, device)
        timesteps = sample.timesteps.to(device)

        # ------------------------------------------------------------------
        # 2. Terminal adjoint: a_K = -reward_scale * ∇_{x_K} r(x_K).
        # env.sample stores rewards under @torch.no_grad, so re-evaluate.
        # ------------------------------------------------------------------
        x_terminal = sample.latent.to(device).requires_grad(True)
        terminal_out = env.base_model.postprocess(x_terminal)
        rewards_diff, valids = env.reward(terminal_out, x_terminal, **kwargs_dev)

        valid_rewards = (env.reward_scale * rewards_diff * valids.float()).sum()

        # DDBatch.gradient computes per-sample grads; samples are independent so
        # differentiating the sum gives per-sample ∇_{x_i} r(x_i).
        a_K = x_terminal.gradient(valid_rewards)
        a_K: D = (-a_K).detach()  # ty: ignore[invalid-assignment]

        # Guard: if reward is non-differentiable the gradient will be zero.
        if a_K.aggregate("sum").abs().max().item() == 0.0:
            raise ValueError(
                "Adjoint Matching requires a differentiable reward. "
                "The terminal adjoint ∇_{x_K} r(x_K) is zero everywhere. "
                "Supply a reward whose output depends differentiably on x_K."
            )

        # ------------------------------------------------------------------
        # 3. Lean adjoint backward integration through the BASE drift.
        # a_{k-1} = a_k + dt * (∇_x b(x_k, t_k))^T a_k
        # where b = reference_drift (NOT the policy drift).
        # ------------------------------------------------------------------
        num_steps = sample.num_steps
        adjoints: list[D] = [a_K.cpu()] * (num_steps + 1)
        adjoints[num_steps] = a_K.cpu()

        a = a_K

        for k in range(num_steps - 1, -1, -1):
            dt = timesteps[k + 1] - timesteps[k]
            t_eval = timesteps[k].clamp_min(2e-2)
            t_curr = t_eval.expand(n)

            x_k = sample.trajectory[k].to(device).requires_grad(True)

            # reference_drift is NOT under no_grad by design.
            ref_drift = env.reference_drift(x_k, t_curr, **kwargs_dev)

            # VJP: (∇_x ref_drift)^T a_k  via  ∂(ref_drift · a)/∂x_k
            dot = (ref_drift * a.to(device)).aggregate("sum").sum()
            vjp = x_k.gradient(dot, create_graph=False, retain_graph=False)

            a = (a + dt * vjp.detach()).detach()
            adjoints[k] = a.cpu()

        # ------------------------------------------------------------------
        # 4. Per-step regression targets in drift-space.
        # target_drift_k = ref_drift(x_k, t_k) - sigma_k² * a_k
        # ------------------------------------------------------------------
        # Collect all (x_k, t_k, target_drift_k) triples for training.
        xs_list: list[D] = []
        ts_list: list[torch.Tensor] = []
        targets_list: list[D] = []

        with torch.no_grad():
            for k in range(num_steps):
                t_eval = timesteps[k].clamp_min(2e-2)
                t_curr = t_eval.expand(n)
                x_k = sample.trajectory[k].to(device)

                ref_drift_k = env.reference_drift(x_k, t_curr, **kwargs_dev)
                sigma_k = env.scheduler.sigma(x_k, t_curr)

                a_k = adjoints[k].to(device)
                target_k = ref_drift_k - sigma_k.square() * a_k

                xs_list.append(x_k.cpu())
                ts_list.append(t_curr.cpu())
                targets_list.append(target_k.cpu())

        # ------------------------------------------------------------------
        # 5. Mini-batch training: regress policy drift toward targets.
        # ------------------------------------------------------------------
        policy.train()

        # Flatten all steps into one big pool.
        all_x = type(xs_list[0]).collate(xs_list)  # (K*N, ...)
        all_t = torch.cat(ts_list, dim=0)  # (K*N,)
        all_targets = type(targets_list[0]).collate(targets_list)  # (K*N, ...)

        total = len(all_x)
        opt.zero_grad()

        for step in range(train_steps_per_iter):
            idx = torch.randint(0, total, (train_batch_size,)).tolist()

            x_b = type(all_x).collate([all_x[i] for i in idx]).to(device)
            t_b = all_t[idx].to(device)
            target_b = (
                type(all_targets).collate([all_targets[i] for i in idx]).to(device)
            )

            # All trajectory steps share the same condition kwargs; pick per-sample index.
            sample_idx = [i % n for i in idx]
            sample_idx_t = torch.tensor(sample_idx, device=device)
            kw_b = {
                k: (
                    [v[j] for j in sample_idx]
                    if isinstance(v, list)
                    else v[sample_idx_t]
                )
                for k, v in kwargs_dev.items()
            }

            policy_pred = policy(x_b, t_b, **kw_b)
            policy_drift = env.drift_from_prediction(x_b, t_b, policy_pred)

            loss = ((policy_drift - target_b) ** 2).aggregate("mean").mean()
            loss.backward()

            if (step + 1) % 1 == 0:
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        if it % log_every == 0:
            metrics = {
                "r_mean": sample.rewards[sample.valids].mean(),
                "r_std": sample.rewards[sample.valids].std(),
                "r_min": sample.rewards[sample.valids].min(),
                "r_max": sample.rewards[sample.valids].max(),
                "valid": sample.valids.float().mean(),
                "running_cost_mean": sample.running_costs.sum(dim=0).mean(),
            }
            logging.info(
                f"(iter={it:05d}) {', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])}"
            )

        if exp_dir is not None:
            torch.save(policy.state_dict(), exp_dir / "last.pt")

    # Restore env to consistent state.
    env.base_model = policy
    env.policy = None
    env.mode = EnvironmentMode.BASE_INFERENCE
    policy.eval()
