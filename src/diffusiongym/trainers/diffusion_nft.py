"""Diffusion NFT (DPO-style contrastive fine-tuning for diffusion models)."""

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from diffusiongym.environments import Environment
from diffusiongym.train import DDDataset
from diffusiongym.trainers._utils import RunningRewardStats, filter_valid
from diffusiongym.types import DDBatch


def diffusion_nft[D: DDBatch](
    env: Environment[D],
    samples_per_iter: int,
    sample_batch_size: int,
    ft_batch_size: int,
    num_iterations: int = 100,
    beta: float = 1.0,
    ema_decay: float = 0.995,
    inner_epochs: int = 10,
    lr: float = 1e-4,
    postprocess_latents: Callable[[D], D] | None = None,
    log_every: int | None = None,
    exp_dir: Path | None = None,
):
    """Diffusion NFT fine-tuning via DPO-style contrastive objectives.

    Maintains an EMA old/sampling policy, constructs per-sample optimality
    probabilities from group-normalized rewards, and applies positive/negative
    prediction blending per the NFT paper (Theorem 3.2).

    Parameters
    ----------
    env:
        Environment. env.policy is set to a slow-moving EMA copy used for
        sampling; env.base_model is the trainable model.
    samples_per_iter:
        Endpoints to sample per outer iteration.
    sample_batch_size:
        Batch size for env.batch_sample.
    ft_batch_size:
        Mini-batch size for the inner training loop.
    num_iterations:
        Total outer iterations.
    beta:
        Contrastive blending coefficient (β=1 → full positive/negative flip).
    ema_decay:
        EMA decay for the sampling policy (0.995 → slow-moving). Should be
        close to 1 so the sampling policy lags behind training.
    inner_epochs:
        Training epochs over each sampled batch.
    lr:
        AdamW learning rate.
    postprocess_latents:
        Optional transform applied to sampled latents before training.
    log_every:
        Log every this many iterations (default: 1% of total).
    exp_dir:
        If provided, checkpoints saved here as last.pt.
    """
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    # env.policy = slow-moving EMA copy used for sampling.
    # env.base_model = trainable model.
    env.policy = copy.deepcopy(env.base_model)
    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)

    reward_stats = RunningRewardStats(halflife_iters=0.1 * num_iterations)

    for it in range(1, num_iterations + 1):
        # ------------------------------------------------------------------
        # 1. Sample endpoints using the EMA sampling policy.
        # ------------------------------------------------------------------
        sample = env.batch_sample(samples_per_iter, sample_batch_size)
        latents, rewards, kwargs = filter_valid(sample)

        if postprocess_latents is not None:
            latents = postprocess_latents(latents)

        # ------------------------------------------------------------------
        # 2. Optimality probabilities from EMA-normalized rewards.
        # ------------------------------------------------------------------
        reward_stats.update(rewards)
        r_norm = reward_stats.normalize(rewards)
        # Map to [0, 1]: r=1 means "definitely positive", r=0 "definitely negative".
        r = (0.5 + 0.5 * r_norm.clamp(-1, 1)).to(env.device)

        # ------------------------------------------------------------------
        # 3–6. Forward-process training with contrastive blending.
        # ------------------------------------------------------------------
        dataset = DDDataset([latents.to(env.device)], [kwargs], [r])
        loader = DataLoader(
            dataset,
            ft_batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
            num_workers=0,
            pin_memory=False,
        )

        env.base_model.train()

        x1: D
        batch_kwargs: dict[str, Any]
        opt_prob: torch.Tensor
        for _ in range(inner_epochs):
            for x1, batch_kwargs, opt_prob in loader:
                x1 = x1.to(env.device)
                opt_prob = opt_prob.to(env.device)

                # Standard forward-process interpolation.
                x0 = x1.randn_like()
                t = torch.rand(len(x1), device=env.device)
                alpha = env.scheduler.alpha(x1, t)
                beta_t = env.scheduler.beta(x1, t)
                xt = alpha * x1 + beta_t * x0

                new_pred = env.base_model(xt, t, **batch_kwargs)
                with torch.no_grad():
                    old_pred = env.policy(xt, t, **batch_kwargs)

                # Contrastive blending (Theorem 3.2):
                # v_pos = (1-β)v_old + β*v_new  →  regressed against FM target
                # v_neg = (1+β)v_old - β*v_new  →  pushed away from FM target
                pos_pred = (1 - beta) * old_pred + beta * new_pred
                neg_pred = (1 + beta) * old_pred - beta * new_pred

                pos_loss = env.base_model.train_loss(
                    x1, xt=xt, t=t, pred=pos_pred, **batch_kwargs
                )
                neg_loss = env.base_model.train_loss(
                    x1, xt=xt, t=t, pred=neg_pred, **batch_kwargs
                )

                loss = (opt_prob * pos_loss + (1 - opt_prob) * neg_loss).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(env.base_model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        # ------------------------------------------------------------------
        # 7. EMA update: slowly advance sampling policy toward training model.
        # θ_old = decay * θ_old + (1-decay) * θ_new
        # ------------------------------------------------------------------
        with torch.no_grad():
            for p_old, p_new in zip(
                env.policy.parameters(), env.base_model.parameters()
            ):
                p_old.data.mul_(ema_decay).add_(p_new.data, alpha=1.0 - ema_decay)

        if it % log_every == 0:
            metrics = {
                "r_mean": sample.rewards[sample.valids].mean(),
                "r_std": sample.rewards[sample.valids].std(),
                "r_min": sample.rewards[sample.valids].min(),
                "r_max": sample.rewards[sample.valids].max(),
                "valid": sample.valids.float().mean(),
            }
            logging.info(
                f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
            )

        if exp_dir is not None:
            torch.save(env.base_model.state_dict(), exp_dir / "last.pt")

    env.policy = env.base_model
    env.base_model.eval()
