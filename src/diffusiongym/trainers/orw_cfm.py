"""Online Reward-Weighted Conditional Flow Matching (ORW-CFM)."""

import logging
from collections.abc import Callable
from pathlib import Path

import torch

from diffusiongym.environments import Environment
from diffusiongym.train import train_base_model
from diffusiongym.trainers._utils import RunningRewardStats, filter_valid
from diffusiongym.types import DDBatch


def orw_cfm[D: DDBatch](
    env: Environment[D],
    samples_per_iter: int,
    sample_batch_size: int,
    train_batch_size: int,
    steps_per_iter: int,
    num_iterations: int = 100,
    lr: float = 1e-5,
    temperature: float = 1.0,
    postprocess_latents: Callable[[D], D] | None = None,
    log_every: int | None = None,
    exp_dir: Path | None = None,
):
    """Online Reward-Weighted CFM fine-tuning.

    At each iteration, samples endpoints from the environment, normalizes
    rewards via EMA tracking, computes exponential importance weights scaled by
    ``temperature``, and trains the base model for ``steps_per_iter`` gradient
    steps.

    Parameters
    ----------
    env:
        Environment whose base_model is fine-tuned.
    samples_per_iter:
        Number of endpoints to sample per iteration.
    sample_batch_size:
        Batch size for env.batch_sample.
    train_batch_size:
        Mini-batch size for the inner training loop.
    steps_per_iter:
        Gradient steps per iteration.
    num_iterations:
        Total outer iterations.
    lr:
        AdamW learning rate.
    temperature:
        Inverse temperature for the exponential importance weights
        ``exp(temperature * r_normalized)``. Separate from env.reward_scale,
        which governs the SDE cost functional.
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

    logging.info(f"Reward scale: {env.reward_scale}")

    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)
    reward_stats = RunningRewardStats(halflife_iters=0.1 * num_iterations)

    for it in range(1, num_iterations + 1):
        sample = env.batch_sample(samples_per_iter, sample_batch_size)
        latents, rewards, kwargs = filter_valid(sample)

        if postprocess_latents is not None:
            latents = postprocess_latents(latents)

        reward_stats.update(rewards)
        r_norm = reward_stats.normalize(rewards).clamp(-5, 5)

        weights = torch.exp(temperature * r_norm)
        weights = weights / weights.mean()

        if it % log_every == 0:
            ess = (weights.sum() ** 2) / (weights.pow(2).sum() + 1e-8) / len(weights)
            metrics = {
                "r_mean": sample.rewards[sample.valids].mean(),
                "r_std": sample.rewards[sample.valids].std(),
                "r_min": sample.rewards[sample.valids].min(),
                "r_max": sample.rewards[sample.valids].max(),
                "valid": sample.valids.float().mean(),
                "ess": ess,
            }
            logging.info(
                f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
            )

        train_base_model(
            env.base_model,
            opt,
            [latents.to(env.base_model.device)],
            [kwargs],
            weights=[weights],
            steps=steps_per_iter,
            batch_size=train_batch_size,
            pbar=False,
        )

        if exp_dir is not None:
            torch.save(env.base_model.state_dict(), exp_dir / "last.pt")
