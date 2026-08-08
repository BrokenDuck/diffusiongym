"""DiffusionNFT: DPO-style contrastive fine-tuning for flow models.

Algorithm (Theorem 3.2 of the NFT paper):
  1. Roll out the EMA sampling policy to collect endpoints.
  2. Map rewards to optimality probabilities r in [0,1].
  3. For each training batch, compute the positive and negative blended predictions:
       v_pos = (1-β)*v_old + β*v_new   → regressed toward FM target
       v_neg = (1+β)*v_old - β*v_new   → pushed away from FM target
  4. Loss = r * error(v_pos, target) + (1-r) * error(v_neg, target)
  5. EMA-update the sampling policy toward the train policy.

The rollout policy (v_old) lags behind the train policy via EMA. This prevents
collapse while still improving the sampling distribution over time.
"""

from collections.abc import Mapping

import torch
from torch import Generator, Tensor

from diffusiongym.core import FlowDynamics, RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    EndpointExperience,
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
)
from diffusiongym.trainers.orw_cfm import _index_conditioning, _RewardStats
from diffusiongym.types import DDBatch


class DiffusionNFT[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, EndpointExperience[StateT]]
):
    """DPO-style contrastive fine-tuning for flow models.

    Parameters
    ----------
    beta:
        Contrastive blending coefficient.
        β=1 → full positive/negative flip.
        β=0 → no change from the old policy.
    ema_decay:
        EMA decay for the sampling (rollout) policy.
        Close to 1 means the rollout policy lags behind training.
    inner_epochs:
        Training epochs over each collected batch.
    batch_size:
        Mini-batch size for inner training.
    halflife_iters:
        EMA halflife for reward normalization.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        ema_decay: float = 0.995,
        inner_epochs: int = 10,
        batch_size: int = 64,
        halflife_iters: float = 10.0,
    ) -> None:
        self.beta = beta
        self.ema_decay = ema_decay
        self.inner_epochs = inner_epochs
        self.batch_size = batch_size
        self._reward_stats = _RewardStats(halflife_iters=halflife_iters)

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=False,
            needs_stochastic_rollout=False,
            rollout_storage=RolloutStorage(),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        n: int,
        time_grid: Tensor,
        conditioning: Mapping[str, object],
        generator: Generator | None = None,
    ) -> EndpointExperience[StateT]:
        request = RolloutRequest(
            time_grid=time_grid,
            storage=self.requirements.rollout_storage,
            evaluate_reward=True,
        )
        rollout = context.ode_sampler.rollout(
            environment=context.environment,
            model=context.policies.rollout,
            dynamics=dynamics,
            n=n,
            conditioning=conditioning,
            request=request,
            generator=generator,
        )
        assert rollout.reward is not None
        return EndpointExperience(
            latent=rollout.terminal_latent,
            rewards=rollout.reward.rewards,
            valid=rollout.reward.valid,
            conditioning=rollout.conditioning,
        )

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: EndpointExperience[StateT],
    ) -> Mapping[str, float]:
        env = context.environment
        train_model = context.policies.train
        rollout_model = context.policies.rollout
        opt = context.optimizer
        device = train_model.device

        rewards = experience.rewards
        valid = experience.valid
        valid_rewards = rewards[valid] if valid is not None else rewards

        self._reward_stats.update(valid_rewards)
        r_norm = self._reward_stats.normalize(rewards)
        # Map normalized rewards to [0,1] optimality probability
        r = (0.5 + 0.5 * r_norm.clamp(-1.0, 1.0)).to(device)

        latent = experience.latent.to(device)
        n = len(latent)

        total_loss = 0.0
        steps = 0

        for _ in range(self.inner_epochs):
            idx = torch.randint(0, n, (min(self.batch_size, n),), device=device)
            x_data_b = latent[idx]
            r_b = r[idx]
            cond_b = _index_conditioning(experience.conditioning, idx)

            batch = env.make_forward_batch(x_data_b, conditioning=cond_b)

            with torch.no_grad():
                v_old = env.predict_velocity(
                    rollout_model, x_t=batch.x_t, t=batch.t, conditioning=cond_b
                )

            v_new = env.predict_velocity(
                train_model, x_t=batch.x_t, t=batch.t, conditioning=cond_b
            )

            # Contrastive blending
            v_pos = v_old * (1.0 - self.beta) + v_new * self.beta
            v_neg = v_old * (1.0 + self.beta) - v_new * self.beta

            pos_err = env.velocity_error(v_pos, batch.target_velocity)
            neg_err = env.velocity_error(v_neg, batch.target_velocity)

            loss = (r_b * pos_err + (1.0 - r_b) * neg_err).mean()
            opt.zero_grad()
            loss.backward()
            if hasattr(train_model, "parameters"):
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)  # ty: ignore[call-non-callable]
            opt.step()
            total_loss += loss.item()
            steps += 1

        r_valid = rewards[valid] if valid is not None else rewards
        return {
            "loss": total_loss / max(steps, 1),
            "r_mean": r_valid.mean().item(),
            "r_std": r_valid.std().item() if len(r_valid) > 1 else 0.0,
            "valid_frac": valid.float().mean().item() if valid is not None else 1.0,
        }

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """EMA-update the rollout (sampling) policy toward the train policy."""
        train = context.policies.train
        rollout = context.policies.rollout
        if not (hasattr(train, "parameters") and hasattr(rollout, "parameters")):
            return
        d = self.ema_decay
        with torch.no_grad():
            for p_old, p_new in zip(rollout.parameters(), train.parameters()):  # ty: ignore[call-non-callable]
                p_old.data.mul_(d).add_(p_new.data, alpha=1.0 - d)
