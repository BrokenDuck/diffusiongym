"""Online Reward-Weighted Conditional Flow Matching (ORW-CFM).

At each iteration:
  1. Roll out the current model deterministically (ODE).
  2. Normalize rewards using an exponential moving average.
  3. Compute exponential importance weights: exp(temperature * r_normalized).
  4. Regress the train model toward the flow-matching target on sampled endpoints,
     weighted by importance weights.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.rollout import RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    EndpointExperience,
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
)
from diffusiongym.types import DDBatch

type Conditioning = Mapping[str, object]


@dataclass
class _RewardStats:
    """Exponential moving average reward normalizer."""

    halflife_iters: float
    _m1: float | None = field(default=None, init=False, repr=False)
    _m2: float | None = field(default=None, init=False, repr=False)

    @property
    def _beta(self) -> float:
        return 1.0 - 2.0 ** (-1.0 / max(self.halflife_iters, 1.0))

    def update(self, rewards: Tensor) -> None:
        r = rewards.float()
        m1 = r.mean().item()
        m2 = (r**2).mean().item()
        if self._m1 is None or self._m2 is None:
            self._m1, self._m2 = m1, m2
        else:
            b = self._beta
            self._m1 = (1.0 - b) * self._m1 + b * m1
            self._m2 = (1.0 - b) * self._m2 + b * m2

    def normalize(self, rewards: Tensor) -> Tensor:
        if self._m1 is None:
            return torch.zeros_like(rewards)
        var = max(self._m2 - self._m1**2, 1e-6) if self._m2 is not None else 1e-6
        return (rewards - self._m1) / (var**0.5 + 1e-8)


def _index_conditioning(conditioning: Mapping, idx: Tensor) -> dict:
    """Sub-select conditioning dict to a batch of indices."""
    result = {}
    for k, v in conditioning.items():
        if isinstance(v, torch.Tensor):
            result[k] = v[idx]
        elif isinstance(v, list):
            result[k] = [v[i] for i in idx.tolist()]
        else:
            result[k] = v
    return result


class ORWCFM[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, EndpointExperience[StateT]]
):
    """Online Reward-Weighted CFM fine-tuning.

    Parameters
    ----------
    temperature:
        Inverse temperature for importance weights exp(temperature * r_norm).
    steps_per_update:
        Number of gradient steps per collected batch.
    batch_size:
        Mini-batch size for the inner training loop.
    halflife_iters:
        EMA halflife for reward normalization (in iterations).
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        steps_per_update: int = 10,
        batch_size: int = 64,
        halflife_iters: float = 10.0,
    ) -> None:
        self.temperature = temperature
        self.steps_per_update = steps_per_update
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
        model = context.policies.train
        opt = context.optimizer
        device = model.device

        rewards = experience.rewards
        valid = experience.valid
        valid_rewards = rewards[valid] if valid is not None else rewards

        self._reward_stats.update(valid_rewards)
        r_norm = self._reward_stats.normalize(rewards).clamp(-5.0, 5.0)
        weights = torch.exp(self.temperature * r_norm).to(device)
        weights = weights / weights.mean().clamp_min(1e-8)

        latent = experience.latent.to(device)  # type: ignore[attr-defined]
        n = len(latent)

        total_loss = 0.0

        for _ in range(self.steps_per_update):
            idx = torch.randint(0, n, (min(self.batch_size, n),), device=device)
            x_data_b = latent[idx]
            w_b = weights[idx]
            cond_b = _index_conditioning(experience.conditioning, idx)

            batch = env.make_forward_batch(x_data_b, conditioning=cond_b)
            pred_v = env.predict_velocity(
                model, x_t=batch.x_t, t=batch.t, conditioning=cond_b
            )
            per_sample_loss = env.velocity_error(pred_v, batch.target_velocity)

            loss = (w_b * per_sample_loss).mean()
            opt.zero_grad()
            loss.backward()
            if hasattr(model, "parameters"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ty: ignore[call-non-callable]
            opt.step()
            total_loss += loss.item()

        r_valid = rewards[valid] if valid is not None else rewards
        return {
            "loss": total_loss / max(self.steps_per_update, 1),
            "r_mean": r_valid.mean().item(),
            "r_std": r_valid.std().item() if len(r_valid) > 1 else 0.0,
            "valid_frac": valid.float().mean().item() if valid is not None else 1.0,
        }
