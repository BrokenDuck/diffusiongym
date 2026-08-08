"""Online Reward-Weighted Conditional Flow Matching (ORW-CFM-W2).

See algorithm_specs/orw_cfm_w2.md. At each iteration:
  1. Roll out the *rollout* policy deterministically (ODE).
  2. Normalize rewards using an exponential moving average.
  3. Compute exponential importance weights: exp(temperature * r_normalized).
  4. Regress the train model toward the flow-matching target on sampled endpoints,
     weighted by importance weights, plus a W2 surrogate penalty
     alpha_w2 * ||v_theta - v_ref||^2 against the frozen reference field.
  5. Every `rollout_update_interval` iterations, refresh the rollout policy from
     the train policy.

Steps 4 and 5 are what make this the algorithm in the spec rather than a
one-shot importance-weighted refit:

  - Without step 5 the rollout policy stays at the pretrained weights forever, so
    every iteration redraws from p_base and the fit converges to
    p_base * exp(lambda r) directly. That is the right answer only while the
    target is close enough to p_base for the importance weights to have usable
    variance; it is not the online algorithm, and it silently stops being a
    reward-fine-tuning method at all.
  - Without step 4 the online iteration reweights its *own* samples every round,
    p_{k+1} ∝ p_k * exp(tau * z_k), which has no stationary point — the W2 term
    is what the spec relies on to prevent that collapse.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Generator, Tensor

from diffusiongym.core import FlowDynamics, RolloutRequest, RolloutStorage
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
        Because r_norm is standardized, the tilt this targets is
        lambda_eff = temperature / std_r — divide std_r back out to compare
        against an algorithm that applies the reward on its own scale.
    alpha_w2:
        Coefficient of the W2 surrogate penalty ||v_theta - v_ref||^2 against
        the frozen reference field. Requires a reference policy when positive.
        Zero reproduces plain ORW-CFM, which the spec says is the configuration
        that collapses; it is the default only because the spec gives no value
        for it, not because it is the recommended setting.
    rollout_update_interval:
        Refresh the rollout policy from the train policy every this many
        iterations (spec step 4, "H"). This is what makes the method online.
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
        alpha_w2: float = 0.0,
        rollout_update_interval: int = 1,
        steps_per_update: int = 10,
        batch_size: int = 64,
        halflife_iters: float = 10.0,
    ) -> None:
        if alpha_w2 < 0.0:
            raise ValueError(f"alpha_w2 must be non-negative, got {alpha_w2}.")
        if rollout_update_interval < 1:
            raise ValueError(
                "rollout_update_interval must be at least 1, got "
                f"{rollout_update_interval}."
            )
        self.temperature = temperature
        self.alpha_w2 = alpha_w2
        self.rollout_update_interval = rollout_update_interval
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size
        self._reward_stats = _RewardStats(halflife_iters=halflife_iters)
        self._iterations = 0

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            # The W2 surrogate regresses against the frozen reference field, so
            # it needs one; plain ORW-CFM (alpha_w2 = 0) does not.
            needs_reference_policy=self.alpha_w2 > 0.0,
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

        ref_model = context.policies.reference
        use_w2 = self.alpha_w2 > 0.0 and ref_model is not None

        total_loss = 0.0
        total_w2 = 0.0

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

            # W2 surrogate: penalize the pointwise velocity discrepancy against
            # the frozen reference field on exactly the same (x_t, t). This is
            # not a KL term — it bounds the terminal Wasserstein-2 distance.
            if use_w2:
                with torch.no_grad():
                    ref_v = env.predict_velocity(
                        ref_model, x_t=batch.x_t, t=batch.t, conditioning=cond_b
                    )
                w2 = env.velocity_error(pred_v, ref_v).mean()
                loss = loss + self.alpha_w2 * w2
                total_w2 += w2.item()

            opt.zero_grad()
            loss.backward()
            if hasattr(model, "parameters"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ty: ignore[call-non-callable]
            opt.step()
            total_loss += loss.item()

        r_valid = rewards[valid] if valid is not None else rewards
        return {
            "loss": total_loss / max(self.steps_per_update, 1),
            "w2": total_w2 / max(self.steps_per_update, 1),
            "r_mean": r_valid.mean().item(),
            "r_std": r_valid.std().item() if len(r_valid) > 1 else 0.0,
            "valid_frac": valid.float().mean().item() if valid is not None else 1.0,
        }

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Refresh the rollout policy from the train policy every H iterations.

        Spec step 4. Without this the rollout policy stays at the pretrained
        weights and the method degenerates into a repeated one-shot
        importance-weighted refit of p_base * exp(lambda r) — it stops being
        online, and stops working once the target is far enough from p_base that
        the importance weights degenerate.
        """
        self._iterations += 1
        if self._iterations % self.rollout_update_interval != 0:
            return
        train = context.policies.train
        rollout = context.policies.rollout
        if hasattr(train, "state_dict") and hasattr(rollout, "load_state_dict"):
            rollout.load_state_dict(copy.deepcopy(train.state_dict()))
