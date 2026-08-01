"""Flow-GRPO: Group Relative Policy Optimization for flow models.

Algorithm:
  1. Collect grouped stochastic SDE trajectories under a frozen old policy.
     Store (x_k, x_{k+1}, drift_k, diffusion_k) for each step and the
     Gaussian transition log-prob log p_old(x_{k+1} | x_k).
  2. Compute per-group normalized advantages from terminal rewards.
  3. Run PPO epochs:
     a. For each step k: rebuild the Euler-Maruyama kernel under train/reference policies.
     b. Compute log ratio = log p_new - log p_old.
     c. Clipped PPO objective.
     d. KL penalty against reference policy (closed-form for Gaussian kernels).
     e. loss = -(ppo_obj - beta_kl * kl)

The kernel factory must match the one used during collection so that log-probs
are exact, not approximate.
"""

import copy
from collections.abc import Mapping

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.rollout import RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
    TrajectoryExperience,
)
from diffusiongym.trainers.orw_cfm import _index_conditioning
from diffusiongym.types import DDBatch


class FlowGRPO[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, TrajectoryExperience[StateT]]
):
    """Flow-GRPO fine-tuning.

    Parameters
    ----------
    group_size:
        Trajectories per condition group for advantage normalization.
    ppo_epochs:
        Gradient epochs over each collected replay buffer.
    ppo_batch_size:
        Number of trajectories per PPO mini-batch.
    clip_epsilon:
        PPO probability-ratio clip range.
    beta_kl:
        Coefficient for the KL penalty against the reference policy.
    halflife_iters:
        EMA halflife for reward normalization (unused here; group normalization used instead).
    """

    def __init__(
        self,
        *,
        group_size: int = 4,
        ppo_epochs: int = 4,
        ppo_batch_size: int = 64,
        clip_epsilon: float = 0.2,
        beta_kl: float = 0.01,
    ) -> None:
        self.group_size = group_size
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=True,
            needs_stochastic_rollout=True,
            needs_tractable_transitions=True,
            rollout_storage=RolloutStorage(
                states=True,
                log_probs=True,
            ),
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
    ) -> TrajectoryExperience[StateT]:
        # Snap n to a multiple of group_size
        num_groups = n // self.group_size
        n_eff = num_groups * self.group_size

        request = RolloutRequest(
            time_grid=time_grid,
            storage=self.requirements.rollout_storage,
            evaluate_reward=True,
        )

        # Collect under the rollout (old) policy — frozen snapshot
        rollout = context.sde_sampler.rollout(
            environment=context.environment,
            model=context.policies.rollout,
            dynamics=dynamics,
            n=n_eff,
            conditioning=conditioning,
            request=request,
            generator=generator,
        )
        assert rollout.reward is not None

        # Per-group normalized advantages
        rewards = rollout.reward.rewards  # shape (n_eff,)
        valid = rollout.reward.valid
        rewards_grouped = rewards.view(num_groups, self.group_size)
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        group_std = rewards_grouped.std(dim=1, keepdim=True).clamp_min(1e-8)
        advantages = ((rewards_grouped - group_mean) / group_std).view(n_eff)
        if valid is not None:
            advantages[~valid] = 0.0

        return TrajectoryExperience(rollout=rollout, advantages=advantages)

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: TrajectoryExperience[StateT],
    ) -> Mapping[str, float]:
        env = context.environment
        train_model = context.policies.train
        ref_model = context.policies.reference
        opt = context.optimizer
        device = train_model.device

        rollout = experience.rollout
        advantages = experience.advantages.to(device)
        num_steps = len(rollout.steps)
        n = len(rollout.terminal_latent)

        # We need the kernel factory from the SDE sampler to rebuild kernels
        kernel_factory = context.sde_sampler.kernel_factory

        total_loss = 0.0
        steps_done = 0

        for _epoch in range(self.ppo_epochs):
            step_order = torch.randperm(num_steps).tolist()

            for k in step_order:
                step = rollout.steps[k]
                assert step.log_prob is not None, (
                    "log_prob must be stored; set storage.log_probs=True"
                )
                assert step.x is not None, (
                    "states must be stored; set storage.states=True"
                )

                x_k = step.x.to(device)  # type: ignore[attr-defined]
                x_next = step.x_next.to(device)  # type: ignore[attr-defined]
                t_k = step.t.to(device)
                dt_k = step.dt.to(device).unsqueeze(0).expand(n)
                log_prob_old = step.log_prob.to(device)
                cond_k = rollout.conditioning  # same conditioning for all steps

                for start in range(0, n, self.ppo_batch_size):
                    end = min(start + self.ppo_batch_size, n)
                    idx = torch.arange(start, end, device=device)

                    x_b = x_k[idx]
                    x_next_b = x_next[idx]
                    t_b = t_k[idx]
                    dt_b = dt_k[idx]
                    adv_b = advantages[idx]
                    lp_old_b = log_prob_old[idx]
                    cond_b = _index_conditioning(cond_k, idx)

                    # New drift under train policy
                    v_new = env.predict_velocity(
                        train_model, x_t=x_b, t=t_b, conditioning=cond_b
                    )

                    # Re-derive drift using the same dynamics
                    # We need to re-compute coefficients with the new velocity
                    # We can't import dynamics here directly, so use the stored diffusion
                    # and recompute the mean directly using the kernel factory
                    diff_b = (
                        step.diffusion.to(device)[idx]
                        if step.diffusion is not None
                        else torch.zeros_like(t_b)
                    )

                    kernel_new = kernel_factory.build(
                        x=x_b, t=t_b, dt=dt_b, drift=v_new, diffusion=diff_b
                    )
                    lp_new_b = kernel_new.log_prob(x_next_b)

                    log_ratio = lp_new_b - lp_old_b
                    ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))

                    obj_unclipped = ratio * adv_b
                    obj_clipped = (
                        ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                        * adv_b
                    )
                    ppo_obj = torch.min(obj_unclipped, obj_clipped)

                    # KL against reference policy
                    kl = torch.zeros_like(adv_b)
                    if ref_model is not None and self.beta_kl > 0:
                        with torch.no_grad():
                            v_ref = env.predict_velocity(
                                ref_model, x_t=x_b, t=t_b, conditioning=cond_b
                            )
                        kernel_ref = kernel_factory.build(
                            x=x_b, t=t_b, dt=dt_b, drift=v_ref, diffusion=diff_b
                        )
                        kl = kernel_new.kl_divergence(kernel_ref)

                    loss = -(ppo_obj - self.beta_kl * kl).mean()
                    opt.zero_grad()
                    loss.backward()
                    if hasattr(train_model, "parameters"):
                        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)  # ty: ignore[call-non-callable]
                    opt.step()
                    total_loss += loss.item()
                    steps_done += 1

        assert rollout.reward is not None
        rewards = rollout.reward.rewards
        valid = rollout.reward.valid
        r_valid = rewards[valid] if valid is not None else rewards
        return {
            "loss": total_loss / max(steps_done, 1),
            "r_mean": r_valid.mean().item(),
            "r_std": r_valid.std().item() if len(r_valid) > 1 else 0.0,
            "valid_frac": valid.float().mean().item() if valid is not None else 1.0,
        }

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Hard-copy train weights into the rollout policy snapshot."""
        train = context.policies.train
        rollout = context.policies.rollout
        if hasattr(train, "state_dict") and hasattr(rollout, "load_state_dict"):
            rollout.load_state_dict(copy.deepcopy(train.state_dict()))  # ty: ignore[call-non-callable]
