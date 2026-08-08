"""Flow-GRPO: Group Relative Policy Optimization for flow models.

Algorithm (see spec_flowgrpo.md):
  1. Collect grouped stochastic SDE trajectories under a frozen old policy.
     Store (x_k, x_{k+1}, log_prob_old) for each step, and per-group normalized
     advantages from the terminal reward. A group is the set of trajectories
     that share a condition c; the terminal advantage is broadcast to every
     transition of a trajectory.
  2. Run PPO epochs over a buffer of ALL (step, trajectory) transitions:
     a. Shuffle the buffer and draw mini-batches of transitions.
     b. Recompute the full SDE drift under train/reference policies via dynamics.
     c. Build the Euler-Maruyama kernel from that drift (same factory as collection).
     d. Clipped PPO objective on the log-ratio.
     e. KL penalty: closed-form Gaussian KL between train and reference kernels.
     f. loss = -(ppo_obj - beta_kl * kl), averaged over the mini-batch.

The kernel factory and dynamics object must be the same ones used during
collection so that log-probs and drift coefficients are computed consistently;
the importance ratio is then exactly 1 before the first gradient step.

Scale note: advantages are group-normalized to unit scale, so the reward term
pushes with constant strength however far the policy has drifted. The KL against
the reference is the only global trust region (PPO clipping only bounds movement
*within* one iteration), which is why beta_kl must be O(1) rather than O(0.01).
"""

import copy
from collections.abc import Mapping, Sequence

import torch
from torch import Generator, Tensor

from diffusiongym.core import (
    FlowDynamics,
    LatentGeometry,
    RolloutRequest,
    RolloutStorage,
)
from diffusiongym.trainers.base import (
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
    TrajectoryExperience,
    check_time_grid_stability,
)
from diffusiongym.trainers.orw_cfm import _index_conditioning
from diffusiongym.types import DDBatch

# exp(_MAX_LOG_RATIO) must stay well inside float32 range; this is a safety valve
# against pathological ratios, not part of the objective.
_MAX_LOG_RATIO = 10.0


def _per_sample_conditioning(conditioning: Mapping[str, object], n: int) -> dict:
    """Conditioning entries that carry one value per batch element."""

    def is_per_sample(value: object) -> bool:
        if isinstance(value, Tensor):
            return value.ndim >= 1 and value.shape[0] == n
        return isinstance(value, (list, tuple)) and len(value) == n

    return {k: v for k, v in conditioning.items() if is_per_sample(v)}


def _labels_from(value: object, n: int) -> Tensor:
    """Turn an explicit group key into contiguous integer labels of shape (n,)."""
    if isinstance(value, Tensor):
        flat = value.reshape(n, -1) if value.ndim > 1 else value.reshape(n, 1)
        _, labels = torch.unique(flat, dim=0, return_inverse=True)
        return labels.reshape(n)
    if isinstance(value, (list, tuple)):
        lookup: dict[object, int] = {}
        labels = []
        for item in value:
            labels.append(lookup.setdefault(item, len(lookup)))
        return torch.tensor(labels, dtype=torch.long)
    raise TypeError(
        f"A group key must be a Tensor or a sequence, got {type(value).__name__}."
    )


def _grouped_advantages(
    rewards: Tensor,
    *,
    labels: Tensor,
    num_groups: int,
    valid: Tensor | None,
    epsilon: float,
) -> Tensor:
    """Group-relative advantages: (R_i - mean_g) / (std_g + eps), shape (n,).

    Only valid samples contribute to the group mean and standard deviation, and
    invalid samples receive zero advantage. Groups with fewer than two valid
    samples have no defined spread and are assigned zero advantage rather than
    an arbitrary one.
    """
    rewards = rewards.float()
    weights = torch.ones_like(rewards) if valid is None else valid.to(rewards.dtype)
    # Group labels are built on the CPU (they may come from a Python list of
    # conditioning keys), while rewards live wherever the policy does. Every
    # accumulator below must follow the rewards, or index_add_ raises on CUDA.
    labels = labels.to(rewards.device)

    def _accumulate(values: Tensor) -> Tensor:
        return torch.zeros(
            num_groups, dtype=rewards.dtype, device=rewards.device
        ).index_add_(0, labels, values)

    counts = _accumulate(weights)
    sums = _accumulate(rewards * weights)
    means = sums / counts.clamp_min(1.0)

    centered = rewards - means[labels]
    sq = _accumulate(centered.square() * weights)
    # Bessel-corrected, matching torch.std / the spec's std over the group.
    stds = (sq / (counts - 1.0).clamp_min(1.0)).sqrt()

    advantages = centered / (stds[labels] + epsilon)
    advantages = torch.where(
        counts[labels] >= 2.0, advantages, torch.zeros_like(advantages)
    )
    if valid is not None:
        advantages = torch.where(valid, advantages, torch.zeros_like(advantages))
    return advantages


class FlowGRPO[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, TrajectoryExperience[StateT]]
):
    """Flow-GRPO fine-tuning.

    The rollout batch must be laid out as ``num_groups`` contiguous blocks of
    ``group_size`` trajectories that share a condition, or carry an explicit
    per-sample group label under ``conditioning[group_key]``. Advantages are
    normalized within a group, so mixing conditions inside a block silently
    compares rewards that are not comparable; ``collect()`` checks for this.

    Parameters
    ----------
    group_size:
        Trajectories per condition group for advantage normalization. Ignored
        when ``conditioning[group_key]`` is supplied.
    ppo_epochs:
        Gradient epochs over each collected replay buffer.
    ppo_batch_size:
        Number of *transitions* per PPO mini-batch. The buffer holds all
        ``num_steps * n`` transitions and is reshuffled every epoch.
    clip_epsilon:
        PPO probability-ratio clip range.
    beta_kl:
        Coefficient for the KL penalty against the reference policy. Advantages
        are unit-scale, so this is the only global trust region; O(1) values are
        required for it to bind at all. It also *is* the target: the stationary
        policy of the objective is

            p*(x) ∝ p_ref(x) · exp(Â / beta_kl),   Â = (r - mean) / std_r,

        (the 1/T factors on the reward and KL gradients cancel), so fine-tuning
        toward p_ref · exp(lambda · r) means beta_kl ≈ 1 / (lambda * std_r).
        Note std_r is measured under the current policy and shrinks as the policy
        concentrates, which slowly sharpens the effective tilt during training.
    advantage_epsilon:
        Additive stabilizer eps_A in (R - mean) / (std + eps_A). Additive rather
        than a clamp so that a degenerate group (near-identical rewards) yields
        near-zero advantages instead of amplified reward noise.
    group_key:
        Conditioning key holding an explicit per-sample group label.
    require_stable_time_grid:
        Raise (rather than warn) when the time grid makes an Euler-Maruyama step
        expansive under the supplied dynamics.
    """

    def __init__(
        self,
        *,
        group_size: int = 4,
        ppo_epochs: int = 4,
        ppo_batch_size: int = 64,
        clip_epsilon: float = 0.2,
        beta_kl: float = 1.0,
        advantage_epsilon: float = 1e-4,
        group_key: str = "group_id",
        require_stable_time_grid: bool = True,
    ) -> None:
        if group_size < 2:
            raise ValueError(
                f"group_size must be at least 2 to define a group spread, got {group_size}."
            )
        self.group_size = group_size
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_epsilon = clip_epsilon
        self.beta_kl = beta_kl
        self.advantage_epsilon = advantage_epsilon
        self.group_key = group_key
        self.require_stable_time_grid = require_stable_time_grid

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

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_labels(
        self,
        *,
        n: int,
        conditioning: Mapping[str, object],
    ) -> tuple[Tensor, int, int]:
        """Return (labels, num_groups, n_effective).

        Groups come from ``conditioning[group_key]`` when present, otherwise from
        contiguous blocks of ``group_size``. In the latter case every per-sample
        conditioning entry must be constant inside a block.
        """
        if self.group_key in conditioning:
            labels = _labels_from(conditioning[self.group_key], n)
            num_groups = int(labels.max().item()) + 1 if n else 0
            return labels, num_groups, n

        if n < self.group_size:
            raise ValueError(
                f"Need at least group_size={self.group_size} trajectories to form a "
                f"group, got n={n}."
            )
        num_groups = n // self.group_size
        n_eff = num_groups * self.group_size
        labels = torch.arange(n_eff) // self.group_size

        for key, value in _per_sample_conditioning(conditioning, n).items():
            if not self._constant_within_groups(value, num_groups, self.group_size):
                raise ValueError(
                    f"Conditioning entry {key!r} varies inside a group. Flow-GRPO "
                    "normalizes rewards within a group of trajectories that share a "
                    "condition; lay the batch out as contiguous blocks of "
                    f"group_size={self.group_size} per condition, or pass explicit "
                    f"labels as conditioning[{self.group_key!r}]."
                )
        return labels, num_groups, n_eff

    @staticmethod
    def _constant_within_groups(
        value: object, num_groups: int, group_size: int
    ) -> bool:
        n_eff = num_groups * group_size
        if isinstance(value, Tensor):
            blocks = value[:n_eff].reshape(num_groups, group_size, -1)
            return bool(torch.all(blocks == blocks[:, :1]).item())
        if isinstance(value, Sequence):
            return all(
                value[g * group_size + i] == value[g * group_size]
                for g in range(num_groups)
                for i in range(group_size)
            )
        return True

    # ------------------------------------------------------------------
    # Time-grid stability
    # ------------------------------------------------------------------

    def _check_time_grid(
        self,
        *,
        rollout,
        dynamics: FlowDynamics[StateT],
        geometry: LatentGeometry[StateT],
    ) -> float:
        """Flag Euler-Maruyama steps whose deterministic part expands the state."""
        return check_time_grid_stability(
            rollout=rollout,
            dynamics=dynamics,
            geometry=geometry,
            require=self.require_stable_time_grid,
            algorithm=type(self).__name__,
        )

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------

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
        labels, num_groups, n_eff = self._group_labels(n=n, conditioning=conditioning)

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

        self._check_time_grid(
            rollout=rollout,
            dynamics=dynamics,
            geometry=context.environment.geometry,
        )

        advantages = _grouped_advantages(
            rollout.reward.rewards,
            labels=labels.to(rollout.reward.rewards.device),
            num_groups=num_groups,
            valid=rollout.reward.valid,
            epsilon=self.advantage_epsilon,
        )

        return TrajectoryExperience(
            rollout=rollout, advantages=advantages, dynamics=dynamics
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

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
        dynamics = experience.dynamics

        rollout = experience.rollout
        advantages = experience.advantages.to(device)
        num_steps = len(rollout.steps)
        n = len(rollout.terminal_latent)
        num_transitions = num_steps * n

        kernel_factory = context.sde_sampler.kernel_factory
        use_kl = ref_model is not None and self.beta_kl > 0.0

        totals = {"loss": 0.0, "kl": 0.0, "ratio": 0.0, "clip_frac": 0.0}
        max_log_ratio = 0.0
        steps_done = 0

        for _epoch in range(self.ppo_epochs):
            # One buffer of every (step, trajectory) transition, reshuffled per
            # epoch: a mini-batch mixes time steps, so each gradient step sees the
            # objective averaged over T as the spec prescribes rather than a
            # single time slice.
            order = torch.randperm(num_transitions)

            for start in range(0, num_transitions, self.ppo_batch_size):
                flat = order[start : start + self.ppo_batch_size]
                step_ids = flat // n
                sample_ids = flat % n

                batch = self._gather(
                    rollout=rollout,
                    step_ids=step_ids,
                    sample_ids=sample_ids,
                    advantages=advantages,
                    device=device,
                )
                x_b, x_next_b, t_b, dt_b, lp_old_b, adv_b, cond_b = batch

                # Recompute the full SDE drift under the train policy. Passing raw
                # velocity to the kernel factory would give a different mean than
                # was used during collection, making ρ = π_θ / π_θ_old wrong.
                v_new = env.predict_velocity(
                    train_model, x_t=x_b, t=t_b, conditioning=cond_b
                )
                coeffs_new = dynamics.coefficients(x=x_b, t=t_b, velocity=v_new)

                kernel_new = kernel_factory.build(
                    x=x_b,
                    t=t_b,
                    dt=dt_b,
                    drift=coeffs_new.drift,
                    diffusion=coeffs_new.diffusion,
                )
                lp_new_b = kernel_new.log_prob(x_next_b)

                log_ratio = lp_new_b - lp_old_b
                ratio = torch.exp(log_ratio.clamp(-_MAX_LOG_RATIO, _MAX_LOG_RATIO))

                obj_unclipped = ratio * adv_b
                obj_clipped = (
                    ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * adv_b
                )
                ppo_obj = torch.min(obj_unclipped, obj_clipped)

                # KL against the reference policy — both kernels must use the full
                # SDE drift (via dynamics.coefficients), not the raw velocity.
                kl = torch.zeros_like(adv_b)
                if use_kl:
                    with torch.no_grad():
                        v_ref = env.predict_velocity(
                            ref_model, x_t=x_b, t=t_b, conditioning=cond_b
                        )
                        coeffs_ref = dynamics.coefficients(x=x_b, t=t_b, velocity=v_ref)
                        kernel_ref = kernel_factory.build(
                            x=x_b,
                            t=t_b,
                            dt=dt_b,
                            drift=coeffs_ref.drift,
                            diffusion=coeffs_ref.diffusion,
                        )
                    kl = kernel_new.kl_divergence(kernel_ref)

                loss = -(ppo_obj - self.beta_kl * kl).mean()
                opt.zero_grad()
                loss.backward()
                if hasattr(train_model, "parameters"):
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
                opt.step()

                with torch.no_grad():
                    outside = (
                        (ratio < 1.0 - self.clip_epsilon)
                        | (ratio > 1.0 + self.clip_epsilon)
                    ).float()
                    totals["loss"] += loss.item()
                    totals["kl"] += kl.mean().item()
                    totals["ratio"] += ratio.mean().item()
                    totals["clip_frac"] += outside.mean().item()
                    max_log_ratio = max(max_log_ratio, log_ratio.abs().max().item())
                steps_done += 1

        assert rollout.reward is not None
        rewards = rollout.reward.rewards
        valid = rollout.reward.valid
        r_valid = rewards[valid] if valid is not None else rewards
        denominator = max(steps_done, 1)
        return {
            "loss": totals["loss"] / denominator,
            "kl": totals["kl"] / denominator,
            "ratio_mean": totals["ratio"] / denominator,
            "clip_frac": totals["clip_frac"] / denominator,
            "max_log_ratio": max_log_ratio,
            "adv_abs_mean": experience.advantages.abs().mean().item(),
            "r_mean": r_valid.mean().item(),
            "r_std": r_valid.std().item() if len(r_valid) > 1 else 0.0,
            "valid_frac": valid.float().mean().item() if valid is not None else 1.0,
        }

    def _gather(
        self,
        *,
        rollout,
        step_ids: Tensor,
        sample_ids: Tensor,
        advantages: Tensor,
        device: torch.device,
    ):
        """Collect one mini-batch of transitions spanning several time steps."""
        states: list[StateT] = []
        next_states: list[StateT] = []
        times: list[Tensor] = []
        step_sizes: list[Tensor] = []
        log_probs: list[Tensor] = []
        sample_order: list[Tensor] = []

        for k in step_ids.unique().tolist():
            selected = sample_ids[step_ids == k]
            step = rollout.steps[k]
            assert step.log_prob is not None, (
                "log_prob must be stored; set storage.log_probs=True"
            )
            assert step.x is not None, "states must be stored; set storage.states=True"

            states.append(step.x.index_select(selected))
            next_states.append(step.x_next.index_select(selected))
            times.append(step.t[selected])
            step_sizes.append(step.dt.reshape(1).expand(len(selected)))
            log_probs.append(step.log_prob[selected])
            sample_order.append(selected)

        order = torch.cat(sample_order)
        x_b = type(states[0]).concat(states).to(device)
        x_next_b = type(next_states[0]).concat(next_states).to(device)
        return (
            x_b,
            x_next_b,
            torch.cat(times).to(device),
            torch.cat(step_sizes).to(device),
            torch.cat(log_probs).to(device),
            advantages[order.to(advantages.device)].to(device),
            _index_conditioning(rollout.conditioning, order),
        )

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Hard-copy train weights into the rollout policy snapshot."""
        train = context.policies.train
        rollout = context.policies.rollout
        if hasattr(train, "state_dict") and hasattr(rollout, "load_state_dict"):
            rollout.load_state_dict(copy.deepcopy(train.state_dict()))
