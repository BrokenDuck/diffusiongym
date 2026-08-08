"""Adjoint Matching fine-tuning (https://arxiv.org/abs/2409.08861).

Algorithm (see spec_adjoint_matching.md):
  1. Roll out the train policy under the memoryless SDE (no_grad) on an interior
     time grid.
  2. Terminal lean adjoint, with terminal cost g = lambda_reward * cost:
       a_K = ∇_{x_K} g(x_K)
     For cost = -r this is -lambda_reward * ∇r, so the control pushes *up* the
     reward gradient. The sign matters: flipping it minimizes the reward.
  3. Integrate the lean adjoint backward through the REFERENCE drift:
       a_k = a_{k+1} + dt * (∇_x b_ref(x_k, t_k))^T a_{k+1}
     with the stiff kappa(t) part of that Jacobian propagated exactly rather
     than by Euler (see the loop below) — the one place this deviates from the
     spec's pseudocode, because kappa dt = 1 on the first interior step.
  4. Per-step regression targets. The optimal control is u* = -sigma(t) a, and
     the memoryless drift is b = 2 v - kappa(t) x with sigma^2 = 2 eta, so
       b_target = b_ref - sigma^2 a   <=>   v_target = v_ref - eta * a.
     Targets are stored in *velocity* space: the kappa(t) x term cancels
     analytically between prediction and target, and subtracting it numerically
     instead would lose most of the signal at small t (kappa = 1/t for
     rectified flow) — fatal in bf16.
  5. Weighted mini-batch regression of the train velocity onto those targets.
     The Adjoint Matching loss is 1/2 |u_theta + sigma a|^2 with
     u_theta = sqrt(2 / eta) (v_theta - v_ref), which equals
       (1 / eta) |v_theta - v_target|^2 = (2 / sigma^2) |v_theta - v_target|^2,
     hence the per-step weight. Trajectories and targets are stop-gradient
     quantities; gradients flow only through v_theta.

Requirements:
  - Memoryless dynamics: sigma^2 = 2*eta (MemorylessFlowSDE).
  - Differentiable terminal cost (DifferentiableTerminalCost protocol).
  - Reference policy.
  - An interior time grid: the memoryless drift carries a kappa(t) = 1/t term,
    so a grid touching t=0 makes the first Euler-Maruyama step expansive and
    destroys the rollout. collect() rejects such grids.

Resolution: Adjoint Matching needs a *finer* time grid than the other
algorithms, and gets quietly weaker rather than unstable when it does not have
one. The lean adjoint is integrated backward with explicit Euler, so its error
accumulates along the whole trajectory (measured against a closed-form ground
truth: -2.7% at t=0.95, -5.4% at t=0.86, -9.2% at t=0.76, -17% at t=0.67 and
growing), and an underestimated adjoint is an under-applied tilt. On the 2-D toy
at lambda = 2 the tilt actually achieved was 0.45 at 10 steps, 0.66 at 20, and
1.84 at 40 — only the last is right. Nothing else recovers it: more gradient
steps per rollout does not (and above ~50 it hurts, by overfitting stale
targets), and more outer iterations only does so very slowly. If a run applies
less tilt than requested, raise the step count first.
"""

from collections.abc import Mapping

import torch
from torch import Generator, Tensor

from diffusiongym.core import FlowDynamics, RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    AdjointExperience,
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
    check_time_grid_stability,
)
from diffusiongym.trainers.orw_cfm import _index_conditioning
from diffusiongym.types import DDBatch


class AdjointMatching[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, AdjointExperience[StateT]]
):
    """Adjoint Matching fine-tuning.

    Parameters
    ----------
    lambda_reward:
        Reward scale. The terminal cost supplied by the environment is treated
        as g = lambda_reward * cost, so the fine-tuned model targets
        p* ∝ p_ref · exp(lambda_reward · r) when cost = -r. Equivalently, for
        max E[r] - beta·KL, set lambda_reward = 1 / beta.
    train_steps_per_iter:
        Gradient steps per inner training loop. The regression targets are
        computed once per rollout, so large values make the update increasingly
        off-policy with respect to the trajectory distribution.
    train_batch_size:
        Mini-batch size (drawn from the pool of all trajectory steps × samples).
    require_stable_time_grid:
        Raise (rather than warn) when the time grid makes an Euler-Maruyama step
        expansive under the memoryless dynamics.
    """

    def __init__(
        self,
        *,
        lambda_reward: float = 1.0,
        train_steps_per_iter: int = 50,
        train_batch_size: int = 64,
        require_stable_time_grid: bool = True,
    ) -> None:
        if lambda_reward <= 0.0:
            raise ValueError(f"lambda_reward must be positive, got {lambda_reward}.")
        self.lambda_reward = lambda_reward
        self.train_steps_per_iter = train_steps_per_iter
        self.train_batch_size = train_batch_size
        self.require_stable_time_grid = require_stable_time_grid

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=True,
            needs_stochastic_rollout=True,
            needs_memoryless_dynamics=True,
            needs_differentiable_terminal_cost=True,
            # The reference drift is recomputed during the backward adjoint pass,
            # so the rollout only has to retain the states it was evaluated at.
            rollout_storage=RolloutStorage(states=True),
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
    ) -> AdjointExperience[StateT]:
        env = context.environment
        train_model = context.policies.train
        ref_model = context.policies.reference
        device = train_model.device

        assert ref_model is not None  # validated by validate()
        assert env.terminal_cost is not None  # validated by validate()

        # ------------------------------------------------------------------
        # 1. Forward rollout under TRAIN policy (no_grad).
        # ------------------------------------------------------------------
        request = RolloutRequest(
            time_grid=time_grid,
            storage=self.requirements.rollout_storage,
            evaluate_reward=True,
        )
        rollout = context.sde_sampler.rollout(
            environment=env,
            model=train_model,
            dynamics=dynamics,
            n=n,
            conditioning=conditioning,
            request=request,
            generator=generator,
        )

        check_time_grid_stability(
            rollout=rollout,
            dynamics=dynamics,
            geometry=env.geometry,
            require=self.require_stable_time_grid,
            algorithm=type(self).__name__,
        )

        num_steps = len(rollout.steps)
        cond = rollout.conditioning

        schedule = getattr(dynamics, "affine_schedule", None)
        if schedule is None:
            raise ValueError(
                f"{type(self).__name__} needs the interpolation schedule that "
                "defines kappa(t) to integrate the lean adjoint; the supplied "
                "dynamics exposes no `affine_schedule`. Use MemorylessFlowSDE."
            )

        # ------------------------------------------------------------------
        # 2. Terminal adjoint: a_K = ∇_{x_K} g(x_K),  g = lambda_reward * cost.
        # ------------------------------------------------------------------
        x_terminal: StateT = rollout.terminal_latent.to(device)
        x_terminal_req: StateT = x_terminal.as_leaf(True)

        with torch.enable_grad():
            costs = env.terminal_cost(x_terminal_req, conditioning=cond)  # (n,)
            a = x_terminal_req.gradient(costs.sum(), create_graph=False)

        a = a.detach() * self.lambda_reward

        terminal_adjoint_norm = (
            env.geometry.squared_norm(a, reduction="sum").sqrt().mean().item()
        )
        if terminal_adjoint_norm == 0.0:
            raise ValueError(
                "Adjoint Matching requires a differentiable terminal cost. "
                "The gradient ∇_{x_K} terminal_cost is zero everywhere."
            )

        # ------------------------------------------------------------------
        # 3+4. Lean adjoint backward through the REFERENCE drift, and the
        #      velocity target it induces at each step. Both use the same
        #      reference evaluation at (x_k, t_k).
        # ------------------------------------------------------------------
        velocity_targets: list[StateT | None] = [None] * num_steps
        loss_weights: list[Tensor | None] = [None] * num_steps

        for k in range(num_steps - 1, -1, -1):
            step = rollout.steps[k]
            dt = step.dt.to(device)
            t_k = step.t.to(device)

            x_k: StateT = step.x.to(device)
            x_k_req: StateT = x_k.as_leaf(True)

            with torch.enable_grad():
                v_ref = env.predict_velocity(
                    ref_model, x_t=x_k_req, t=t_k, conditioning=cond
                )
                coeffs_ref = dynamics.coefficients(x=x_k_req, t=t_k, velocity=v_ref)

                # <b_ref(x_k, t_k), a_{k+1}>; its state gradient is the
                # vector-Jacobian product (∇_x b_ref)^T a_{k+1}.
                inner = sum(
                    (drift_i * a_i).sum()
                    for drift_i, a_i in zip(
                        coeffs_ref.drift.state_tensors(),
                        a.state_tensors(),
                        strict=True,
                    )
                )

            vjp: StateT = x_k_req.gradient(inner, create_graph=False)  # ty: ignore[invalid-argument-type]

            # Lean adjoint ODE da/dt = -(∇_x b_ref)^T a, integrated backward from
            # t_{k+1} to t_k. The Jacobian splits into a stiff schedule part and
            # a model part,
            #
            #   (∇_x b_ref)^T a = 2 (∇_x v_ref)^T a - kappa(t) a,
            #
            # and the schedule part is solved exactly (its propagator over the
            # step is b(t_k) / b(t_{k+1})) instead of by the Euler factor
            # 1 - kappa dt. The two agree to O(dt^2), but kappa = 1/t for
            # rectified flow, so on an interior grid the very first step has
            # kappa dt = 1 and plain Euler annihilates the adjoint there.
            b_k = schedule.b(t_k)
            b_next = schedule.b(t_k + dt).clamp_min(1e-6)
            kappa_k = schedule.db_dt(t_k) / b_k.clamp_min(1e-6)
            model_vjp = vjp.detach() + a * kappa_k
            a = (a * (b_k / b_next) + model_vjp * dt).detach()

            # v_target = v_ref - eta * a_k, with eta = sigma^2 / 2 (memoryless).
            sigma_sq = coeffs_ref.diffusion.detach() ** 2
            velocity_targets[k] = (v_ref.detach() - a * (0.5 * sigma_sq)).detach()
            # 2 / sigma^2 converts the velocity error into the control-space
            # Adjoint Matching loss; dt makes the sum a Riemann integral over t.
            loss_weights[k] = (2.0 / sigma_sq.clamp_min(1e-12)) * dt.abs()

        return AdjointExperience(
            rollout=rollout,
            velocity_targets=[t for t in velocity_targets if t is not None],
            loss_weights=[w for w in loss_weights if w is not None],
            dynamics=dynamics,
        )

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: AdjointExperience[StateT],
    ) -> Mapping[str, float]:
        env = context.environment
        train_model = context.policies.train
        opt = context.optimizer
        device = train_model.device

        rollout = experience.rollout
        num_steps = len(rollout.steps)
        n = len(rollout.terminal_latent)

        all_x = [rollout.steps[k].x.to(device) for k in range(num_steps)]
        all_t = [rollout.steps[k].t.to(device) for k in range(num_steps)]
        all_targets = [target.to(device) for target in experience.velocity_targets]
        all_weights = [weight.to(device) for weight in experience.loss_weights]

        total = num_steps * n
        batch_size = min(self.train_batch_size, total)
        total_loss = 0.0

        for _ in range(self.train_steps_per_iter):
            flat_idx = torch.randint(0, total, (batch_size,))
            step_ids = flat_idx // n
            sample_ids = flat_idx % n

            # A mini-batch spans several time steps; accumulate the weighted
            # squared error over all of them and take one weighted mean, so the
            # gradient step sees the objective averaged over t rather than a
            # single time slice.
            weighted_error = torch.zeros((), device=device)
            weight_total = torch.zeros((), device=device)

            for k in step_ids.unique().tolist():
                idx = sample_ids[step_ids == k].to(device)
                x_b = all_x[k].index_select(idx)
                t_b = all_t[k][idx]
                target_b = all_targets[k].index_select(idx)
                weight_b = all_weights[k][idx]
                cond_b = _index_conditioning(rollout.conditioning, idx)

                v_new = env.predict_velocity(
                    train_model, x_t=x_b, t=t_b, conditioning=cond_b
                )
                err = env.geometry.squared_norm(v_new - target_b, reduction="mean")

                weighted_error = weighted_error + (weight_b * err).sum()
                weight_total = weight_total + weight_b.sum()

            loss = weighted_error / weight_total.clamp_min(1e-12)

            opt.zero_grad()
            loss.backward()
            if hasattr(train_model, "parameters"):
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)  # ty: ignore[call-non-callable]
            opt.step()
            total_loss += loss.item()

        # The reward is logged, not optimized directly: without it there is no
        # way to tell a working run from a sign error in the adjoint.
        metrics = {"loss": total_loss / max(self.train_steps_per_iter, 1)}
        if rollout.reward is not None:
            rewards = rollout.reward.rewards
            valid = rollout.reward.valid
            r_valid = rewards[valid] if valid is not None else rewards
            metrics["r_mean"] = r_valid.mean().item()
            metrics["r_std"] = r_valid.std().item() if len(r_valid) > 1 else 0.0
        return metrics
