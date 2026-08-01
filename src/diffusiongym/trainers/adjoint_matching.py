"""Adjoint Matching fine-tuning (https://arxiv.org/abs/2409.08861).

Algorithm:
  1. Roll out the train policy under the memoryless SDE (no_grad).
  2. Compute terminal adjoint a_K = -∇_{x_K} terminal_cost(x_K).
  3. Integrate lean adjoint backward through the REFERENCE drift:
       a_{k-1} = a_k + dt * (∇_x b_ref(x_k, t_k))^T a_k
  4. Per-step drift regression targets:
       target_drift_k = b_ref(x_k, t_k) - sigma_k^2 * a_k
  5. Mini-batch regression of train policy drift toward targets.

Requirements:
  - Memoryless dynamics: sigma^2 = 2*eta (MemorylessFlowSDE).
  - Differentiable terminal cost (DifferentiableTerminalCost protocol).
  - Reference policy.
"""

from collections.abc import Mapping
from typing import cast

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.rollout import RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    AdjointExperience,
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
)
from diffusiongym.trainers.orw_cfm import _index_conditioning
from diffusiongym.types import DDBatch


class AdjointMatching[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, AdjointExperience[StateT]]
):
    """Adjoint Matching fine-tuning.

    Parameters
    ----------
    train_steps_per_iter:
        Gradient steps per inner training loop.
    train_batch_size:
        Mini-batch size (drawn from the pool of all trajectory steps × samples).
    """

    def __init__(
        self,
        *,
        train_steps_per_iter: int = 50,
        train_batch_size: int = 64,
    ) -> None:
        self.train_steps_per_iter = train_steps_per_iter
        self.train_batch_size = train_batch_size

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=True,
            needs_stochastic_rollout=True,
            needs_memoryless_dynamics=True,
            needs_differentiable_terminal_cost=True,
            rollout_storage=RolloutStorage(
                states=True,
                drifts=True,
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
            evaluate_reward=False,
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

        num_steps = len(rollout.steps)
        cond = rollout.conditioning

        # ------------------------------------------------------------------
        # 2. Terminal adjoint: a_K = -∇_{x_K} terminal_cost(x_K).
        # ------------------------------------------------------------------
        x_terminal: StateT = rollout.terminal_latent.to(device)
        x_terminal_req: StateT = x_terminal.as_leaf(True)

        with torch.enable_grad():
            costs = env.terminal_cost(x_terminal_req, conditioning=cond)  # (n,)
            total_cost = costs.sum()
            a_K = x_terminal_req.gradient(total_cost, create_graph=False)

        a_K = a_K.detach() * -1.0

        if env.geometry.squared_norm(a_K, reduction="sum").abs().max().item() == 0.0:
            raise ValueError(
                "Adjoint Matching requires a differentiable terminal cost. "
                "The gradient ∇_{x_K} terminal_cost is zero everywhere."
            )

        # ------------------------------------------------------------------
        # 3. Lean adjoint backward through REFERENCE drift.
        # ------------------------------------------------------------------
        adjoints: list[StateT | None] = [None] * (num_steps + 1)
        adjoints[num_steps] = a_K.cpu()

        a = a_K

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
                ref_drift = coeffs_ref.drift

                a_dev: StateT = a.to(device)
                # dot product: elementwise multiply then sum all elements
                dot = sum(
                    (x * y).sum()
                    for x, y in zip(
                        ref_drift.state_tensors(),
                        a_dev.state_tensors(),
                        strict=True,
                    )
                )

            vjp: StateT = x_k_req.gradient(dot, create_graph=False, retain_graph=False)  # ty: ignore[invalid-argument-type]
            vjp = vjp.detach()

            a = a_dev + vjp * dt
            a = a.detach()
            adjoints[k] = a.cpu()

        # ------------------------------------------------------------------
        # 4. Per-step drift targets: target_k = b_ref(x_k) - sigma_k^2 * a_k.
        # ------------------------------------------------------------------
        adjoint_targets: list[StateT] = []

        with torch.no_grad():
            for k in range(num_steps):
                step = rollout.steps[k]
                t_k = step.t.to(device)
                x_k = step.x.to(device)

                v_ref = env.predict_velocity(
                    ref_model, x_t=x_k, t=t_k, conditioning=cond
                )
                coeffs_ref = dynamics.coefficients(x=x_k, t=t_k, velocity=v_ref)
                ref_drift_k = coeffs_ref.drift
                sigma_k = coeffs_ref.diffusion  # shape (n,)

                a_k = cast(StateT, adjoints[k]).to(device)
                sigma_sq = sigma_k**2
                target = ref_drift_k - a_k * sigma_sq
                adjoint_targets.append(target.cpu())

        return AdjointExperience(
            rollout=rollout,
            adjoint_targets=adjoint_targets,
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
        dynamics: FlowDynamics[StateT] = experience.dynamics

        rollout = experience.rollout
        targets = experience.adjoint_targets
        num_steps = len(rollout.steps)
        n = len(rollout.terminal_latent)

        all_x = [rollout.steps[k].x.to(device) for k in range(num_steps)]
        all_t = [rollout.steps[k].t.to(device) for k in range(num_steps)]
        all_targets = [targets[k].to(device) for k in range(num_steps)]

        total = num_steps * n
        total_loss = 0.0

        for _ in range(self.train_steps_per_iter):
            flat_idx = torch.randint(0, total, (min(self.train_batch_size, total),))
            k_indices = (flat_idx // n).tolist()
            i_indices = (flat_idx % n).tolist()

            # Group by step
            step_groups: dict[int, list[int]] = {}
            for k_i, i_i in zip(k_indices, i_indices):
                step_groups.setdefault(k_i, []).append(i_i)

            loss = torch.tensor(0.0, device=device)
            count = 0

            for k, sample_indices in step_groups.items():
                idx = torch.tensor(sample_indices, device=device)
                x_b = all_x[k][idx]
                t_b = all_t[k][idx]
                target_b = all_targets[k][idx]
                cond_b = _index_conditioning(rollout.conditioning, idx)

                # Predicted drift from train policy
                v_new = env.predict_velocity(
                    train_model, x_t=x_b, t=t_b, conditioning=cond_b
                )
                coeffs_new = dynamics.coefficients(x=x_b, t=t_b, velocity=v_new)
                pred_drift = coeffs_new.drift

                err = env.geometry.squared_norm(
                    pred_drift - target_b,
                    reduction="mean",
                )
                loss = loss + err.mean()
                count += 1

            if count > 0:
                loss = loss / count
                opt.zero_grad()
                loss.backward()
                if hasattr(train_model, "parameters"):
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)  # ty: ignore[call-non-callable]
                opt.step()
                total_loss += loss.item()

        return {
            "loss": total_loss / max(self.train_steps_per_iter, 1),
        }
