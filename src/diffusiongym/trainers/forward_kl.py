"""Forward-KL distillation of a reward-tilted teacher into the train policy.

    L(theta) = sum_k E_{x_k ~ u} KL( q_k(. | x_k) || p_theta,k(. | x_k) )

where the teacher `q ∝ rho^+ * exp(log_weight)` is built once per round from the
*lazy* rollout policy, and the roll-in `u` is a caller-supplied reference
distribution `rho^+` forward-noised to a random timestep.

Three things distinguish this from the other four trainers, and each is the
reason a fifth exists:

* **The base distribution is supplied per round, not fixed.** `ReferenceSource`
  is a flat pool of endpoints plus a draw probability, so the caller can widen
  it as it discovers new regions. Every other trainer here anchors to
  `policies.reference` or to nothing, and a geometric tilt of a fixed
  `p_theta` cannot place mass outside `supp(p_theta)`.
* **The roll-in endpoint is itself a teacher candidate.** See `_build_teacher`.
* **The teacher is frozen for `inner_epochs` gradient steps**, and the rollout
  policy is refreshed only every `lazy_interval` rounds — the timescale
  separation that keeps a simultaneously-moving generator, weighting function
  and reference from oscillating.

`log_weight` maps a batch of endpoints to a log weight and is supplied on the
experience rather than the constructor, for the same reason `SMCSampler` takes
`log_potential` at `rollout()` time: it is typically bound to one iteration's
surrogate state, and this file has no opinion about what it scores.

**The loss is computed in endpoint space**, not velocity space. Three reasons,
in order of force: `PredictionConverter.to_endpoint` from a velocity needs only
a non-zero Wronskian and is well-defined at both `t=0` and `t=1`, whereas the
reverse map divides by `a(t)` and raises near `t=1`, so a velocity-space
formulation would have to convert the target back through the singular
direction; the teacher weights are evaluated on endpoints anyway; and the
implied per-timestep weight vanishes as `t -> 1`, where a one-step teacher has
no leverage, rather than diverging there as the exact `1/sigma^2` does. The
three parameterisations otherwise differ only by a positive `t`-dependent
factor — `to_endpoint` is affine in the prediction at fixed `(x_t, t)`, and the
Euler drift is affine in the velocity — so this is a choice of loss weighting,
not of minimiser.
"""

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.model import PredictionKind
from diffusiongym.core.rollout import RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
)
from diffusiongym.types import DDBatch
from diffusiongym.utils import index_conditioning

type Conditioning = Mapping[str, Any]
type LogWeight[StateT] = Callable[[StateT], Tensor]


@dataclass(frozen=True)
class ReferenceSource[StateT]:
    """`rho^+`, flattened: endpoints plus the probability of drawing each.

    One pool rather than several named ones on purpose. Which component a row
    came from, how the components are weighted, and which rows survived a
    validity filter are all the caller's business; what this trainer needs is
    only "draw an endpoint from rho^+", and flattening keeps every one of those
    decisions out of here.
    """

    endpoints: StateT
    conditioning: Conditioning
    probs: Tensor  # shape (len(endpoints),), non-negative, normalised on use


@dataclass
class DistillExperience[StateT]:
    """Fresh rollout-policy endpoints, plus what `update()` cannot otherwise reach.

    `dynamics` and `time_grid` are carried because `update()`'s signature has
    neither and the teacher's transition kernel needs both — the same reason
    `TrajectoryExperience` and `AdjointExperience` carry `dynamics`.

    `reference` and `log_weight` are filled in by the driving loop before
    `update()` runs (via `dataclasses.replace`), because they are round-level
    quantities the loop owns.
    """

    latent: StateT
    rewards: Tensor
    valid: Tensor | None
    conditioning: Conditioning
    dynamics: FlowDynamics[StateT]
    time_grid: Tensor
    metadata: Mapping[str, Any] | None = None
    reference: ReferenceSource[StateT] | None = None
    log_weight: LogWeight[StateT] | None = None


@dataclass
class _Teacher[StateT]:
    """One round's roll-in states and their frozen regression targets."""

    x_t: StateT
    t: Tensor
    target: StateT
    conditioning: Conditioning


class ForwardKLDistillation[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, DistillExperience[StateT]]
):
    """Off-policy timestep-wise forward-KL distillation of a tilted teacher.

    Parameters
    ----------
    num_teacher_proposals:
        M — transition proposals drawn from the lazy policy per roll-in state.
        The teacher is a weighted average over these plus the roll-in's own
        endpoint, so M=0 leaves only that endpoint and reduces the whole method
        to plain flow matching on `rho^+`.
    seed_candidate_weight:
        Prior mass on the roll-in's own endpoint relative to each proposal.
        With M proposals its share of an untilted teacher is
        `w / (w + M)`; raise it to keep the reference distribution's
        contribution from shrinking as M grows. 0 removes it — see
        `_build_teacher` for why that guts the method.
    t_min, t_max:
        Roll-in timestep range. Near t=1 the endpoint-space gradient vanishes
        and those roll-ins are wasted compute; near t=0 the roll-in carries
        almost no information about its endpoint.
    roll_in_size:
        Roll-in states built per round. The teacher's cost is
        `roll_in_size * M` weight evaluations, paid once per round rather than
        once per gradient step.
    inner_epochs:
        Gradient steps taken against that frozen teacher.
    lazy_interval:
        K_lazy — rounds between rollout-policy refreshes.
    """

    def __init__(
        self,
        *,
        num_teacher_proposals: int = 8,
        seed_candidate_weight: float = 1.0,
        t_min: float = 0.1,
        t_max: float = 0.95,
        roll_in_size: int = 256,
        inner_epochs: int = 50,
        batch_size: int = 64,
        lazy_interval: int = 5,
    ) -> None:
        if num_teacher_proposals < 0:
            raise ValueError(
                f"num_teacher_proposals must be >= 0, got {num_teacher_proposals}."
            )
        if seed_candidate_weight < 0:
            raise ValueError(
                f"seed_candidate_weight must be >= 0, got {seed_candidate_weight}."
            )
        if not 0.0 < t_min < t_max < 1.0:
            raise ValueError(f"need 0 < t_min < t_max < 1, got {t_min}, {t_max}.")
        if roll_in_size < 1:
            raise ValueError(f"roll_in_size must be >= 1, got {roll_in_size}.")
        if inner_epochs < 1:
            raise ValueError(f"inner_epochs must be >= 1, got {inner_epochs}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        if lazy_interval < 1:
            raise ValueError(f"lazy_interval must be >= 1, got {lazy_interval}.")

        self.num_teacher_proposals = num_teacher_proposals
        self.seed_candidate_weight = seed_candidate_weight
        self.t_min = t_min
        self.t_max = t_max
        self.roll_in_size = roll_in_size
        self.inner_epochs = inner_epochs
        self.batch_size = batch_size
        self.lazy_interval = lazy_interval
        self._iterations = 0

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            # rho^+ arrives as endpoint samples, never as a density or a drift,
            # so there is nothing to query a frozen model for. make() would
            # otherwise build a third copy of the network for nothing.
            needs_reference_policy=False,
            # Decisive, not cosmetic: the kernel's variance is
            # diffusion**2 * dt and ProbabilityFlowODE reports diffusion = 0,
            # so under deterministic dynamics all M proposals coincide, the
            # teacher weights go uniform, and the tilt silently does nothing.
            needs_stochastic_rollout=True,
            rollout_storage=RolloutStorage(),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        n: int,
        time_grid: Tensor,
        conditioning: Conditioning,
        generator: Generator | None = None,
    ) -> DistillExperience[StateT]:
        """Endpoints from the lazy rollout policy — `rho`'s "current" component.

        `dynamics` and `time_grid` are stashed on the experience because
        `update()` receives neither and the teacher needs both.
        """
        rollout = context.sde_sampler.rollout(
            environment=context.environment,
            model=context.policies.rollout,
            dynamics=dynamics,
            n=n,
            conditioning=conditioning,
            request=RolloutRequest(
                time_grid=time_grid,
                storage=self.requirements.rollout_storage,
                evaluate_reward=True,
            ),
            generator=generator,
        )
        assert rollout.reward is not None
        return DistillExperience(
            latent=rollout.terminal_latent,
            rewards=rollout.reward.rewards,
            valid=rollout.reward.valid,
            conditioning=rollout.conditioning,
            dynamics=dynamics,
            time_grid=time_grid,
            metadata=rollout.reward.metadata,
        )

    # ------------------------------------------------------------------
    # Teacher
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_teacher(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: DistillExperience[StateT],
        reference: ReferenceSource[StateT],
        log_weight: LogWeight[StateT],
    ) -> tuple[_Teacher[StateT], dict[str, float]]:
        """Roll in from `rho^+`, then average the tilted teacher's candidates.

        `sum_m w_m ||y_theta - c_m||^2 = ||y_theta - c_bar||^2 + const` with
        `c_bar = sum_m w_m c_m`, so the per-candidate loss and the regression
        onto the weighted mean have identical gradients. Only `c_bar` is
        retained, which is what lets the teacher be built once per round
        instead of once per gradient step.

        **The roll-in's own endpoint is candidate 0, and that is load-bearing.**
        Every other candidate is the lazy policy's own denoiser output, so for
        a roll-in noised from a point outside `supp(p_rollout)` the target
        would be the lazy policy's guess — pulling *back* toward where it
        already puts mass, and nothing ever expands. The roll-in endpoint is
        also the one candidate that is an exact draw from `q(x_1 | x_t)` under
        `rho^+`, since `x_t` was produced by forward-noising it. Including it
        is what anchors the student to the reference distribution rather than
        to the current generator.
        """
        env = context.environment
        model = context.policies.rollout
        device = model.device
        grid = experience.time_grid.to(device)
        num_proposals = self.num_teacher_proposals
        size = self.roll_in_size

        # 1. Draw endpoints from rho^+.
        probs = reference.probs.to(device).clamp_min(0)
        idx = torch.multinomial(
            probs / probs.sum().clamp_min(1e-12), size, replacement=True
        )
        x1 = reference.endpoints.to(device).index_select(idx)
        cond = index_conditioning(dict(reference.conditioning), idx)

        # 2. Roll in: a grid index per row, so the loss is literally
        #    timestep-wise on the discretisation the sampler uses and `dt` is
        #    never ambiguous.
        usable = ((grid >= self.t_min) & (grid <= self.t_max)).nonzero().flatten()
        usable = usable[usable < grid.numel() - 1]
        if usable.numel() == 0:
            usable = torch.zeros(1, dtype=torch.long, device=device)
        k = usable[torch.randint(usable.numel(), (size,), device=device)]
        t, t_next = grid[k], grid[k + 1]
        x_t = env.forward_process.make_batch(x1, conditioning=cond, t=t).x_t

        # 3. Candidates: the roll-in endpoint, then M transition proposals
        #    mapped to endpoints. Row `m * size + j` is candidate `m` of
        #    roll-in `j` (`repeat`, never `repeat_interleave` — every reshape
        #    below reads that layout and only that one).
        #
        #    Each proposal contributes only its *displacement* from the
        #    kernel's mean, added to the lazy policy's own endpoint estimate at
        #    `(x_t, t)`. Mapping proposals to endpoints at `t_next` directly
        #    would instead leave an O(dt) offset — the Euler mean is
        #    `x + drift*dt` and the drift is not the velocity (it carries the
        #    `kappa*x` and `c(t)` corrections), so the candidate average would
        #    sit a little away from `x_hat1_theta_bar(x_t, t)` and an *untilted*
        #    teacher would still drag the student off its own prediction.
        #    Centring makes `theta = theta_bar` under uniform weights an exact
        #    stationary point, which is the one correctness property of this
        #    loss that can be checked without a ground-truth target.
        converter = env.regression.converter
        candidates = [x1]
        if num_proposals > 0:
            velocity = env.predict_velocity(model, x_t=x_t, t=t, conditioning=cond)
            base = converter.to_endpoint(
                prediction=velocity, kind=PredictionKind.VELOCITY, x_t=x_t, t=t
            )
            coeffs = experience.dynamics.coefficients(x=x_t, t=t, velocity=velocity)
            kernel = context.sde_sampler.kernel_factory.build(
                x=x_t,
                t=t,
                dt=t_next - t,
                drift=coeffs.drift,
                diffusion=coeffs.diffusion,
            )
            tile = torch.arange(size, device=device).repeat(num_proposals)
            proposals = type(x_t).concat(
                [kernel.rsample() for _ in range(num_proposals)]
            )
            displacement = proposals - kernel.mean.index_select(tile)
            # `to_endpoint` is linear in (prediction, x_t) jointly, so a zero
            # prediction maps a pure displacement to its endpoint-space image
            # without this file needing to know the schedule.
            candidates.append(
                base.index_select(tile)
                + converter.to_endpoint(
                    prediction=displacement * 0.0,
                    kind=PredictionKind.VELOCITY,
                    x_t=displacement,
                    t=t_next.repeat(num_proposals),
                )
            )
        stacked = type(x1).concat(candidates)

        # 4. Tilt, then average.
        logits = log_weight(stacked).to(device).view(num_proposals + 1, size)
        logits[0] = logits[0] + torch.log(
            torch.tensor(self.seed_candidate_weight, device=device).clamp_min(1e-12)
        )
        weights = torch.softmax(logits, dim=0)

        rows = torch.arange(size, device=device)
        target = stacked.index_select(rows) * weights[0]
        for m in range(1, num_proposals + 1):
            target = target + stacked.index_select(rows + m * size) * weights[m]

        metrics = {
            "teacher_ess": float((1.0 / weights.square().sum(0)).mean().item()),
            "teacher_seed_weight": float(weights[0].mean().item()),
        }
        return _Teacher(x_t=x_t, t=t, target=target, conditioning=cond), metrics

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: DistillExperience[StateT],
    ) -> Mapping[str, float]:
        if experience.reference is None or experience.log_weight is None:
            raise ValueError(
                "ForwardKLDistillation.update() needs both a ReferenceSource "
                "and a log_weight. This algorithm is driven by a loop that "
                "owns the expanding reference distribution and the surrogates "
                "the tilt is computed from, and attaches them to the "
                "experience before update() runs — a bare diffusiongym loop "
                "cannot supply them."
            )

        teacher, metrics = self._build_teacher(
            context=context,
            experience=experience,
            reference=experience.reference,
            log_weight=experience.log_weight,
        )

        env = context.environment
        model = context.policies.train
        opt = context.optimizer
        device = model.device
        converter = env.regression.converter
        total_loss = 0.0
        for _ in range(self.inner_epochs):
            idx = torch.randint(self.roll_in_size, (self.batch_size,), device=device)
            x_t = teacher.x_t.index_select(idx)
            t = teacher.t[idx]
            cond = index_conditioning(dict(teacher.conditioning), idx)

            velocity = env.predict_velocity(model, x_t=x_t, t=t, conditioning=cond)
            x1_theta = converter.to_endpoint(
                prediction=velocity, kind=PredictionKind.VELOCITY, x_t=x_t, t=t
            )
            # Not env.velocity_error: numerically the same reduction, but these
            # are endpoints and the name would be a lie at every call site.
            loss = env.geometry.squared_norm(
                x1_theta - teacher.target.index_select(idx), reduction="mean"
            ).mean()

            opt.zero_grad()
            loss.backward()
            if hasattr(model, "parameters"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ty: ignore[call-non-callable]
            opt.step()
            total_loss += loss.item()

        return {"loss": total_loss / self.inner_epochs, **metrics}

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Hard-copy train -> rollout every `lazy_interval` calls.

        The counter advances once per call, i.e. once per outer round, so
        `lazy_interval` is in rounds rather than gradient steps — the more
        useful unit given that one round already takes `inner_epochs` steps
        against a frozen teacher.
        """
        self._iterations += 1
        if self._iterations % self.lazy_interval != 0:
            return
        train = context.policies.train
        rollout = context.policies.rollout
        if hasattr(train, "state_dict") and hasattr(rollout, "load_state_dict"):
            rollout.load_state_dict(copy.deepcopy(train.state_dict()))
