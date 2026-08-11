"""Sequential Monte Carlo guidance: reweight and resample a rollout by a potential.

Solves "self-generate x ~ argmax_q E_q[a(x)] - beta*KL(q || p_theta)" at inference
time, without touching p_theta: particles are drawn from the model's own SDE and
periodically resampled in proportion to exp(a(x_hat1)/beta), where x_hat1 is a
one-step endpoint estimate re-derived from the model's own velocity prediction at
each step (no extra model call — see `PredictionConverter.to_endpoint`). This
realises the twisted target p_tilde ∝ p_theta * exp(a/beta) via the standard
Feynman-Kac / twisted-SMC construction: the potential is applied incrementally
and the increments telescope to the terminal ratio, so a resample decision at any
point along the trajectory only ever needs information already computed for the
SDE step itself.

The telescoping is what makes the intermediate `x_hat1` estimates *only* a
variance-reduction device: they steer early resampling, then cancel, and the
weight the final resample acts on is `exp(a(x_1)/beta)` at the true terminal
state. That last increment is applied explicitly at the end of `rollout()` and
is load-bearing — without it the intermediate estimates leak into the target,
and how good they are (i.e. the step count) silently decides what distribution
is sampled.

`log_potential` is supplied at `rollout()` time, not the constructor, because it
is typically bound to a specific iteration's surrogate state (see
`reward_actflow`'s acquisition reward) and this sampler has no opinion about what
it scores.
"""

from collections.abc import Callable, Mapping
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Generator, Tensor

from diffusiongym.core.dynamics import FlowDynamics
from diffusiongym.core.kernel import EulerGaussianKernelFactory, GaussianMarkovKernel
from diffusiongym.core.model import PredictionKind
from diffusiongym.core.reward import RewardBatch
from diffusiongym.core.rollout import Rollout, RolloutRequest, RolloutStep, SMCStats
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch

type Conditioning = Mapping[str, Any]
type LogPotential[StateT] = Callable[[StateT, Tensor], Tensor]
type ResampleMethod = Literal["systematic", "multinomial"]


def _index_conditioning(conditioning: Conditioning, idx: Tensor) -> dict:
    """Sub-select a conditioning dict to a batch of indices (with repeats)."""
    result: dict[str, Any] = {}
    for k, v in conditioning.items():
        if isinstance(v, Tensor):
            result[k] = v[idx]
        elif isinstance(v, list):
            result[k] = [v[i] for i in idx.tolist()]
        else:
            result[k] = v
    return result


def _systematic_resample(
    weights: Tensor, *, generator: Generator | None = None
) -> Tensor:
    """Lower-variance alternative to multinomial resampling.

    One uniform draw fixes n evenly spaced offsets into the CDF, so every
    particle with weight >= 1/n is guaranteed at least one copy — unlike
    multinomial resampling, which can and does drop such particles by chance.
    Under exactly uniform weights this returns the identity permutation.
    """
    n = weights.shape[0]
    u = torch.rand((), generator=generator, device=weights.device, dtype=weights.dtype)
    positions = (u + torch.arange(n, device=weights.device, dtype=weights.dtype)) / n
    cumsum = torch.cumsum(weights, dim=0)
    cumsum[-1] = 1.0  # guard float error so the last position always finds a slot
    return torch.searchsorted(cumsum, positions).clamp_max(n - 1)


class SMCSampler[StateT: DDBatch, RawT]:
    """SDE rollout twisted by a caller-supplied terminal-endpoint log-potential.

    Mirrors `EulerMaruyamaSampler`'s step loop exactly (same drift, same
    diffusion, same kernel), and adds: a per-step endpoint estimate from the
    velocity already computed for the step, an incremental potential update, and
    an effective-sample-size-triggered resample. The returned batch is always
    unweighted — every downstream consumer can treat it exactly like the output
    of the other two samplers.

    Requires stochastic dynamics, for the same reason `EulerMaruyamaSampler`
    does: without transition noise, resampled particles are exact duplicates
    with no way to decorrelate over later steps, and SMC silently degenerates
    into copying whichever single particle the base rollout happened to favour.
    """

    def __init__(
        self,
        geometry: LatentGeometry[StateT],
        kernel_factory: EulerGaussianKernelFactory[StateT],
        *,
        ess_threshold: float = 0.5,
        potential_every: int = 1,
        resample: ResampleMethod = "systematic",
    ) -> None:
        if not (0.0 < ess_threshold <= 1.0):
            raise ValueError(f"ess_threshold must be in (0, 1], got {ess_threshold}.")
        if potential_every < 1:
            raise ValueError(f"potential_every must be >= 1, got {potential_every}.")
        self.geometry = geometry
        self.kernel_factory = kernel_factory
        self.ess_threshold = ess_threshold
        self.potential_every = potential_every
        self.resample_method: ResampleMethod = resample

    def _resample_indices(self, logw: Tensor, *, generator: Generator | None) -> Tensor:
        weights = torch.softmax(logw, dim=0)
        if self.resample_method == "systematic":
            return _systematic_resample(weights, generator=generator)
        n = weights.shape[0]
        return torch.multinomial(weights, n, replacement=True, generator=generator)

    @staticmethod
    def _ess(logw: Tensor) -> Tensor:
        """exp(2*logsumexp(logw) - logsumexp(2*logw)) == (sum w)^2 / sum(w^2)."""
        return torch.exp(
            2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)
        )

    @torch.no_grad()
    def rollout(
        self,
        *,
        environment: Any,  # FlowEnvironment (avoid circular import)
        model: Any,  # FlowModel
        dynamics: FlowDynamics[StateT],
        n: int,
        conditioning: Conditioning,
        request: RolloutRequest,
        log_potential: LogPotential[StateT],
        generator: Generator | None = None,
    ) -> Rollout[StateT, RawT]:
        if not dynamics.stochastic:
            raise ValueError(
                "SMCSampler requires stochastic dynamics (dynamics.stochastic must "
                "be True). Without transition noise, resampled particles are exact "
                "duplicates that cannot decorrelate over later steps, and SMC "
                "silently degenerates into copying the single highest-potential "
                "draw. Use EulerMaruyamaSampler for a plain SDE rollout, or supply "
                "stochastic dynamics (e.g. AffineFlowMarginalPreservingSDE)."
            )

        device = model.device
        time_grid = request.time_grid.to(device)

        x, conditioning = environment.base_sampler.sample(
            n, conditioning=conditioning, device=device, generator=generator
        )

        converter = environment.regression.converter
        logw = torch.zeros(n, device=device)
        # logphi_prev is the potential value the *current* logw is measured
        # against. It starts at 0 (log 1): before any step, nothing is known
        # about where the trajectory is headed, so every particle is equally
        # weighted. It must travel with a particle's identity across a resample
        # (reindexed below), not reset to 0 — logw is what resets.
        logphi_prev = torch.zeros(n, device=device)

        steps: list[RolloutStep[StateT]] = []
        ess_trace: list[float] = []
        resampled_trace: list[bool] = []

        for step_idx, (t0, t1) in enumerate(pairwise(time_grid)):
            dt = t1 - t0
            t_eval = t0.clamp_min(1e-2).expand(n)

            velocity = environment.predict_velocity(
                model, x_t=x, t=t_eval, conditioning=conditioning
            )
            coeffs = dynamics.coefficients(x=x, t=t_eval, velocity=velocity)

            kernel: GaussianMarkovKernel[StateT] = self.kernel_factory.build(
                x=x,
                t=t_eval,
                dt=dt.unsqueeze(0).expand(n),
                drift=coeffs.drift,
                diffusion=coeffs.diffusion,
            )
            x_next = kernel.rsample(generator=generator)

            log_prob = kernel.log_prob(x_next) if request.storage.log_probs else None
            noise = (x_next - kernel.mean) if request.storage.noises else None

            steps.append(
                RolloutStep(
                    x=x.clone().detach() if request.storage.states else x,
                    x_next=x_next.clone().detach(),
                    t=t_eval.detach().cpu(),
                    dt=dt.detach().cpu(),
                    drift=coeffs.drift.detach() if request.storage.drifts else None,
                    diffusion=coeffs.diffusion.detach().cpu(),
                    noise=noise.detach() if noise is not None else None,
                    log_prob=log_prob.detach().cpu() if log_prob is not None else None,
                )
            )

            if step_idx % self.potential_every == 0:
                # Endpoint estimate from the *pre-step* state and the velocity
                # already computed for the drift above — no extra model call.
                x_hat1 = converter.to_endpoint(
                    prediction=velocity, kind=PredictionKind.VELOCITY, x_t=x, t=t_eval
                )
                logphi_k = log_potential(x_hat1, t_eval).to(device)
                logw = logw + (logphi_k - logphi_prev)
                logphi_prev = logphi_k

            ess = self._ess(logw)
            ess_trace.append(float(ess.item()))

            do_resample = bool(ess.item() < self.ess_threshold * n)
            resampled_trace.append(do_resample)
            if do_resample:
                idx = self._resample_indices(logw, generator=generator)
                x_next = x_next.index_select(idx)
                logphi_prev = logphi_prev.index_select(0, idx)
                conditioning = _index_conditioning(conditioning, idx)
                logw = torch.zeros(n, device=device)

            x = x_next

        # Terminal anchor. Every increment above is a *twisting* term: it steers
        # resampling early, and must cancel out of the final weight. It only
        # does so if the last potential in the telescoping sum is the one at
        # `x_1` itself, which is what Uehara et al.'s Algorithm 1 evaluates —
        # the weight update is always taken at the state actually reached, and
        # the last state reached is the terminal one. With this increment the
        # accumulated `logw` since the last resample is exactly
        # `log_potential(x_1)`, i.e. the importance weight for
        # `p_theta * exp(a/beta)` under the proposal `p_theta` the kernel
        # already is, so `potential_every` and the step count change only the
        # variance, never the target law.
        #
        # Free, and exact: at `t = 1` the endpoint estimate *is* the state, so
        # this needs no model call and no `to_endpoint` extrapolation — it is
        # the one potential evaluation in the whole rollout that carries no
        # `x_hat1` error.
        logphi_terminal = log_potential(x, time_grid[-1].expand(n)).to(device)
        logw = logw + (logphi_terminal - logphi_prev)

        # Final resample: the returned batch must be unweighted, like the
        # ODE/SDE samplers' output, or every downstream consumer would need to
        # know about `logw` to use these particles correctly.
        idx = self._resample_indices(logw, generator=generator)
        x = x.index_select(idx)
        conditioning = _index_conditioning(conditioning, idx)

        terminal_latent = x
        terminal_sample = None
        reward_batch: RewardBatch | None = None
        if request.evaluate_reward:
            terminal_sample, reward_batch = environment.evaluate_terminal(
                terminal_latent, conditioning=conditioning
            )

        stats = SMCStats(
            ess_trace=torch.tensor(ess_trace),
            resampled=torch.tensor(resampled_trace, dtype=torch.bool),
            num_resamples=int(sum(resampled_trace)),
        )

        return Rollout(
            terminal_latent=terminal_latent.detach(),
            terminal_sample=terminal_sample,
            reward=reward_batch,
            steps=steps,
            conditioning=conditioning,
            smc=stats,
        )
