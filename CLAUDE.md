# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this package is

`diffusiongym` implements a deliberately narrow **affine-Gaussian flow-matching environment** used to reward-fine-tune pretrained flow models (toy models, SD3.5-style image latents, FlowMol Gaussian molecular states). It is a git submodule of the `reward-actflow` monorepo (checked out at `/home/jonat/reward-actflow/packages/diffusiongym`), managed as a `uv` workspace member from the repo root.

The design intent used to be written out in `specs.md`, `specs_additional.md`, `specs_types.md`, `specs_test_example.md` and `spec_flowgrpo.md`; **none of those files exist in this repo or the monorepo any more**, so what follows is the record. The core rule: only support affine Gaussian paths `x_t = a(t) z + b(t) x_1` with a Gaussian base, scalar state-independent diffusion `g(t)`, and Euler/Euler–Maruyama integration. Generalizing beyond that (non-Gaussian bases, non-affine interpolants, state-dependent diffusion, manifold states, adaptive solvers) is explicitly out of scope until an actual experiment needs it — don't add it speculatively.

## ⚠️ Two parallel architectures — use `core/`, not `environments/`

The codebase is mid-refactor. There are **two independent implementations** living side by side:

- **`src/diffusiongym/core/` — current, working architecture.** This is what `src/diffusiongym/__init__.py`, `trainers/`, `toy/`, `tests/`, and `examples/` actually import and exercise. Build all new work on top of this.
- **`src/diffusiongym/environments/`, `base_models.py`, `train.py` — legacy/broken.** Not imported by the package `__init__.py`. Do not build on it, and don't assume it works without checking first — if you touch it, expect to either finish deleting it or repair it, not patch around it.
- **`make.py` and `registry.py` — rewritten against `core/`, current.** They were previously part of the legacy stack (`import diffusiongym.make` raised `NameError`); they now assemble the `core/` interfaces. See "Assembly" below.

`src/diffusiongym/core/__init__.py` states explicitly: "It is separate from the legacy `environments/` and `train.py` code." Treat that as authoritative.

## Commands

Run everything through `uv` from this directory (`packages/diffusiongym`); it resolves against the monorepo's root `uv.lock`.

```bash
uv run pytest                          # full test suite (183 passing, 2 xfailed)
uv run pytest tests/test_core.py       # core framework unit tests only
uv run pytest tests/test_trainers_smoke.py   # end-to-end smoke tests per algorithm
uv run pytest tests/test_core.py::TestGaussianMarkovKernel::test_kl_nonnegative  # single test

uv run ruff check src tests            # lint (repo currently has pre-existing lint findings)
uv run ruff format src tests           # format
```

`pyproject.toml` declares a `[tool.mypy]` strict config (`files = ["diffusiongym"]`), but `mypy` is not installed in this environment/workspace — don't assume a `mypy` command works without first confirming the binary is available.

`.pre-commit-config.yaml` shells out to `pixi run fmt` / `pixi run lint`; there is no `pixi.toml` in this repo or the monorepo root, so pre-commit hooks as configured will fail here — use the `uv run ruff ...` commands above instead.

No build/lint/test commands are defined at the monorepo root beyond the shared `uv.lock`; there's no separate root Makefile to defer to.

## Core architecture (`src/diffusiongym/core/`)

Everything is generic over a state type `StateT: DDBatch` (see `types/batch.py`) — a batch of continuous latent states supporting batched arithmetic, indexing, and concatenation. `types/tensor.py`'s `DDTensor` is the simple unconstrained implementation used by the toy models; a structured molecular state would implement the same `DDBatch` protocol.

Three concepts that are easy to conflate are kept as **separate objects** on purpose:

| Concept | Type | Meaning |
|---|---|---|
| Interpolation schedule | `AffineSchedule` (`core/schedule.py`) | `a(t)`, `b(t)` and their derivatives defining the training path `x_t = a(t)·x_base + b(t)·x_data` |
| Diffusion schedule | `ScalarDiffusionSchedule` (`core/schedule.py`) | `g(t)`, the SDE noise coefficient during sampling |
| Time grid | plain `Tensor` (e.g. `torch.linspace(...)`), owned by the caller | numerical integration points |

Key modules and how they compose:

- **`core/schedule.py`** — `AffineSchedule` (default: `RectifiedFlowSchedule`, `a(t)=1-t, b(t)=t`) and `ScalarDiffusionSchedule` (`ConstantDiffusionSchedule`, `MemorylessDiffusionSchedule`).
- **`core/space.py`** — `LatentGeometry` protocol: projection onto constrained subspaces, Gaussian sampling, and squared-norm reductions (`"mean"` for training loss, `"sum"` for Gaussian log-probability — these are *not* interchangeable). `TensorGeometry` is the unconstrained Euclidean implementation.
- **`core/process.py`** — `AffineGaussianForwardProcess`: samples `t`, builds `x_t` and the conditional target velocity `u_t* = da/dt·x_base + db/dt·x_data` for training.
- **`core/model.py`** — `FlowModel` protocol + `PredictionConverter`, which converts a model's native output (`VELOCITY`, `ENDPOINT`, or `NOISE`) into the canonical path-velocity field that every algorithm regresses against. `VelocityRegression` is the shared predict+error primitive used by all trainers. Endpoint/noise conversion is singular at the schedule boundary (`t≈0` or `t≈1`) — always sample interior times.
- **`core/dynamics.py`** — `FlowDynamics` turns a predicted velocity into SDE drift/diffusion coefficients. Three profiles: `ProbabilityFlowODE` (deterministic), `AffineFlowMarginalPreservingSDE` (stochastic, marginal-preserving), `MemorylessFlowSDE` (stochastic + memoryless, required by Adjoint Matching to avoid initial-value bias — `g(t) = sqrt(2·eta(t))`).
- **`core/kernel.py`** — `GaussianMarkovKernel` / `EulerGaussianKernelFactory`: the *same* kernel object produces both the rollout sample (`rsample`) and its likelihood (`log_prob`), which is what makes Flow-GRPO's importance ratios exact.
- **`core/rollout.py`** — `EulerODESampler` (requires deterministic dynamics) and `EulerMaruyamaSampler` (requires stochastic dynamics) produce a `Rollout` of `RolloutStep`s. `RolloutStorage` flags gate what gets retained per step (states/noises/drifts/log_probs) — only turn on what the algorithm actually needs, to control memory.
- **`core/smc.py`** — `SMCSampler`, a third sampler alongside the two above: the same Euler-Maruyama step (requires stochastic dynamics, for the same reason `EulerMaruyamaSampler` does), plus incremental resampling toward a caller-supplied `log_potential(x_hat1, t)` evaluated on a per-step endpoint estimate (`PredictionConverter.to_endpoint`, the velocity-to-endpoint inverse alongside `to_velocity`). This solves "sample from `p_theta` tilted by a potential, without touching `p_theta`" at inference time — no new path family, no new dynamics profile, so it stays inside this file's scope rule. The potential itself is caller-defined; `core/` has no opinion about what it scores. Returns a `Rollout` with `smc: SMCStats` (ESS trace, resample count) attached; the returned particles are always unweighted, like the other two samplers' output.

  **Verified correct in its default configuration, with two known defects.** `tests/test_smc_correctness.py` checks it against a closed-form ground truth (`toy/analytic1d.py`) rather than against direction-of-movement: with `potential_every=1` on an interior grid it reproduces the exact exponentially tilted mixture — mean, variance, per-mode masses and full CDF — to Monte-Carlo error, and `examples/check_smc.py` draws the same comparison. What is broken, each pinned by an `xfail(strict=True)` test:

  - **A `-inf` log-potential silently produces NaN weights.** `logw` starts at 0, so the first increment is `-inf - 0 = -inf` for every infeasible particle and the next is `-inf - (-inf) = NaN`. ESS goes NaN, `NaN < threshold` is `False` so no resample ever fires, and the final resample indexes through a NaN CDF. Measured: 0% constraint satisfaction, 1 distinct particle out of 8192, no error raised. Hard constraints are a headline use of inference-time guidance, so express them as a large finite negative value until this is fixed.
  - **`steps` is not re-indexed on resampling.** Only the live particles and `conditioning` are, so after the first resample `steps[k].x_next` and `steps[k+1].x` describe different particles and `terminal_latent` does not match `steps[-1].x_next`. Harmless while only `terminal_latent` is consumed; a trap for anything reading trajectories (Flow-GRPO's per-step log-probs, Adjoint Matching's backward pass), which the "interchangeable with the other two samplers' output" framing above invites.

  **Fixed: the twisting now anchors at the terminal state.** `rollout()` applies a final increment `log_potential(x_1) - logphi_prev` before the final resample, so the increments telescope to `exp(a(x_1)/beta)` — the importance weight for `p_theta * exp(a/beta)` under the proposal the kernel already is (Uehara et al.'s Algorithm 1 evaluates the weight update at the state actually reached; with `q = p^pre` the transition ratio cancels and only that value difference remains). It is free and exact: at `t = 1` the endpoint estimate *is* the state, so no model call and no `to_endpoint` extrapolation. `potential_every` and the step count now change only the variance, never the target law. Before the fix the intermediate estimates leaked into the target, and on `reward_actflow`'s toy at 16 steps — where `corr(sigma(x_hat1), sigma(x_1))` is 0.05 — SMC realised 1.6% of the reweighting its potential offered; after, 70%. **Consequence to watch:** the weights are now genuinely non-uniform, so the spec's `alpha -> 0` collapse caveat is live. Measured at n=16 on that toy, distinct terminal particles were 15.0/16 at `acq_beta=1`, 14.1 at 0.5, 8.7 at 0.2, 2.8 at 0.05, 1.4 at 0.01 — de-duplicate before spending a per-sample verifier budget, or keep `acq_beta >= 0.5`.
- **`core/refine.py`** — `local_proposals` + `tail_time_grid`: forward-noise clean seeds to an intermediate level `s` and denoise back under the model's own SDE, `L` independent times per seed, returning both `z` at level `s` and the endpoint. A *local* mutation operator where `SMCSampler` is a global one, with `tail_time_grid` as the mutation-radius dial. Requires stochastic dynamics for the usual reason.

  **It performs no selection, and that is a deliberate reversal.** An earlier `RefinementSampler` here tilted the denoising itself, picking among `L` proposals at every step in proportion to `exp(log_potential(x_hat1))`. That targeted no distribution at all — its own docstring said so — and forced a temperature knob onto a sampler with no business having an opinion. Selection now belongs to the caller and is applied once to the whole candidate set, which is what lets `reward_actflow`'s CGD make it an optimisation step in probability space. If you need per-step guidance toward a *specified* law, that is `SMCSampler`, which is correct for it.
- **`core/rescale.py`** — `TemporalScoreRescaling`: Temporal Score Rescaling (Xu et al., arXiv:2510.01184), a *local* sampling temperature. Multiplies the score by `r_t(k, sigma) = (SNR·sigma² + 1) / (SNR·sigma²/k + 1)`, which runs from `1` at the noise end to `k` at the data end, and carries that through the affine change of variables to velocity space. `k > 1` sharpens, `k < 1` broadens, `k = 1` is the exact identity.

  Two things about it. **The schedule is the mechanism, not a detail.** Scaling the score by a *constant* — the obvious way to sample colder or hotter — over-suppresses exploration while the model is still choosing a mode and under-suppresses it near the data, which biases samples onto central modes and drops peripheral ones (the paper's Theorem C.1: no prior generates a temperature-scaled law that way). Leaving `r_t = 1` at high noise keeps mode *selection* untouched and rescales only the within-mode spread, so the target is `sum_m w_m N(mu_m, Sigma_m/k)` — same weights, rescaled covariances. It is therefore not `p^(1/T)`, which reweights the modes too.

  **It is a `FlowModel` wrapper, not a sampler or a dynamics profile.** It declares `prediction_kind = VELOCITY` and returns velocity, so every sampler inherits it by being handed the wrapper in place of the model, and no sampler needed a new argument. Sampling only — it holds no parameters and exposes no `state_dict`, because rescaling the field a *training* loss regresses onto would change what is being fit rather than how it is sampled. `_coefficients` evaluates an algebraically cancelled form, so the `db/b` pole at `t = 0` never appears and no interior-time restriction applies.
- **`core/ot.py`** — `sinkhorn_potentials` / `sinkhorn_cost` / `sinkhorn_divergence_potential`: entropic OT on a cost matrix. Plain tensors in and out, no `DDBatch` anywhere, so it is data-type abstract by construction — the caller decides what a point is and supplies `cost[i, j]`. It exists for **first variations**: the dual potential `f` at `x_i` *is* `delta W_c(nu, rho) / delta nu`, so an optimal-transport locality term costs one scaling loop rather than a critic network and its training loop. Log-domain throughout (`exp(-C/eps)` underflows at every useful `eps`), and debiased by default — the entropic bias otherwise makes a cloud look local *to itself*, which reads downstream as the penalty working. Checked against `scipy.optimize.linear_sum_assignment`, exact for equal-size uniform marginals, in `tests/test_ot.py`; `eps = 0.05 * mean(C)` lands within 3%, always high.
- **`core/reward.py`** — two distinct protocols: `RewardEvaluator` (black-box, used by every algorithm) vs. `DifferentiableTerminalCost` (required only by Adjoint Matching; a black-box reward is not sufficient because it may have zero gradients).
- **`core/environment.py`** — `FlowEnvironment` is an **immutable** facade bundling geometry, base sampler, forward process, regression, codec, and reward. It deliberately owns **no policies** — those live in `PolicyBundle` (`train`/`rollout`/`reference` models), owned by the fine-tuning algorithm, so multi-algorithm comparisons and checkpointing don't get tangled with mutable environment state.

## Fine-tuning algorithms (`src/diffusiongym/trainers/`)

All five algorithms implement `FineTuningAlgorithm` (`trainers/base.py`) with the same lifecycle:

```
validate()   → check FineTuningRequirements against context/dynamics, fail fast with a clear message
collect()    → sample online experience under the current rollout policy
update()     → compute loss, take a gradient step on the train policy, return metrics
synchronize_rollout_policy() → optional EMA/hard-copy of train → rollout (no-op by default)
```

Each algorithm declares its own `FineTuningRequirements` (needs reference policy? stochastic rollout? memoryless dynamics? differentiable terminal cost?) and its own experience dataclass, because each needs different trajectory data:

- **`orw_cfm.py`** (`ORWCFM`) — Online Reward-Weighted CFM **-W2** (`algorithm_specs/orw_cfm_w2.md`). Deterministic ODE rollout under `policies.rollout` → EMA-normalized reward → exponential importance weights → flow-matching regression on endpoints (`EndpointExperience`), plus `alpha_w2 * ||v_θ - v_ref||²` against the frozen reference, plus a rollout-policy refresh every `rollout_update_interval` iterations. **Both of those last two are load-bearing and were originally absent:** without the refresh the rollout policy never leaves the pretrained weights, so the method is not online — it repeatedly refits `p_base·exp(λr)` from base samples, which looks perfect near the base model and degenerates once importance weights do; with the refresh but no W2 the online iteration `p_{k+1} ∝ p_k·exp(τ·z_k)` has no stationary point and collapses (measured: E[r] → 11.9 against an analytic 2.05, all mass on one mode). `alpha_w2` needs a reference policy. Note the W2 penalty is **not** a KL term, so this method targets a W2-regularized optimum, *not* the KL-tilted `p* ∝ p_base·exp(λr)` — comparing it against `tilted_target_*` is only approximate.
- **`diffusion_nft.py`** (`DiffusionNFT`) — DPO-style contrastive fine-tuning. EMA rollout policy collects endpoints; rewards map to optimality probabilities; positive/negative blended velocity predictions are regressed toward/away from the FM target (`EndpointExperience`).
- **`flow_grpo.py`** (`FlowGRPO`) — Group Relative Policy Optimization. Requires stochastic dynamics + a reference policy. Collects grouped stochastic SDE trajectories with per-step log-probs (`TrajectoryExperience`), then runs PPO epochs with a clipped objective plus a closed-form Gaussian KL penalty against the reference kernel. The kernel factory/dynamics used at update time must match collection time. Two things bite here and are enforced/documented in the code: (a) advantages are unit-normalized, so `beta_kl` is the *only* global trust region (PPO clipping bounds movement within an iteration only) and it sets the target — the stationary policy is `p_ref · exp(Â / beta_kl)`, i.e. `lambda_eff = 1 / (beta_kl · std_r)`; (b) the marginal-preserving drift carries a `kappa(t) = 1/t` term, so a time grid touching `t=0` makes the first Euler–Maruyama step expansive — `collect()` rejects such grids. Use an interior grid (`linspace(0, 1, T+2)[1:]`) and `ScaledMemorylessDiffusionSchedule` to pick the SDE noise level `a` (Flow-GRPO does not need `MemorylessFlowSDE`, which pins the stiffest `a = sqrt(2)`).
- **`adjoint_matching.py`** (`AdjointMatching`) — Requires `MemorylessFlowSDE`, a `DifferentiableTerminalCost`, and an interior time grid (same `kappa(t) = 1/t` stiffness as Flow-GRPO, checked by the shared `check_time_grid_stability` in `trainers/base.py`; memoryless pins the noisiest `a = sqrt(2)`, so it is the more fragile of the two). Rolls out under the train policy (no grad), sets the terminal adjoint to `a_K = +∇_x g` with `g = lambda_reward · terminal_cost` — **the sign is load-bearing: flipping it trains the model to minimize the reward, silently** — integrates the lean adjoint backward through the *reference* drift, and regresses the train-policy *velocity* toward `v_ref - eta·a_k` weighted by `2·dt/sigma²` (`AdjointExperience`). Targets are velocities, not drifts: the `kappa(t)·x` term cancels analytically between prediction and target, and subtracting it numerically instead destroys the signal at small `t` in low precision. The trainer owns the reward scale via `lambda_reward` (the terminal cost supplies the shape, `lambda_reward` the tilt).

  **It needs a finer time grid than the other three, and degrades quietly when it doesn't get one.** The lean adjoint is integrated backward with explicit Euler, so its error accumulates along the trajectory and an underestimated adjoint is an under-applied tilt — not instability, just a weaker result that looks like a converged run. Measured on the toy at `lambda = 2`, the tilt actually achieved was 0.45 at 10 steps, 0.66 at 20, 1.84 at 40. Nothing else recovers it: `train_steps_per_iter` does not (above ~50 it *hurts*, overfitting stale targets — `r^2` fell to 0.17), and more outer iterations only does so very slowly (0.44 → 0.73 over 60 → 300). Raise the step count first. A second-order (Heun) adjoint integrator would buy the same accuracy at fewer steps and is the obvious optimization if AM's cost ever matters.

- **`forward_kl.py`** (`ForwardKLDistillation`) — timestep-wise forward-KL distillation of a reward-tilted teacher, and the only one of the five whose base distribution is **supplied per round** rather than fixed. `ReferenceSource` is a flat pool of endpoints plus a draw probability, so the caller can widen it as it discovers new regions; a geometric tilt of a fixed `p_theta` cannot place mass outside `supp(p_theta)`, which is why none of the other four can be used for set expansion. Two details are load-bearing and both were originally absent: the roll-in's own endpoint is candidate 0 of the teacher (without it, a roll-in from outside `supp(p_rollout)` regresses onto the lazy policy's guess and nothing ever expands), and each proposal contributes only its *displacement* from the kernel mean, which makes `theta = theta_bar` under uniform weights an exact stationary point — the one correctness property of this loss checkable without a ground-truth target, and `test_forward_kl.py` checks it. Loss is in endpoint space; see the module docstring for the three reasons.

When adding a sixth algorithm, follow this same pattern rather than introducing a new lifecycle shape.

### Comparing algorithms: knobs are not on the same scale

Three of the four normalize the reward by its own spread before applying their knob, so the tilt they actually target is reward-dependent. **A fixed knob across reward families silently compares different target distributions** — on the 2-D toy, `std_r` is 2.0 for the linear reward but 6.4 for the quadratic one, so a hard-coded `beta_kl=0.5` targets λ=1 on one and λ=0.31 on the other, which reads as "Flow-GRPO is worse at quadratic" when it is only being asked for less.

| algorithm | knob | effective λ |
|---|---|---|
| ORW-CFM | `temperature` | `temperature / std_r` |
| Flow-GRPO | `beta_kl` | `1 / (beta_kl · std_r)`, and **rises during training** as the policy concentrates and `std_r` shrinks |
| DiffusionNFT | `beta` | none — `r = 0.5 + 0.5·clamp(r_norm, ±1)` is a *linear, saturating* tilt, so `beta` is a velocity-space step size, not an inverse temperature |
| AdjointMatching | `lambda_reward` | `lambda_reward` exactly (no normalization) |

`examples/finetune_2d_gmm.py` calibrates via `estimate_reward_std()` and prints the effective λ per run. Do the same in any new comparison, and prefer `mode_weight_error` against the analytic `tilted_target_*` over eyeballing sample scatter plots — at λ=1 the box reward's analytic target is `P(A)` 0.0008 → 0.0021, i.e. *visually identical to the base model*, so the box sample plot cannot distinguish success from failure (and at n=64 rollouts the reward is observed ~0.12 times per iteration, so there is no signal to learn from either).

**Only Flow-GRPO and Adjoint Matching target `p* ∝ p_ref·exp(λr)`** — the distribution `toy/gmm2d.py`'s `tilted_target_*` computes. They are the two that anchor to a reference via a KL term. ORW-CFM-W2 anchors via a *W2* surrogate, so its optimum is a different distribution; DiffusionNFT has no anchor at all and its `beta` is an implicit-guidance scale, not an inverse temperature (its own spec says so), so its iterates keep moving — measured on the linear reward, it crosses the analytic target around iteration 400 and then overshoots indefinitely. Do not read "matches the target at iteration N" as convergence for those two, and only benchmark against `tilted_target_*` for Flow-GRPO and Adjoint Matching.

The specs for all four live in `algorithm_specs/`. Read the relevant one before changing a trainer — three of the four bugs found so far were *omissions* that left a plausible-looking algorithm behind (a flipped adjoint sign, a rollout policy that never refreshed, a missing regularizer), and none of them showed up as a crash, a bad loss curve, or an obviously wrong sample plot.

## Assembly (`make.py`, `registry.py`)

`diffusiongym.make()` turns three registry ids into a `FineTuningSetup` (environment, context, algorithm, dynamics, time grid) and calls `algorithm.validate()` before returning:

```python
setup = diffusiongym.make(
    modality="toy/gmm2d", reward="toy/linear",
    algorithm="adjoint_matching", discretization_steps=40,
)
experience = setup.algorithm.collect(context=setup.context, dynamics=setup.dynamics,
                                     n=64, time_grid=setup.time_grid, conditioning={})
setup.algorithm.update(context=setup.context, experience=experience)
```

It exists because four of the wiring decisions are ones that **fail silently when made by hand**, and all four are derivable from `algorithm.requirements`: the SDE profile (memoryless for AM, stochastic for Flow-GRPO, ODE otherwise), an interior time grid whenever the dynamics are stochastic, a reference policy only where one is needed, and a differentiable terminal cost only where one exists. Every bug this codebase has had in that area was one of those four.

`algorithm` and `requirements` are **mutually exclusive, and exactly one is required**. Pass `requirements=FineTuningRequirements(...)` instead of an id when the caller owns its own training step: the four decisions above are still made and still have to agree with that step, but `setup.algorithm` is `None` and nothing is validated against it — there is no algorithm object to validate. This is what `reward_actflow`'s two loops use. Registering a class whose only real content is a `requirements` property, purely to reach this function, would be a registry entry that names nothing.

Two provider protocols in `registry.py` are the seam a new data type plugs into — implement these and nothing else changes:

| protocol | supplies | changes for a graph modality |
|---|---|---|
| `ModalityProvider` | `geometry`, `schedule`, `base_sampler`, `codec`, `model` | graph geometry + a base sampler that also draws structure |
| `RewardProvider` | `reward`, `terminal_cost` (may be `None`) | nothing structural |

`ModalityProvider.model()` is called once per policy and **must return independent instances with identical weights** — `toy/providers.py` resolves its weights once and caches them, because pretraining per call would leave the train/rollout/reference policies silently different. Registration is lazy (`_register_builtins`) to keep `toy/` and any optional modality dependency off the base import path. `base_model_registry` / `reward_registry` remain only for the legacy `toy/gmm.py` and `toy/rewards.py`.

## Non-tensor states

`core/` and `trainers/` are already state-type agnostic: the only `.data` accesses in `core/` live in `TensorGeometry`, which is *supposed* to be the dense implementation. `tests/test_graph_state.py` proves it — it defines `SegmentBatch`, a ragged state with variable node counts, a batch-offset vector and an integer structural field (the layout of a PyG `Batch`), and runs the forward process, both samplers, the Gaussian kernel and all four trainers on it. **No change to `core/` or `trainers/` was needed to make that pass.** Use it as the template for a real PyG subclass.

The three things a graph state must get right, all of which that file demonstrates:

- **`scale(coefficient)`** — index 0 of a state tensor is a *node*, not a batch element, so a per-sample coefficient of shape `(num_graphs,)` must be expanded by node counts (`repeat_interleave`), not reshaped.
- **`LatentGeometry.squared_norm` / `active_dimensions`** — segment reductions producing one value per graph. Per-graph `active_dimensions` is what lets `GaussianMarkovKernel.log_prob` handle graphs of different sizes as Gaussians of different dimension; it is already written for that.
- **`assert_compatible`** — the one failure mode a graph state has that a dense one does not: two batches can share a total node count while describing different graphs, and an elementwise add would then be silently wrong. `concat` on a real PyG state must also re-offset `edge_index`.

## Toy environment (`src/diffusiongym/toy/`)

Used by `tests/` and `examples/finetune_2d_gmm.py` to exercise the whole pipeline without GPU/large weights:

- `gmm.py` — 1D GMM base model (`OneDimensionalBaseModel`, `MLP`) and `rewards.py` (`BinaryReward`, `GaussianReward`) — these use the **legacy** `BaseModel`/`Reward` interfaces from `base_models.py`/`rewards/base.py`.
- `analytic1d.py` — a 1-D Gaussian-mixture flow with **no trained model at all**: the exact rectified-flow velocity is available in closed form (`ExactVelocityModel`), so a marginal-preserving SDE rollout has the mixture as its exact terminal law, and `Mixture1D.tilt(alpha)` gives the exponentially tilted law exactly. That is what makes it a ground truth for *samplers* rather than for fine-tuning algorithms — `gmm2d.py` entangles sampler error with model error, this does not. Used by `tests/test_smc_correctness.py` and `examples/check_smc.py`. Reach for it whenever the question is "does this sampler target the right distribution", and add exact tilts here rather than inventing new scaffolding.
- `gmm2d.py` — 2D four-Gaussian-mixture problem built directly on the **`core/`** stack (`GMMFlowModel`, `GMMBaseSampler`, `VelocityMLP`, `exact_velocity` for verification, five reward families with matching differentiable costs). This is the reference example for how to wire a new modality into `core/`; see `specs_test_example.md`.

**Rewards.** `LinearReward` (monotone, aligned with the mode structure — so easy that any roughly-correct gradient direction passes it; use it as a smoke test, not as evidence), `QuadraticReward`, `BoxReward` (indicator, non-differentiable, so Adjoint Matching cannot run on it), plus two harder ones: `BimodalReward` (two bumps of unequal height on opposite modes — non-monotone, so a method must reproduce a *ratio* rather than a direction) and `RingReward` (a target radius, which asks every mode to move inward — within-mode geometry rather than mode reweighting, which reweighting the existing samples cannot achieve).

**Exact densities.** Two dimensions make the instantaneous change of variables cheap — the divergence is an exact 2x2 trace — so `log_density()` returns exact `log p_theta(x)` rather than an estimate, and everything else follows from it: `make_density_grid`, `analytic_frontier` (the best achievable `(E[r], KL)` trade-off), and `tilt_diagnostics`. Validated by `check_density_normalization` and `tests/test_gmm2d_density.py`; note `make_density_grid` renormalizes the reference *on the grid*, without which `KL` at `lambda = 0` comes out at `-log Z` instead of 0 and biases every frontier point.

### Two entry points

- `examples/finetune_2d_gmm.py` — qualitative sample figures per (algorithm, reward). Calibrates the effective lambda via `estimate_reward_std()`.
- `examples/analyze_finetuning.py` — quantitative behaviour analysis, three modes: `frontier` (multi-seed `E[r]`-vs-`KL` against the analytic optimum), `maps` (where the probability mass actually moved), `sweep` (effect of one hyperparameter). Caches the pretrained model in `examples/figures/pretrained.pt`.

**Judge runs on `(E[r], KL)`, never `E[r]` alone** — a collapsed policy maximizes reward outright (measured: unregularized ORW-CFM reached `E[r]` 11.9 against an analytic optimum of 2.05, with 99.8% of its mass on one mode). Two diagnostics from `tilt_diagnostics` carry most of the signal: for any KL-regularized optimum `log p_theta - log p_ref = lambda r - log Z` exactly, so regressing the measured log-ratio on the reward recovers `achieved_lambda` (the tilt the method really applied, versus the one requested) and `tilt_r2` (**how much of the density change the reward explains at all — this is the precise statement of "fine-tuning only moved mass where it was needed"**). When plotting a log-ratio field, mask to the reference support: outside it both densities are ~0 and their ratio is numerical noise that renders as large spurious blobs.

## Testing conventions

`tests/conftest.py` builds a full toy `FlowEnvironment` fixture graph (schedule → geometry → base_sampler → forward_process → env, plus ODE/SDE samplers and `PolicyBundle`s with/without a reference policy) around a tiny 1D `_TinyFlowModel`/`_TinyMLP` with zero-initialized output layers — reuse these fixtures for new core-level tests rather than building parallel scaffolding. `test_core.py` tests individual `core/` primitives in isolation; `test_trainers_smoke.py` runs one iteration of each algorithm end-to-end plus targeted requirement-validation checks (e.g. `test_requires_memoryless_dynamics`, `test_requires_reference_policy`).

**Assert the distribution, not the direction.** `test_smc.py` and `test_smc_correctness.py` are the worked contrast: the first checks shapes, validation and "a positive potential moves the mean positive", which every one of the three SMC defects above passes; the second checks the sampled law against a closed form and catches them. When a component's job is to produce a distribution, find or build the case where that distribution is analytic (`toy/analytic1d.py`) rather than settling for a monotonicity assertion.

**Known-wrong behaviour gets an `xfail(strict=True)` test asserting the correct outcome**, with the mechanism in the `reason` string — not a test pinned to the buggy output, and not a comment. Strict mode turns the eventual fix into an unexpected-pass that says "delete this marker", so the defect can neither be forgotten nor silently re-introduced.
