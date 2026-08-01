You should build a deliberately narrow **affine-Gaussian flow-matching environment**, not a universal continuous generative-process framework.

That scope naturally covers:

* Toy flow-matching models
* SD3.5 flow matching
* FlowMol Gaussian
* ACTFLOW
* DiffusionNFT
* ORW-CFM
* Flow-GRPO
* Adjoint Matching, with its additional reward and dynamics requirements

The key is to generalize only where the cost is small and the semantics remain clear.

# Recommended V1 contract

## 1. Euclidean model state, with masks and linear constraints

Assume that after preprocessing, the model operates on a finite-dimensional continuous state

[
x\in\mathbb R^d.
]

The actual implementation may be structured:

```python
State = {
    "coordinates": Tensor,
    "atom_types": Tensor,
    "bonds": Tensor,
    ...
}
```

but every component must support:

* Addition and scalar multiplication
* Gaussian noise sampling
* A masked/projected squared norm
* Batched model evaluation

Allow **known linear constraints**, because FlowMol coordinates may need centering or projection onto the zero-center-of-mass subspace. Do not attempt to support general manifolds.

The useful assumption is therefore:

> The model state is Euclidean, possibly restricted to a known linear subspace and equipped with masks.

This covers image latents, toy vectors and Gaussian molecular representations.

---

## 2. Gaussian base distribution

Assume

[
z\sim\mathcal N(0,I)
]

in the model state space, with optional masking or projection:

[
z=P\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
]

Do not generalize to arbitrary priors in V1.

For your models:

* Toy flows: standard Gaussian
* SD3.5 latent flow: Gaussian latent noise
* FlowMol Gaussian: Gaussian state variables, potentially with centering and masking

The base distribution should be an object because the exact tensor structure is modality-specific:

```python
class GaussianBase:
    def sample_like(
        self,
        template: State,
        generator: torch.Generator | None = None,
    ) -> State: ...

    def project(self, x: State) -> State: ...
```

But only implement Gaussian bases.

### Cost of generalizing the base distribution

| Generalization            |       Cost | Recommendation  |
| ------------------------- | ---------: | --------------- |
| Isotropic Gaussian        |        Low | Support         |
| Masked/projected Gaussian |        Low | Support         |
| Fixed diagonal covariance | Low–medium | Possibly later  |
| Full covariance           |     Medium | Avoid initially |
| Non-Gaussian base         |       High | Exclude         |
| Learned base distribution |  Very high | Exclude         |

A non-Gaussian base breaks several convenient score identities and complicates the ODE-to-SDE conversion needed by Flow-GRPO.

---

## 3. Affine conditional interpolants only

Use

[
x_t=a(t)z+b(t)x_1,
]

where:

[
a(0)=1,\quad b(0)=0,
\qquad
a(1)=0,\quad b(1)=1.
]

Choose the canonical time convention:

* (t=0): Gaussian base
* (t=1): data

The conditional target velocity is then

[
u_t^\star
=========

\dot a(t)z+\dot b(t)x_1.
]

This is sufficient for all your continuous models. DiffusionNFT also works with a general affine Gaussian path and defines the target velocity through the derivatives of the interpolation schedule. Rectified flow is its linear special case. 

### Default schedule

Use rectified flow:

[
a(t)=1-t,
\qquad
b(t)=t,
]

so that

[
x_t=(1-t)z+tx_1,
\qquad
u_t^\star=x_1-z.
]

This should be the default for:

* Toy experiments
* Newly trained models
* Any controlled benchmark where you choose the pretraining setup

### Why still support general affine schedules?

Supporting arbitrary differentiable scalar (a(t),b(t)) is cheap. You only need:

```python
class AffineSchedule:
    def a(self, t): ...
    def b(self, t): ...
    def da_dt(self, t): ...
    def db_dt(self, t): ...
```

This lets a pretrained model preserve its original schedule without adding meaningful architectural complexity.

### Cost of interpolant generalization

| Interpolant                              |        Cost | Recommendation     |
| ---------------------------------------- | ----------: | ------------------ |
| Linear rectified flow                    |      Lowest | Default            |
| General affine scalar schedule           |         Low | Support            |
| Affine matrix-valued schedule            | Medium–high | Exclude            |
| Nonlinear state-dependent interpolant    |        High | Exclude            |
| Schrödinger bridge / learned interpolant |   Very high | Exclude            |
| Manifold interpolation                   |   Very high | Separate framework |

The important boundary is:

> Support scalar affine schedules, not arbitrary stochastic interpolants.

---

## 4. Independent coupling only

Initially sample

[
z\sim p_0,\qquad x_1\sim p_{\mathrm{data}}
]

independently.

Do not initially support:

* Minibatch optimal transport coupling
* Learned couplings
* Data-dependent base distributions
* Multi-sample conditional paths

These may improve ordinary flow pretraining, but they are not needed for reward fine-tuning of existing models.

You can leave one narrow extension point:

```python
class PairSampler:
    def pair(self, x_data, base_distribution) -> tuple[State, State]:
        ...
```

Implement only `IndependentPairSampler` in V1.

That avoids hardcoding the coupling without committing to maintaining several coupling algorithms.

---

# Canonical model output

Use **path velocity** as the canonical training field:

[
v_\theta(x_t,t)\approx
\mathbb E[u_t^\star\mid x_t].
]

This directly matches the flow-matching objective:

[
\mathcal L_{\mathrm{FM}}
========================

\mathbb E\left[
w(t)
\left|
v_\theta(x_t,t)-u_t^\star
\right|^2
\right].
]

ACTFLOW updates using standard flow-matching losses on accepted and optionally rejected replay samples.  DiffusionNFT similarly constructs its positive and negative policies in velocity space. 

## Native parameterizations

For the selected models, support only:

1. `VELOCITY`
2. `ENDPOINT`

For an endpoint predictor (\hat x_{1,\theta}),

[
\hat v_\theta
=============

\dot a(t)z_\theta+\dot b(t)\hat x_{1,\theta},
]

where

[
z_\theta
========

\frac{x_t-b(t)\hat x_{1,\theta}}{a(t)}.
]

Equivalently,

[
\hat v_\theta
=============

\frac{\dot a(t)}{a(t)}x_t
+
\left(
\dot b(t)-\frac{\dot a(t)b(t)}{a(t)}
\right)\hat x_{1,\theta}.
]

Endpoint prediction is useful because some molecular models may internally predict clean endpoints rather than directly emitting velocity.

Do not implement noise and score prediction unless an actual selected model needs them. Under an affine Gaussian path, adding them later is algebraically straightforward.

---

# Separate three meanings of “schedule”

This distinction should be explicit in the code.

## 1. Interpolation schedule

Defines the training path:

[
a(t),\quad b(t).
]

This determines (x_t) and the flow-matching target.

## 2. Sampling stochasticity schedule

Defines the SDE diffusion coefficient:

[
g(t).
]

This determines how much stochasticity is injected during sampling.

## 3. Numerical time grid

Defines the actual discrete integration points:

[
0=t_0<t_1<\cdots<t_N=1.
]

These should be three different objects.

Do not call all three a `Scheduler`.

For example:

```python
process.schedule          # a(t), b(t), derivatives
dynamics.diffusion        # g(t)
integrator.time_grid      # numerical discretization
```

This will prevent many subtle bugs.

---

# Do not globally assume memoryless dynamics

The memoryless schedule is not a requirement of flow matching itself. It is a requirement of the specific stochastic-optimal-control formulation used by Adjoint Matching.

Adjoint Matching requires the memoryless SDE during fine-tuning to avoid the initial-value bias, but the paper explicitly notes that sampling after fine-tuning can use any schedule, including a deterministic ODE. 

Therefore, model memorylessness as a special dynamics profile:

```python
class DynamicsProfile:
    ...

class DeterministicFlowODE(DynamicsProfile):
    ...

class MarginalPreservingFlowSDE(DynamicsProfile):
    diffusion_schedule: ScalarDiffusionSchedule

class MemorylessFlowSDE(MarginalPreservingFlowSDE):
    ...
```

Then:

* DiffusionNFT: any rollout profile
* ORW-CFM: usually ODE sampling
* ACTFLOW: whatever sampler the model normally uses
* Flow-GRPO: stochastic marginal-preserving SDE
* Adjoint Matching: memoryless marginal-preserving SDE during training

This is much cleaner than requiring every flow to be memoryless.

---

# Sampling: implement exactly two integrators initially

## ODE Euler

For

[
dX_t=v_\theta(X_t,t),dt,
]

use

[
X_{k+1}
=======

X_k+\Delta t_k,v_\theta(X_k,t_k).
]

This supports:

* Standard model evaluation
* ACTFLOW data generation
* DiffusionNFT rollouts
* ORW-CFM rollouts
* SD3.5 inference
* Toy models
* FlowMol sampling

DiffusionNFT explicitly decouples sampling from training and permits arbitrary black-box solvers, requiring only terminal clean samples and rewards. 

## Euler–Maruyama

For a marginal-preserving SDE

[
dX_t
====

b_\theta(X_t,t),dt+g(t),dW_t,
]

use

[
X_{k+1}
=======

X_k+\Delta t_k b_\theta(X_k,t_k)
+
\sqrt{\Delta t_k},g(t_k)\epsilon_k.
]

This supports:

* Flow-GRPO stochastic exploration
* Exact Gaussian one-step transition likelihoods
* Adjoint Matching memoryless trajectories
* Stochastic ablations

Flow-GRPO's central construction is precisely an ODE-to-SDE conversion preserving the model's time marginals, followed by first-order stochastic sampling. 

## Do not initially implement

* Adaptive ODE solvers
* High-order SDE solvers
* Predictor-corrector samplers
* Arbitrary `torchdiffeq` integration
* Likelihood evaluation under multi-stage solvers
* Solver-specific adjoints

You can add Heun later as an ODE quality improvement, but it should not be part of the core interface initially.

The minimal solver enum can be:

```python
class IntegratorKind(Enum):
    EULER_ODE = "euler_ode"
    EULER_MARUYAMA = "euler_maruyama"
```

---

# One SDE family, not arbitrary SDE dynamics

For a learned flow velocity, support only **state-independent scalar diffusion**:

[
g(x,t)=g(t).
]

The SDE drift should be the unique marginal-preserving correction associated with the selected affine flow and (g(t)).

This gives a Gaussian Euler transition:

[
X_{k+1}\mid X_k
\sim
\mathcal N\left(
X_k+\Delta t_k b_\theta(X_k,t_k),
g(t_k)^2\Delta t_k I
\right).
]

That is exactly what Flow-GRPO needs for policy ratios and KL penalties.

Do not support:

* State-dependent diffusion (g(x,t))
* Matrix-valued diffusion
* Learned variance
* Degenerate diffusion except a known fixed projection
* Arbitrary user-provided drift functions unrelated to the forward path

These substantially complicate:

* Transition likelihoods
* Relative controls
* Girsanov costs
* Memoryless validation
* Adjoint equations
* Structured-state handling

---

# State geometry is a required abstraction

Images and molecules should not be forced to use identical raw MSE reduction.

Assume a block-additive quadratic geometry:

[
|x|_{\mathcal X}^2
==================

\sum_j \lambda_j
\left|
M_j x^{(j)}
\right|_2^2,
]

where:

* (j) indexes state components
* (M_j) applies masks or projections
* (\lambda_j) balances modalities or feature groups

For SD3.5, this may simply be latent-space MSE.

For FlowMol, different weights may be needed for:

* Coordinates
* Atom-type features
* Charges
* Bonds

Keep two reductions distinct:

```python
geometry.training_error(...)   # often mean per active dimension
geometry.gaussian_quadratic(...)  # sum required for log probability
```

A mean MSE and a Gaussian log density are not interchangeable.

---

# Recommended framework boundary

I would name the main object something explicit:

```python
class AffineGaussianFlowEnvironment:
    process: AffineGaussianFlowProcess
    base_distribution: GaussianBase
    geometry: EuclideanStateGeometry
    codec: DataCodec
    reward: RewardEvaluator
```

Its minimal mathematical API should be:

```python
batch = env.make_forward_batch(
    x_data,
    t=t,
    generator=generator,
)

velocity = env.predict_velocity(
    model,
    batch.x_t,
    batch.t,
    conditioning=batch.conditioning,
)

target = env.target_velocity(batch)

error = env.velocity_error(
    velocity,
    target,
)
```

Sampling should be separate:

```python
rollout = env.rollout(
    model=model,
    dynamics=dynamics_profile,
    integrator=IntegratorKind.EULER_ODE,
    time_grid=time_grid,
    storage=storage_spec,
)
```

Transition kernels should only exist for Euler–Maruyama:

```python
kernel = env.euler_transition_kernel(
    model=model,
    x=x_t,
    t=t,
    dt=dt,
    dynamics=dynamics_profile,
)

x_next = kernel.rsample()
log_prob = kernel.log_prob(x_next)
```

The same kernel must produce the rollout sample and its likelihood.

---

# What the environment should explicitly reject

Fail early when given:

* A non-Gaussian base
* A non-affine interpolant
* A state-dependent diffusion
* A matrix diffusion without an implemented geometry
* A discrete state
* A manifold-valued state
* A solver whose transition likelihood is requested but unavailable
* Adjoint Matching without memoryless dynamics
* Flow-GRPO without a Gaussian Euler kernel
* An endpoint conversion at a singular schedule endpoint
* A model whose state constraints cannot be represented by masks or linear projections

Explicit exclusion is better than partially supporting these cases incorrectly.

---

# Generalization budget

A practical division is:

| Feature                          | Include now? |     Engineering burden |
| -------------------------------- | -----------: | ---------------------: |
| Rectified linear path            |          Yes |               Very low |
| General scalar affine path       |          Yes |                    Low |
| Gaussian base                    |          Yes |               Very low |
| Masked/projected Gaussian        |          Yes |                    Low |
| Velocity prediction              |          Yes |               Very low |
| Endpoint prediction              |          Yes |                    Low |
| Score/noise prediction           |        Later | Low, but extra testing |
| ODE Euler                        |          Yes |                    Low |
| Euler–Maruyama                   |          Yes |                    Low |
| Memoryless SDE profile           |          Yes |             Low–medium |
| General scalar SDE stochasticity |          Yes |                    Low |
| Heun ODE                         |        Later |                    Low |
| Arbitrary ODE solver             |           No |                 Medium |
| Adaptive integration             |           No |            Medium–high |
| Non-affine interpolants          |           No |                   High |
| Non-Gaussian bases               |           No |                   High |
| State-dependent diffusion        |           No |                   High |
| Manifold flows                   |           No |              Very high |
| Discrete diffusion               |     Separate |              Very high |

# Final recommendation

Your controlled environment should make the following assumptions:

[
\boxed{
\begin{aligned}
&x_t=a(t)z+b(t)x_1,\
&z\sim\mathcal N(0,I)
\text{ on a masked/projected Euclidean state},\
&v^\star_t=\dot a(t)z+\dot b(t)x_1,\
&\text{canonical model field}=v_\theta,\
&\text{default }a(t)=1-t,;b(t)=t,\
&\text{ODE solver}=\text{Euler},\
&\text{SDE solver}=\text{Euler--Maruyama},\
&g(t)\text{ is scalar and state-independent},\
&\text{memoryless }g(t)\text{ is an algorithm-specific profile.}
\end{aligned}
}
]

That is narrow enough to remain maintainable, but broad enough to test all your continuous baselines and ACTFLOW across toy data, images and Gaussian molecular flows. The largest mistake would be generalizing the mathematical process beyond affine Gaussian flows before an actual experiment requires it.
