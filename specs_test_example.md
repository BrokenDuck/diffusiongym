Use a **two-dimensional Gaussian-mixture flow with an analytically tractable reward tilt**. It is simple enough to solve almost exactly, but complex enough to reveal mode collapse, incorrect policy gradients, trajectory mistakes, and overly aggressive reward optimization.

# Recommended toy problem: tilted four-Gaussian mixture

## 1. Base distribution

Let the pretrained model represent

[
p_{\mathrm{base}}(x)
====================

\frac14\sum_{k=1}^4
\mathcal N(x;\mu_k,\sigma_{\mathrm{data}}^2 I_2),
]

with

[
\mu_1=(-2,-2),\quad
\mu_2=(-2,2),\quad
\mu_3=(2,-2),\quad
\mu_4=(2,2),
]

and, for example,

[
\sigma_{\mathrm{data}}=0.35.
]

Use the base distribution

[
z\sim p_0=\mathcal N(0,I_2).
]

This gives four clearly separated, visually identifiable modes.

The problem is preferable to a single Gaussian because a single Gaussian often lets an incorrect algorithm look correct: moving the mean in approximately the right direction may be enough to obtain a convincing plot. With four modes, you can test whether the algorithm changes:

* mode probabilities;
* within-mode positions;
* covariance;
* diversity;
* probability mass between modes.

## 2. Flow-matching pretraining

Use rectified flow:

[
x_t=(1-t)z+t x_1,
\qquad
z\sim\mathcal N(0,I),
\quad
x_1\sim p_{\mathrm{base}}.
]

The conditional velocity target is

[
u^\star(x_t,t\mid z,x_1)=x_1-z.
]

Train

[
\mathcal L_{\mathrm{FM}}(\theta)
================================

\mathbb E\left[
\left|
u_\theta(x_t,t)-(x_1-z)
\right|^2
\right].
]

This is the conventional affine flow-matching construction: a Gaussian base, an affine conditional interpolant, and velocity regression. The same rectified-flow interpolation and target (x_1-z) are used in the Flow-GRPO formulation. 

A small MLP is sufficient:

[
(x_1,x_2,t)\rightarrow
128\rightarrow128\rightarrow128\rightarrow 2,
]

with sinusoidal or Fourier time features.

---

# 3. Reward choices

Use three rewards in increasing order of difficulty.

## Reward A: linear reward

[
r_{\mathrm{lin}}(x)=c^\top x,
\qquad c=(1,0).
]

For inverse temperature (\lambda), the target distribution is

[
p_\lambda^\star(x)
==================

\frac{
p_{\mathrm{base}}(x)e^{\lambda c^\top x}
}{
Z_\lambda
}.
]

This target is analytically available. For a Gaussian component,

[
\mathcal N(x;\mu_k,\Sigma)
e^{\lambda c^\top x}
\propto
\exp\left(
\lambda c^\top\mu_k+
\frac{\lambda^2}{2}c^\top\Sigma c
\right)
\mathcal N(x;\mu_k+\lambda\Sigma c,\Sigma).
]

Therefore,

[
p_\lambda^\star(x)
==================

\sum_{k=1}^4 w_k^\star
\mathcal N(x;\mu_k+\lambda\sigma_{\mathrm{data}}^2c,
\sigma_{\mathrm{data}}^2I),
]

where

[
w_k^\star
=========

\frac{\exp(\lambda c^\top\mu_k)}
{\sum_j\exp(\lambda c^\top\mu_j)}.
]

This verifies two effects separately:

1. **Between-mode reweighting:** right-hand modes become more probable.
2. **Within-mode transport:** every component mean moves slightly right by
   [
   \lambda\sigma_{\mathrm{data}}^2c.
   ]

An implementation that only performs weighted replay may get the mode weights approximately right but fail to produce the correct within-mode shift.

## Reward B: quadratic target reward

Choose a target point (y=(3,1)):

[
r_{\mathrm{quad}}(x)
====================

-\frac{1}{2\tau_r^2}|x-y|^2.
]

Then

[
p_\lambda^\star(x)
\propto
p_{\mathrm{base}}(x)
\exp\left(
-\frac{\lambda}{2\tau_r^2}|x-y|^2
\right).
]

The product of each Gaussian component with the reward factor is another Gaussian. Let

[
\Sigma=\sigma_{\mathrm{data}}^2I,
\qquad
R=\frac{\tau_r^2}{\lambda}I.
]

For each component,

[
\Sigma_k^\star
==============

\left(\Sigma^{-1}+R^{-1}\right)^{-1},
]

[
\mu_k^\star
===========

\Sigma_k^\star
\left(
\Sigma^{-1}\mu_k+R^{-1}y
\right),
]

and its unnormalized weight is

[
\widetilde w_k^\star
====================

\frac14,
\mathcal N(y;\mu_k,\Sigma+R).
]

Normalize these weights across components.

This reward tests whether the algorithm correctly changes:

* mode weights;
* means;
* covariance.

## Reward C: non-differentiable region reward

[
r_{\mathrm{box}}(x)
===================

\mathbf 1
\left[
2.2<x_1<3.2,,
0.2<x_2<1.2
\right].
]

The tilted distribution is

[
p^\star_\lambda(x)
\propto
p_{\mathrm{base}}(x)
\begin{cases}
e^\lambda,&x\in A,\
1,&x\notin A.
\end{cases}
]

Its normalizer is exactly

[
Z_\lambda
=========

1+(e^\lambda-1)P_{\mathrm{base}}(A),
]

and

[
P^\star_\lambda(A)
==================

\frac{
e^\lambda P_{\mathrm{base}}(A)
}{
1+(e^\lambda-1)P_{\mathrm{base}}(A)
}.
]

Because the base distribution is a Gaussian mixture, (P_{\mathrm{base}}(A)) can be computed using products of one-dimensional Gaussian CDF differences.

This is useful for testing black-box methods such as Flow-GRPO and DiffusionNFT without relying on reward gradients. Flow-GRPO treats denoising as an MDP with terminal reward, while DiffusionNFT instead performs reward adaptation through forward-process flow-matching objectives.  

---

# 4. Exact density and exact flow target

For this toy problem, you can do more than compare final samples.

For the affine interpolation

[
x_t=(1-t)z+t x_1,
]

and a Gaussian-mixture terminal distribution

[
x_1\sim \sum_k w_k\mathcal N(\mu_k,\Sigma_k),
]

the time-(t) marginal is also a Gaussian mixture:

[
p_t(x)
======

\sum_k w_k
\mathcal N
\left(
x;
t\mu_k,
(1-t)^2I+t^2\Sigma_k
\right).
]

For component (k), define

[
C_{k,t}=(1-t)^2I+t^2\Sigma_k.
]

The posterior responsibility is

[
\gamma_k(x,t)
=============

\frac{
w_k\mathcal N(x;t\mu_k,C_{k,t})
}{
\sum_jw_j\mathcal N(x;t\mu_j,C_{j,t})
}.
]

The conditional expectation of the endpoint is

[
m_k(x,t)
========

# \mathbb E[x_1\mid x_t=x,k]

\mu_k+t\Sigma_k C_{k,t}^{-1}(x-t\mu_k).
]

Since

[
x_t=(1-t)z+t x_1
]

implies

[
x_1-z=\frac{x_1-x_t}{1-t},
]

the exact marginal flow-matching velocity is

[
u^\star(x,t)
============

\sum_k
\gamma_k(x,t)
\frac{m_k(x,t)-x}{1-t}.
]

This gives you an unusually strong correctness test:

[
\mathrm{VelocityMSE}(t)
=======================

\mathbb E_{x\sim p_t^\star}
\left[
|u_\theta(x,t)-u^\star(x,t)|^2
\right].
]

Thus, you can test both:

* whether the final distribution is correct;
* whether the learned vector field is correct at intermediate times.

---

# 5. Theoretical verification suite

## A. Moment checks

Compute analytically and empirically:

[
\mathbb E_{p^\star}[X],
\qquad
\operatorname{Cov}_{p^\star}(X).
]

Report

[
|\widehat\mu-\mu^\star|_2,
\qquad
|\widehat\Sigma-\Sigma^\star|_F.
]

These are simple smoke tests, but not sufficient by themselves.

## B. Mode-mass checks

Assign samples to the closest original mode:

[
\widehat k(x)
=============

\arg\min_k|x-\mu_k|.
]

Compare

[
\widehat w_k
============

\frac1N\sum_i
\mathbf 1[\widehat k(x_i)=k]
]

against the analytic (w_k^\star).

This catches mode collapse and incorrect relative weighting.

## C. Density-grid divergence

Because the problem is two-dimensional, evaluate both distributions on a dense grid.

Approximate

[
D_{\mathrm{KL}}(p^\star|p_\theta),
\qquad
D_{\mathrm{KL}}(p_\theta|p^\star),
]

or preferably Jensen-Shannon divergence:

[
D_{\mathrm{JS}}(p_\theta,p^\star).
]

A more stable sample-based alternative is sliced Wasserstein distance.

## D. Expected reward

Compare

[
\widehat J_\theta
=================

\frac1N\sum_i r(x_i)
]

to

[
J^\star=\mathbb E_{p^\star}[r(X)].
]

Do not use expected reward alone. A collapsed distribution can have higher reward than the prescribed tilted target.

## E. Regularized objective

For methods intended to solve

[
\max_q
\left{
\lambda\mathbb E_q[r(X)]
------------------------

D_{\mathrm{KL}}(q|p_{\mathrm{base}})
\right},
]

the optimizer is

[
q^\star(x)
\propto
p_{\mathrm{base}}(x)e^{\lambda r(x)}.
]

Estimate the objective for the learned model and compare it with the analytic target. Adjoint Matching explicitly targets this kind of reward-tilted distribution under its stochastic-control construction. 

## F. Intermediate-marginal checks

At times such as

[
t\in{0.1,0.25,0.5,0.75,0.9},
]

compare generated intermediate states against the analytic

[
p_t^\star
=========

\sum_k
w_k^\star
\mathcal N
\left(
t\mu_k^\star,
(1-t)^2I+t^2\Sigma_k^\star
\right).
]

This is particularly useful for identifying:

* reversed time conventions;
* incorrect drift scaling;
* SDE sign errors;
* wrong score-to-velocity conversion;
* endpoint versus noise parameterization mistakes.

## G. ODE/SDE marginal equivalence

For algorithms that convert the flow ODE into an SDE, generate samples with both samplers and compare their marginals at each time. Flow-GRPO relies on an ODE-to-SDE construction designed to preserve time marginals, so this is a direct implementation test. 

---

# 6. Visualization suite

Produce the same figures for the base model, analytic target, and each fine-tuned model.

## Figure 1: endpoint samples and density contours

Three columns:

1. base samples;
2. analytic tilted target;
3. fine-tuned samples.

Overlay density contours.

This immediately reveals:

* correct mode reweighting;
* collapse;
* off-support mass;
* incorrect mean shifts.

## Figure 2: learned vector field

At fixed times (t=0.25,0.5,0.75), plot:

* analytic velocity field (u^\star(x,t));
* learned velocity field (u_\theta(x,t));
* error magnitude
  [
  |u_\theta-u^\star|.
  ]

Use a grid of arrows with a heat map of the error norm.

## Figure 3: trajectories

Sample fixed initial noise points (z_i), and plot the trajectories

[
t\mapsto x_t^{(i)}.
]

Use the same initial noise seeds before and after fine-tuning.

This shows how the fine-tuning modifies transport rather than merely showing the endpoint result.

## Figure 4: mode probabilities over training

Plot

[
\widehat w_k^{(n)}
]

for each mode against optimization iteration, with horizontal lines for (w_k^\star).

This is particularly informative for online algorithms.

## Figure 5: reward versus divergence

For every checkpoint, plot

[
\left(
D_{\mathrm{JS}}(p_{\theta_n},p^\star),
\mathbb E_{p_{\theta_n}}[r]
\right).
]

This distinguishes genuine convergence from reward hacking.

## Figure 6: time-marginal animation

Animate samples or contours from (t=0) to (t=1) for:

* the analytic target flow;
* the learned flow.

This is probably the single most useful visual debugging tool.

---

# 7. Algorithm-specific correctness tests

## Adjoint Matching

Use the linear and quadratic rewards first.

Check that:

1. the learned terminal distribution approaches the analytic exponential tilt;
2. the control or velocity correction approaches the analytic target correction;
3. changing the numerical integration step does not qualitatively change the result;
4. the required fine-tuning noise schedule is implemented consistently.

The memoryless schedule is central to the theoretical claim that the controlled terminal distribution matches the prescribed tilt rather than a biased distribution depending on initial noise. 

A strong test is to intentionally replace the memoryless schedule with a naive constant-noise schedule. The correct and deliberately incorrect versions should diverge measurably.

## DiffusionNFT

Use Reward C as well as A and B.

Check:

1. positive and negative subsets have the expected analytic conditional distributions;
2. the update direction aligns with
   [
   u^+-u^-;
   ]
3. setting every reward to (1/2) produces approximately no update;
4. replacing (r) with (1-r) reverses the direction of improvement;
5. freezing the old policy versus updating it by EMA has the expected stability effect.

DiffusionNFT defines implicit positive and negative velocity policies and optimizes both branches through a forward-process regression loss. 

## Flow-GRPO

Use group sizes such as

[
G\in{4,16,64}.
]

Check:

1. the empirical SDE transition log-probability matches the Gaussian formula;
2. the likelihood ratio equals (1) when current and old parameters are identical;
3. normalized group advantages have approximately zero mean and unit variance;
4. the ODE and converted SDE agree in marginal distributions before fine-tuning;
5. results improve as group size increases and variance decreases;
6. setting all rewards equal produces zero policy-gradient signal.

The terminal reward is reused as the group-relative advantage across denoising steps in the basic Flow-GRPO construction. 

## Reward-weighted or contrastive flow matching

Check separately:

* mode reweighting accuracy;
* within-mode movement;
* covariance shrinkage.

This tells you whether the procedure is merely fitting rewarded samples or actually approximating the intended target distribution.

---

# 8. Suggested experiment sequence

Start with the following fixed configuration:

[
\sigma_{\mathrm{data}}=0.35,
\qquad
\lambda\in{0.25,0.5,1.0},
]

and (100{,}000) samples for final evaluation.

Run:

1. **Base pretraining test:** verify all four modes and the exact base velocity.
2. **Linear reward:** verify exact weights and mean shifts.
3. **Quadratic reward:** verify weights, means, and covariances.
4. **Box reward:** verify black-box and non-smooth reward handling.
5. **Strong tilt:** increase (\lambda) until algorithms become unstable or collapse.
6. **Ablations:** remove KL regularization, old-policy freezing, clipping, EMA, or the required SDE schedule one at a time.

The most important final table should contain:

| Metric                   | Base | Exact target | Algorithm |
| ------------------------ | ---: | -----------: | --------: |
| Mean reward              |      |              |           |
| JS divergence to target  |      |            0 |           |
| Mean error               |      |            0 |           |
| Covariance error         |      |            0 |           |
| Mode-weight error        |      |            0 |           |
| Velocity MSE             |      |            0 |           |
| Invalid/off-support mass |      |              |           |

The linear reward should be your unit test. The quadratic reward should be your integration test. The discontinuous box reward should be your realistic black-box stress test.
