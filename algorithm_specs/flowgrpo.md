Below is implementation-oriented pseudocode for **Flow-GRPO with PPO clipping and per-transition KL regularization**. It follows the Flow-GRPO formulation: convert the flow ODE into a marginal-preserving SDE, collect groups of stochastic denoising trajectories, compute group-relative terminal advantages, and optimize the likelihood ratios of every transition. 

### Objective

For each condition (c), sample a group of (G) trajectories using the frozen rollout policy (\pi_{\theta_{\mathrm{old}}}). For sample (i),

[
\widehat A_i
============

\frac{
R(x_0^i,c)-\operatorname{mean}*{j=1}^G R(x_0^j,c)
}{
\operatorname{std}*{j=1}^G R(x_0^j,c)+\epsilon_A
}.
]

Because reward is terminal-only, the same (\widehat A_i) is assigned to every transition in trajectory (i).

The maximized objective is

[
J(\theta)
=========

\frac{1}{G T}
\sum_{i=1}^{G}\sum_{k=0}^{T-1}
\left[
\min\left(
\rho_{i,k}(\theta)\widehat A_i,,
\operatorname{clip}(\rho_{i,k}(\theta),1-\epsilon,1+\epsilon)
\widehat A_i
\right)
-------

\beta_{\mathrm{KL}},
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x_k^i,c)
\Vert
\pi_{\mathrm{ref}}(\cdot\mid x_k^i,c)
\right)
\right],
]

with

[
\rho_{i,k}(\theta)
==================

\frac{
\pi_\theta(x_{k+1}^i\mid x_k^i,c)
}{
\pi_{\theta_{\mathrm{old}}}(x_{k+1}^i\mid x_k^i,c)
}.
]

This is the clipped Flow-GRPO objective given in the paper, with a KL penalty against a fixed reference policy. 

---

## Pseudocode

```text
Algorithm: Flow-GRPO Fine-Tuning with KL Regularization

Require:
    flow velocity model v_theta(x, t, c)
    frozen reference model v_ref(x, t, c)
    reward function R(x_final, c)
    condition / prompt dataset C

    number of outer iterations N
    prompts per rollout batch B
    trajectories per prompt G
    reduced rollout steps T
    policy optimization epochs E
    minibatch size M

    PPO clipping parameter eps_clip
    KL coefficient beta_KL
    optimizer learning rate eta
    advantage stabilizer eps_A

    SDE noise scale a
    time grid t_0 > t_1 > ... > t_T
        # sampling runs from noise toward data

Initialize:
    theta from pretrained flow model
    theta_ref <- copy(theta)
    freeze(theta_ref)

for iteration = 1, ..., N:

    # -----------------------------------------------------------
    # 1. Freeze the behavior policy used for this rollout batch
    # -----------------------------------------------------------
    theta_old <- copy(theta)
    freeze(theta_old)

    rollout_buffer <- empty

    # -----------------------------------------------------------
    # 2. Collect grouped stochastic trajectories
    # -----------------------------------------------------------
    sample conditions c_1, ..., c_B from C

    for each condition c_b:

        group_trajectories <- empty
        group_rewards <- empty

        for i = 1, ..., G:

            x <- sample_base_noise()
            trajectory_i <- empty

            for k = 0, ..., T - 1:

                t <- t_k
                dt <- t_{k+1} - t_k
                h <- abs(dt)

                # Rectified-flow ODE-to-SDE conversion.
                sigma_t <- a * sqrt(t / (1 - t))

                v_old <- v_theta_old(x, t, c_b)

                drift_old <- v_old
                             + sigma_t^2 / (2 t)
                               * (x + (1 - t) * v_old)

                # Mean of the Euler-Maruyama transition.
                mu_old <- x + drift_old * dt

                transition_std <- sigma_t * sqrt(h)

                z <- Normal(0, I)
                x_next <- mu_old + transition_std * z

                # Store enough information to evaluate the exact
                # transition likelihood under theta_old later.
                logp_old <- GaussianLogProbability(
                    value = x_next,
                    mean = mu_old,
                    std = transition_std
                )

                trajectory_i.append(
                    condition       = c_b,
                    x_current       = x,
                    x_next          = x_next,
                    time            = t,
                    dt              = dt,
                    sigma           = sigma_t,
                    logp_old        = stop_gradient(logp_old)
                )

                x <- x_next

            x_final <- postprocess(x)
            reward_i <- R(x_final, c_b)

            group_trajectories.append(trajectory_i)
            group_rewards.append(reward_i)

        # -------------------------------------------------------
        # 3. Compute group-relative advantages
        # -------------------------------------------------------
        reward_mean <- mean(group_rewards)
        reward_std  <- std(group_rewards)

        for i = 1, ..., G:

            advantage_i <- (
                group_rewards[i] - reward_mean
            ) / (reward_std + eps_A)

            # Terminal reward is broadcast to all denoising steps.
            for transition in group_trajectories[i]:
                transition.advantage <- stop_gradient(advantage_i)
                rollout_buffer.add(transition)

    # -----------------------------------------------------------
    # 4. Optimize the current policy on the collected trajectories
    # -----------------------------------------------------------
    for epoch = 1, ..., E:

        shuffle(rollout_buffer)

        for minibatch in minibatches(rollout_buffer, size=M):

            surrogate_sum <- 0
            kl_sum <- 0

            for transition in minibatch:

                c      <- transition.condition
                x      <- transition.x_current
                x_next <- transition.x_next
                t      <- transition.time
                dt     <- transition.dt
                h      <- abs(dt)
                sigma  <- transition.sigma
                A      <- transition.advantage

                # -----------------------------------------------
                # Current transition distribution pi_theta
                # -----------------------------------------------
                v_new <- v_theta(x, t, c)

                drift_new <- v_new
                             + sigma^2 / (2 t)
                               * (x + (1 - t) * v_new)

                mu_new <- x + drift_new * dt
                transition_std <- sigma * sqrt(h)

                logp_new <- GaussianLogProbability(
                    value = x_next,
                    mean = mu_new,
                    std = transition_std
                )

                ratio <- exp(logp_new - transition.logp_old)

                unclipped <- ratio * A
                clipped <- clip(
                    ratio,
                    1 - eps_clip,
                    1 + eps_clip
                ) * A

                surrogate_sum += min(unclipped, clipped)

                # -----------------------------------------------
                # Reference-policy KL regularization
                # -----------------------------------------------
                with no_gradient:
                    v_reference <- v_ref(x, t, c)

                    drift_reference <- v_reference
                                       + sigma^2 / (2 t)
                                         * (
                                             x
                                             + (1 - t)
                                               * v_reference
                                         )

                    mu_reference <- x + drift_reference * dt

                # Both transition kernels have the same covariance.
                kl_transition <- squared_norm(
                    mu_new - mu_reference
                ) / (2 * sigma^2 * h)

                kl_sum += kl_transition

            J_clip <- surrogate_sum / size(minibatch)
            J_kl   <- kl_sum / size(minibatch)

            # Maximize J_clip - beta_KL * J_kl.
            # Equivalently, minimize the negative objective.
            loss <- -J_clip + beta_KL * J_kl

            optimizer.zero_grad()
            backpropagate(loss)
            clip_gradient_norm(theta)
            optimizer.step()

    # Discard rollout buffer.
    # A new theta_old snapshot is taken at the next iteration.

return v_theta
```

## Rectified-flow transition kernel

For rectified flow, the paper constructs the SDE

[
dx_t =
\left[
v_\theta(x_t,t,c)
+
\frac{\sigma_t^2}{2t}
\left(
x_t+(1-t)v_\theta(x_t,t,c)
\right)
\right]dt
+
\sigma_t,dW_t,
]

typically with

[
\sigma_t = a\sqrt{\frac{t}{1-t}}.
]

Euler–Maruyama therefore gives a Gaussian policy

[
\pi_\theta(x_{t+\Delta t}\mid x_t,c)
====================================

\mathcal N\left(
\mu_\theta(x_t,t,c),
\sigma_t^2|\Delta t|I
\right),
]

where

[
\mu_\theta
==========

x_t+
\left[
v_\theta+
\frac{\sigma_t^2}{2t}
\left(x_t+(1-t)v_\theta\right)
\right]\Delta t.
]

This tractable Gaussian transition is what makes both the PPO ratio and KL term computable. 

Because the current and reference transition policies share the same covariance,

[
D_{\mathrm{KL}}
\left(
\pi_\theta\Vert\pi_{\mathrm{ref}}
\right)
=======

\frac{
|\mu_\theta-\mu_{\mathrm{ref}}|_2^2
}{
2\sigma_t^2|\Delta t|
}.
]

For the convention used in the paper, this can also be written directly in velocity space as

[
D_{\mathrm{KL}}
\left(
\pi_\theta\Vert\pi_{\mathrm{ref}}
\right)
=======

\frac{|\Delta t|}{2}
\left(
\frac{\sigma_t(1-t)}{2t}
+
\frac{1}{\sigma_t}
\right)^2
\left|
v_\theta(x_t,t,c)
-----------------

v_{\mathrm{ref}}(x_t,t,c)
\right|_2^2.
]

The mean-difference implementation is generally less error-prone because it remains correct under changes in time orientation or discretization conventions. The paper also uses **denoising reduction** during training—for example, 10 rollout steps instead of the full 40 inference steps—to lower online data-collection cost. 
