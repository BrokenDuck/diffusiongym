Below is implementation-oriented pseudocode for **DiffusionNFT**. It follows the paper’s forward-process, negative-aware objective: collect clean samples from an old policy, convert group-relative rewards into optimality probabilities, and train one model through implicit positive and negative policies. 

## Objective

For samples (x_0\sim \pi_{\text{old}}(\cdot\mid c)), define a normalized optimality probability (r(x_0,c)\in[0,1]). The loss is

[
\mathcal L_{\mathrm{NFT}}(\theta)
=================================

\mathbb E\left[
r,\left|v_\theta^+(x_t,c,t)-v\right|*2^2
+
(1-r),\left|v*\theta^-(x_t,c,t)-v\right|_2^2
\right],
]

with implicit policies

[
v_\theta^+
==========

(1-\beta)v_{\text{old}}+\beta v_\theta,
]

[
v_\theta^-
==========

(1+\beta)v_{\text{old}}-\beta v_\theta.
]

Here (v) is the ordinary forward-process velocity target. 

---

## Pseudocode

```text
Algorithm: Diffusion Negative-Aware Fine-Tuning

Require:
    pretrained velocity model v_ref(x_t, t, c)
    trainable velocity model v_theta(x_t, t, c)
    reward function R_raw(x_0, c)
    condition / prompt dataset C

    outer iterations N
    conditions per rollout batch B
    samples per condition K
    gradient epochs E
    minibatch size M

    implicit-policy coefficient beta > 0
    optimizer learning rate eta
    reward normalization scale Z > 0
    sampling-policy EMA coefficient gamma in [0, 1)

    forward schedule alpha(t), sigma(t)
    derivatives alpha_dot(t), sigma_dot(t)
    timestep weighting w(t), optional

Initialize:
    theta <- parameters(v_ref)

    theta_old <- copy(theta)
    freeze theta_old during each update phase

    replay_buffer <- empty

for iteration = 1, ..., N:

    clear replay_buffer

    # ============================================================
    # 1. Collect clean samples from the old sampling policy
    # ============================================================

    sample conditions c_1, ..., c_B from C

    for each condition c:

        samples <- empty
        rewards <- empty

        for i = 1, ..., K:

            # Sampling may use any black-box solver:
            # ODE, SDE, Euler, Heun, high-order solver, etc.
            x0_i <- sample_clean_output(
                model = v_theta_old,
                condition = c
            )

            reward_i <- R_raw(x0_i, c)

            samples.append(x0_i)
            rewards.append(reward_i)

        # ========================================================
        # 2. Convert group-relative rewards to r in [0, 1]
        # ========================================================

        group_mean <- mean(rewards)

        for i = 1, ..., K:

            centered_reward <- rewards[i] - group_mean

            normalized_reward <- clip(
                centered_reward / Z,
                -1,
                1
            )

            optimality_probability <- (
                0.5 + 0.5 * normalized_reward
            )

            replay_buffer.add(
                condition = c,
                clean_sample = samples[i],
                optimality = stop_gradient(
                    optimality_probability
                )
            )

    # ============================================================
    # 3. Optimize through the forward diffusion / flow process
    # ============================================================

    for epoch = 1, ..., E:

        shuffle replay_buffer

        for minibatch in minibatches(replay_buffer, size=M):

            c  <- minibatch.condition
            x0 <- minibatch.clean_sample
            r  <- minibatch.optimality

            # Sample ordinary supervised-training noise/time.
            t <- sample_time()
            epsilon <- Normal(0, I)

            # Forward noising process.
            x_t <- alpha(t) * x0 + sigma(t) * epsilon

            # Standard velocity target.
            v_target <- (
                alpha_dot(t) * x0
                + sigma_dot(t) * epsilon
            )

            # Frozen sampling policy prediction.
            with no_gradient:
                v_old <- v_theta_old(x_t, t, c)

            # Current trainable prediction.
            v_new <- v_theta(x_t, t, c)

            # Implicit positive policy.
            v_positive <- (
                (1 - beta) * v_old
                + beta * v_new
            )

            # Implicit negative policy.
            v_negative <- (
                (1 + beta) * v_old
                - beta * v_new
            )

            positive_loss <- squared_norm(
                v_positive - v_target
            )

            negative_loss <- squared_norm(
                v_negative - v_target
            )

            sample_loss <- (
                r * positive_loss
                + (1 - r) * negative_loss
            )

            if timestep weighting is used:
                sample_loss <- w(t) * sample_loss

            loss <- mean(sample_loss)

            optimizer.zero_grad()
            backpropagate(loss)
            clip_gradient_norm(theta)
            optimizer.step()

    # ============================================================
    # 4. Soft-update the sampling policy
    # ============================================================

    theta_old <- (
        gamma * theta_old
        + (1 - gamma) * theta
    )

return v_theta
```

## Rectified-flow specialization

For rectified flow,

[
x_t=(1-t)x_0+t\epsilon,
]

so

[
\alpha_t=1-t,\qquad \sigma_t=t,
]

and the target velocity is

[
v=\epsilon-x_0.
]

The forward-process section then becomes:

```text
t <- Uniform(0, 1)
epsilon <- Normal(0, I)

x_t <- (1 - t) * x0 + t * epsilon
v_target <- epsilon - x0
```

No reverse trajectory is required for the optimization step. Only the clean generated sample (x_0), its condition, and its reward-derived optimality value need to be stored. 

## Practical reward transformation

The paper uses

[
r(x_0,c)
========

\frac12
+
\frac12
\operatorname{clip}
\left(
\frac{
R_{\mathrm{raw}}(x_0,c)
-----------------------

\frac1K\sum_{j=1}^K R_{\mathrm{raw}}(x_0^j,c)
}{
Z_c
},
-1,1
\right).
]

A practical implementation can use either:

```text
Z_c = global running reward standard deviation
```

or

```text
Z_c = standard deviation of the current reward group + epsilon
```

The source explicitly describes (Z_c) as a positive normalizing factor and gives a global reward standard deviation as an example. 

## Important implementation details

1. **Detach (v_{\text{old}}).**
   The old sampling policy is a fixed target during each optimization phase.

2. **Do not train two separate positive and negative models.**
   (v_\theta^+) and (v_\theta^-) are algebraic combinations of one trainable model and one frozen old model.

3. **Sampling and training are decoupled.**
   Sampling can use any solver, while training always uses the ordinary forward noising and velocity-regression process.

4. **The EMA update is part of the online algorithm.**
   A hard update corresponds to (\gamma=0). Large (\gamma) makes data collection more off-policy and more stable, but slower.

5. **The branch meanings are asymmetric.**
   Large (r) trains the implicit positive policy toward the sample’s flow target. Small (r) trains the implicit negative policy toward it, which algebraically pushes the learned policy away from the negative-data distribution.

6. **(\beta) controls the implicit guidance scale.**
   It is not a KL coefficient. Smaller (\beta) produces a stronger effective model displacement for a given positive/negative policy difference.
