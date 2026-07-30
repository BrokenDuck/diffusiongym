"""Fine-tune Stable Diffusion with Adjoint Matching."""

from pathlib import Path

import torch
from torchvision.utils import save_image

import diffusiongym
import diffusiongym.images
from diffusiongym.environments.base import EnvironmentMode

device = torch.device("cuda")

env = diffusiongym.make(
    base_model="images/sd1.5",
    reward="images/compression",
    discretization_steps=50,
    reward_scale=100,
    device=device,
    base_model_kwargs={"cfg_scale": 7.5},
    reward_kwargs={"quality_level": 65},
)

env.mode = EnvironmentMode.ADJOINT_MATCHING

# Optimizer for the base model (UNet)
opt = torch.optim.AdamW(env.base_model.parameters(), lr=1e-5)

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

num_iterations = 3
samples_per_iter = 2
train_steps_per_iter = 5

for iteration in range(num_iterations):
    print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

    # Sample trajectories (eval mode for inference)
    env.base_model.eval()
    sample = env.sample(samples_per_iter, prompt="A photo of a cat")

    print(f"  Rewards: {sample.rewards}")
    print(f"  Running costs (sum): {sample.running_costs.sum(dim=0)}")
    print(f"  Cost functionals (at t=0): {sample.cost_functionals[0]}")

    # Save images from this iteration
    for i in range(len(sample.sample)):
        path = output_dir / f"iter{iteration}_sample{i}.png"
        save_image(sample.sample.data[i], path)

    # Train the base model on the trajectories weighted by cost functionals
    # For adjoint matching, we train at each (x_t, t) pair weighted by the cost-to-go
    data = sample.trajectory[:-1]  # x_t at each step (exclude final)
    weights = [sample.cost_functionals[t] for t in range(sample.num_steps)]

    diffusiongym.train_base_model(
        env.base_model,
        opt,
        data=data,
        kwargs=[sample.kwargs] * sample.num_steps,
        weights=weights,
        steps=train_steps_per_iter,
        batch_size=samples_per_iter,
        pbar=True,
    )

# Final sample after training
print("\n--- Final sample ---")
env.base_model.eval()
sample = env.sample(2, prompt="A photo of a cat")
print(f"  Rewards: {sample.rewards}")

for i in range(len(sample.sample)):
    path = output_dir / f"final_sample{i}.png"
    save_image(sample.sample.data[i], path)
    print(f"  Saved: {path}")

