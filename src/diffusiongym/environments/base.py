"""Base environment classes and interfaces for diffusiongym."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto
from itertools import pairwise
from typing import Any, Protocol

import torch
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm, trange

from diffusiongym.base_models import BaseModel
from diffusiongym.rewards import Reward
from diffusiongym.schedulers import MemorylessNoiseSchedule, Scheduler
from diffusiongym.types import DDBatch
from diffusiongym.utils import dict_to_device, index_dict


class EnvironmentMode(StrEnum):
    """How the environment's drift and control cost are used."""

    BASE_INFERENCE = auto()
    POLICY_INFERENCE = auto()
    ADJOINT_MATCHING = auto()
    KL_REGULARIZED_RL = auto()


@dataclass
class Sample[D: DDBatch]:
    """A convenience wrapper for a batch of samples.

    Parameters
    ----------
    sample : D
        The final sample.
    latent : D
        The final latent.
    trajectory : list[D]
        Trajectory that lead to the final sample.
    timesteps : torch.Tensor
        Timestep grid used to sample.
    drifts : list[D]
        Drifts computed at each step.
    diffusions : list[D]
        Diffusions computed at each step.
    noises : list[D]
        Noises sampeld at each step.
    running_costs : torch.Tensor
        The running cost at each step for each trajectory.
    rewards : torch.Tensor
        Observed reward at the end of the sampling processes.
    valids : torch.Tensor
        Whether the samples are valid or not according to the reward class.
    cost_functionals : torch.Tensor
        Cost functionals, i.e., integral of running costs starting from t + reward.
    kwargs : dict
        Keyword arguments passed to the base model at each step.
    """

    sample: D
    latent: D
    trajectory: list[D]
    timesteps: torch.Tensor
    drifts: list[D]
    diffusions: list[D]
    noises: list[D]
    running_costs: torch.Tensor
    rewards: torch.Tensor
    valids: torch.Tensor
    cost_functionals: torch.Tensor
    kwargs: dict[str, Any]

    @property
    def num_steps(self) -> int:
        return self.timesteps.shape[0] - 1

    def __post_init__(self):
        n = len(self.sample)
        assert len(self.latent) == n, (
            f"latent batch size != sample batch size, got {len(self.latent)} != {n}"
        )
        assert len(self.trajectory[0]) == n, (
            f"trajectory batch size != sample batch size, got {len(self.trajectory[0])} != {n}"
        )
        assert len(self.drifts[0]) == n, (
            f"drift batch size != sample batch size, got {len(self.drifts[0])} != {n}"
        )
        assert len(self.diffusions[0]) == n, (
            f"diffusion batch size != sample batch size, got {len(self.diffusions[0])} != {n}"
        )
        assert len(self.noises[0]) == n, (
            f"noise batch size != sample batch size, got {len(self.noises[0])} != {n}"
        )
        assert self.running_costs.shape[1] == n, (
            f"running_costs batch size != sample batch size, got {self.running_costs.shape[0]} != {n}"
        )
        assert self.rewards.shape[0] == n, (
            f"rewards batch size != sample batch size, got {self.rewards.shape[0]} != {n}"
        )
        assert self.valids.shape[0] == n, (
            f"valids batch size != sample batch size, got {self.valids.shape[0]} != {n}"
        )
        assert self.cost_functionals.shape[1] == n, (
            f"cost functionals batch size != sample batch size, got {self.cost_functionals.shape[0]} != {n}"
        )

        m = self.num_steps
        assert len(self.trajectory) == m + 1, (
            f"trajectory length != number of steps + 1, got {len(self.trajectory)} != {m + 1}"
        )
        assert len(self.drifts) == m, (
            f"drifts length != number of steps, got {len(self.drifts)} != {m}"
        )
        assert len(self.diffusions) == m, (
            f"diffusions length != number of steps, got {len(self.diffusions)} != {m}"
        )
        assert len(self.noises) == m, (
            f"noises length != number of steps, got {len(self.noises)} != {m}"
        )
        assert self.running_costs.shape[0] == m, (
            f"running_costs length != number of steps, got {self.running_costs.shape[0]} != {m}"
        )
        assert self.cost_functionals.shape[0] == m + 1, (
            f"cost_functionals length != number of steps + 1, got {self.cost_functionals.shape[0]} != {m + 1}"
        )

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx: int) -> "Sample[D]":
        return Sample(
            sample=self.sample[idx],
            latent=self.latent[idx],
            trajectory=[state[idx] for state in self.trajectory],
            timesteps=self.timesteps,
            drifts=[drift[idx] for drift in self.drifts],
            diffusions=[diffusion[idx] for diffusion in self.diffusions],
            noises=[noise[idx] for noise in self.noises],
            running_costs=self.running_costs[:, idx : idx + 1],
            rewards=self.rewards[idx : idx + 1],
            valids=self.valids[idx : idx + 1],
            cost_functionals=self.cost_functionals[:, idx : idx + 1],
            kwargs=index_dict(self.kwargs, idx, idx + 1),
        )

    @staticmethod
    def concat(samples: list["Sample[D]"]) -> "Sample[D]":
        data_type = type(samples[0].sample)
        num_steps = samples[0].timesteps.shape[0] - 1

        all_kwargs = []
        for sample in samples:
            for i in range(len(sample)):
                all_kwargs.append(index_dict(sample.kwargs, i))

        return Sample(
            sample=data_type.collate([x.sample for x in samples]),
            latent=data_type.collate([x.latent for x in samples]),
            trajectory=[
                data_type.collate([x.trajectory[t] for x in samples])
                for t in range(num_steps + 1)
            ],
            timesteps=samples[0].timesteps,
            drifts=[
                data_type.collate([x.drifts[t] for x in samples])
                for t in range(num_steps)
            ],
            diffusions=[
                data_type.collate([x.diffusions[t] for x in samples])
                for t in range(num_steps)
            ],
            noises=[
                data_type.collate([x.noises[t] for x in samples])
                for t in range(num_steps)
            ],
            running_costs=torch.cat([x.running_costs for x in samples], dim=1),
            rewards=torch.cat([x.rewards for x in samples], dim=0),
            valids=torch.cat([x.valids for x in samples], dim=0),
            cost_functionals=torch.cat([x.cost_functionals for x in samples], dim=1),
            kwargs=default_collate(all_kwargs),
        )


class Policy[D: DDBatch](Protocol):
    """General protocol for a policy function."""

    def __call__(self, x: D, t: torch.Tensor, **kwargs) -> D: ...


class Environment[D: DDBatch](ABC):
    """Abstract base class for all environments.

    The current policy may be represented by:

    1. ``self.policy`` predicting the same quantity as ``base_model``;
    2. ``self.control_policy`` directly predicting an additive SDE control;
    3. both, in which case their reference-relative controls are added.

    Parameters
    ----------
    base_model : BaseModel[D]
        The base generative model used in the environment.
    reward : Reward[D]
        The reward function used to compute the final reward.
    discretization_steps : int
        The number of discretization steps to use when sampling trajectories.
    reward_scale : float, default=1.0
        Scale of the reward (can be negative). This is used to control trade-off between high rewards
        and proximity to base model.
    mode : EnvironmentMode, default=EnvironmentMode.BASE_INFERENCE
        How the environment's drift and control cost are used.
    """

    _SIGMA_EPS = 1e-8

    def __init__(
        self,
        base_model: BaseModel[D],
        reward: Reward[D],
        discretization_steps: int,
        reward_scale: float = 1.0,
        *,
        mode: EnvironmentMode = EnvironmentMode.BASE_INFERENCE,
    ):
        self.base_model = base_model
        self.reward = reward
        self.discretization_steps = discretization_steps
        self.reward_scale = reward_scale
        self.policy: Policy[D] | None = None
        self.control_policy: Policy[D] | None = None
        self.memoryless_schedule = MemorylessNoiseSchedule(self.scheduler)
        self.mode = mode

    @property
    def device(self) -> torch.device:
        """Get the device of the base model."""
        return self.base_model.device

    @property
    def scheduler(self) -> Scheduler[D]:
        """Get the scheduler of the base model."""
        return self.base_model.scheduler

    # ------------------------------------------------------------------
    # Abstract interface: subclasses implement these two methods
    # ------------------------------------------------------------------

    @abstractmethod
    def drift_from_prediction(
        self,
        x: D,
        t: torch.Tensor,
        prediction: D,
    ) -> D:
        """Convert one model prediction into the complete SDE drift.

        This must account for the scheduler's current diffusion coefficient.
        It must not assume that the prediction comes from the base model.
        """

    @abstractmethod
    def control_from_prediction_delta(
        self,
        x: D,
        t: torch.Tensor,
        prediction_delta: D,
    ) -> D:
        r"""Convert ``policy_prediction - base_prediction`` into SDE control.

        It must satisfy

            drift(policy_prediction)
            =
            drift(base_prediction)
            + sigma \* control_from_prediction_delta(delta).
        """

    # ------------------------------------------------------------------
    # Public drift API
    # ------------------------------------------------------------------

    def reference_drift(
        self,
        x: D,
        t: torch.Tensor,
        **kwargs,
    ) -> D:
        """Frozen reference drift.

        Use this for the lean-adjoint dynamics in Adjoint Matching.
        Do not wrap this method in ``torch.no_grad()``, because the adjoint
        requires derivatives with respect to ``x``.
        """
        base_prediction = self.base_model.forward(x, t, **kwargs)
        return self.drift_from_prediction(x, t, base_prediction)

    def current_drift(
        self,
        x: D,
        t: torch.Tensor,
        **kwargs,
    ) -> D:
        """Drift used to generate the current rollout."""
        base_prediction = self.base_model.forward(x, t, **kwargs)

        if self.mode is EnvironmentMode.BASE_INFERENCE:
            return self.drift_from_prediction(x, t, base_prediction)

        if self.policy is not None:
            current_prediction = self.policy(x, t, **kwargs)
        else:
            current_prediction = base_prediction

        drift = self.drift_from_prediction(x, t, current_prediction)

        if self.control_policy is not None:
            sigma = self.scheduler.sigma(x, t)
            direct_control = self.control_policy(x, t, **kwargs)
            drift = drift + sigma * direct_control

        return drift

    def reference_relative_control(
        self,
        x: D,
        t: torch.Tensor,
        **kwargs,
    ) -> D:
        """Return the total current-policy control relative to the base model.

        This is used for:

        - path-space KL regularization;
        - Adjoint Matching control targets;
        - diagnostics.

        It is not added again after ``current_drift`` has already used the
        native policy prediction.
        """
        sigma = self.scheduler.sigma(x, t)
        self._require_stochastic_schedule(sigma)

        total_control = x.zeros_like()
        base_prediction = self.base_model.forward(x, t, **kwargs)

        if self.policy is not None:
            policy_prediction = self.policy(x, t, **kwargs)
            prediction_delta = policy_prediction - base_prediction

            total_control = total_control + self.control_from_prediction_delta(
                x,
                t,
                prediction_delta,
            )

        if self.control_policy is not None:
            total_control = total_control + self.control_policy(x, t, **kwargs)

        return total_control

    def drift(
        self,
        x: D,
        t: torch.Tensor,
        **kwargs,
    ) -> tuple[D, torch.Tensor]:
        """Return rollout drift and the relevant control-energy cost."""
        if self.mode is EnvironmentMode.ADJOINT_MATCHING:
            self._require_memoryless_schedule(x, t)

        drift = self.current_drift(x, t, **kwargs)

        if self.mode in {
            EnvironmentMode.ADJOINT_MATCHING,
            EnvironmentMode.KL_REGULARIZED_RL,
        }:
            control = self.reference_relative_control(x, t, **kwargs)
            running_cost = 0.5 * control.square().aggregate("sum")
        else:
            running_cost = 0.5 * x.zeros_like().square().aggregate("sum")

        return drift, running_cost

    # ------------------------------------------------------------------
    # Schedule validation
    # ------------------------------------------------------------------

    def _require_memoryless_schedule(
        self,
        x: D,
        t: torch.Tensor,
    ) -> None:
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        if not torch.allclose(
            sigma.square().aggregate("sum"),
            (2.0 * eta).aggregate("sum"),
            rtol=1e-5,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Adjoint Matching requires the memoryless schedule sigma**2 == 2 * eta."
            )

    @staticmethod
    def _require_stochastic_schedule(sigma: D) -> None:
        if torch.any(sigma.aggregate("sum").abs() <= Environment._SIGMA_EPS):
            raise ValueError(
                "Reference-relative control is undefined where sigma == 0. "
                "Use interior stochastic training times. Deterministic "
                "inference does not require computing this control."
            )

    def diffusion(self, x: D, t: torch.Tensor) -> D:
        """Compute the diffusion term of the environment's dynamics.

        Parameters
        ----------
        x : D
            The current state.
        t : torch.Tensor, shape (n,)
            The current time step in [0, 1].

        Returns
        -------
        diffusion : D
            The diffusion term at time t.
        """
        return self.scheduler.sigma(x, t)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        pbar: bool = True,
        x0: D | None = None,
        **kwargs,
    ) -> Sample[D]:
        r"""Sample n trajectories from the environment.

        Parameters
        ----------
        n : int
            Number of trajectories to sample.
        pbar : bool, default: True
            Whether to display a progress bar.
        x0 : D, optional
            Initial states to start the trajectories from. If None, samples from :math:`p_0`.
        **kwargs : dict
            Additional keyword arguments to pass to the base model at every timestep (e.g. text
            embedding or class label).

        Returns
        -------
        Sample[D]
            A Sample object containing the sampled trajectories and associated data.
        """
        device = self.base_model.device

        ### Prepare for the foward SDE Pass ###

        latent, kwargs = self.base_model.sample_p0(n, **kwargs)

        # Set initial state if provided
        if x0 is not None:
            if len(x0) != n:
                raise ValueError(f"x0 contains {len(x0)} stats, but n={n}")
            latent = x0.to(device)

        latent, kwargs = self.base_model.preprocess(latent, **kwargs)

        ### Run Euler-Maruyama Forward Pass ###

        # Discrete time steps at which we advance
        timesteps = torch.linspace(
            0.0, 1.0, self.discretization_steps + 1, device=device
        )

        trajectory = [latent.detach().cpu()]  # Sample at each time step
        drifts = []  # Drift term at each time step
        diffusions = []  # Diffusion coeficient at each time step
        noises = []  # Sampled SDE noise at each time step

        running_costs = torch.zeros(self.discretization_steps, n, device=device)

        iterator = enumerate(pairwise(timesteps))
        for i, (t0, t1) in (
            tqdm(iterator, total=self.discretization_steps) if pbar else iterator
        ):
            dt = t1 - t0

            # To prevent anomalies, we do not evaluate at 0 but at 0.02
            t_eval = t0.clamp_min(2e-2)
            t_curr = t_eval.expand(n)

            # Discrete step of SDE
            drift, running_cost = self.drift(latent, t_curr, **kwargs)
            diffusion = self.diffusion(latent, t_curr)
            epsilon = latent.randn_like()
            latent += dt * drift + torch.sqrt(dt) * diffusion * epsilon

            # Integrate the running cost
            running_costs[i] = dt * running_cost

            trajectory.append(latent.detach().cpu())
            drifts.append(drift.detach().cpu())
            diffusions.append(diffusion.detach().cpu())
            noises.append(epsilon.detach().cpu())

        ### Postprocessing to obtain the final samples

        sample = self.base_model.postprocess(latent)
        rewards, valids = self.reward(sample, latent, **kwargs)

        ### Compute Costs ###

        # Discrete approximation of equation 9
        costs = torch.cat(
            [
                running_costs,
                -self.reward_scale * rewards.unsqueeze(0),
            ],
            dim=0,
        )
        cost_functionals = costs.flip(0).cumsum(0).flip(0)

        return Sample(
            sample=sample.detach().cpu(),
            latent=latent.detach().cpu(),
            trajectory=trajectory,
            timesteps=timesteps.detach().cpu(),
            drifts=drifts,
            diffusions=diffusions,
            noises=noises,
            running_costs=running_costs.detach().cpu(),
            rewards=rewards.detach().cpu(),
            valids=valids.detach().cpu(),
            cost_functionals=cost_functionals.detach().cpu(),
            kwargs=dict_to_device(kwargs, "cpu"),
        )

    def batch_sample(
        self, n: int, batch_size: int, pbar: bool = False, **kwargs
    ) -> Sample[D]:
        """Sample n trajectories from the environment in batches.

        Parameters
        ----------
        n : int
            Number of trajectories to sample.
        batch_size : int
            Batch size for sampling.
        pbar : bool, default=False
            Whether to display progress bars or not.
        **kwargs : dict
            Additional keyword arguments to pass to the base model at every timestep (e.g. text
            embedding or class label).
        """
        samples: list[Sample[D]] = []

        iterator = trange(0, n, batch_size) if pbar else range(0, n, batch_size)
        for i in iterator:
            current_n = min(batch_size, n - i)
            current_kwargs = index_dict(kwargs, i, i + current_n)
            samples.append(self.sample(current_n, pbar=False, **current_kwargs))

        return Sample.concat(samples)
