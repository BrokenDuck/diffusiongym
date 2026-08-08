"""Flow model protocol, prediction conversion, velocity regression, and adapters.

All fine-tuning algorithms work in velocity space. PredictionConverter converts
native model outputs (velocity, endpoint, noise) to the canonical path velocity
before any loss or dynamics computation.
"""

from collections.abc import Mapping
from enum import Enum, auto
from typing import Any, Protocol

import torch
from torch import Tensor

from diffusiongym.base_models import BaseModel
from diffusiongym.core.schedule import AffineSchedule
from diffusiongym.core.space import LatentGeometry
from diffusiongym.types import DDBatch

type Conditioning = Mapping[str, Any]


class PredictionKind(Enum):
    """Native output type of a flow model."""

    VELOCITY = auto()
    ENDPOINT = auto()
    NOISE = auto()


class FlowModel[StateT](Protocol):
    """Minimal flow model contract.

    Concrete implementations may wrap an nn.Module, SD3.5, FlowMol, or a toy MLP.
    Use an adapter (TorchFlowModelAdapter) rather than modifying third-party classes.
    """

    prediction_kind: PredictionKind

    @property
    def device(self) -> torch.device: ...

    def __call__(
        self,
        x_t: StateT,
        t: Tensor,
        *,
        conditioning: Conditioning,
    ) -> StateT: ...


class PredictionConverter[StateT: DDBatch]:
    """Convert native affine-flow predictions to path velocity.

    All fine-tuning algorithms request velocity regardless of the native output.
    Velocity is the canonical training field because it directly matches the
    flow-matching regression objective.

    Endpoint conversion: infer noise from x_t and endpoint, then compose.
    Noise conversion: infer endpoint from x_t and noise, then compose.
    Both are singular at the schedule boundary; the caller must use interior times.
    """

    def __init__(
        self,
        *,
        geometry: LatentGeometry[StateT],
        schedule: AffineSchedule,
        denominator_epsilon: float = 1e-6,
    ) -> None:
        self.geometry = geometry
        self.schedule = schedule
        self.denominator_epsilon = denominator_epsilon

    def to_velocity(
        self,
        *,
        prediction: StateT,
        kind: PredictionKind,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        """Convert a native prediction to path velocity."""
        match kind:
            case PredictionKind.VELOCITY:
                return prediction
            case PredictionKind.ENDPOINT:
                return self._endpoint_to_velocity(endpoint=prediction, x_t=x_t, t=t)
            case PredictionKind.NOISE:
                return self._noise_to_velocity(noise=prediction, x_t=x_t, t=t)

    def _endpoint_to_velocity(
        self,
        *,
        endpoint: StateT,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        """v = da_dt * z_inferred + db_dt * endpoint, where z = (x_t - b*endpoint) / a."""
        a = self.schedule.a(t)
        b = self.schedule.b(t)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)

        if torch.any(a.abs() < self.denominator_epsilon):
            raise ValueError(
                "Cannot convert endpoint prediction to velocity because a(t) ≈ 0. "
                "Use interior sampling times (t < 1 - ε)."
            )

        x_base_inferred = (x_t - endpoint * b) * (1.0 / a)
        return x_base_inferred * da + endpoint * db

    def _noise_to_velocity(
        self,
        *,
        noise: StateT,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        """v = da_dt * noise + db_dt * x_data_inferred, where x_data = (x_t - a*noise) / b."""
        a = self.schedule.a(t)
        b = self.schedule.b(t)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)

        if torch.any(b.abs() < self.denominator_epsilon):
            raise ValueError(
                "Cannot convert noise prediction to velocity because b(t) ≈ 0. "
                "Use interior sampling times (t > ε)."
            )

        x_data_inferred = (x_t - noise * a) * (1.0 / b)
        return noise * da + x_data_inferred * db

    def to_endpoint(
        self,
        *,
        prediction: StateT,
        kind: PredictionKind,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        """Convert a native prediction to the predicted data endpoint x̂₁.

        Used by inference-time guidance (SMC, see `core/smc.py`): scoring "where
        this trajectory is headed" needs an endpoint estimate, not velocity, at
        every intermediate step.
        """
        match kind:
            case PredictionKind.ENDPOINT:
                return prediction
            case PredictionKind.VELOCITY:
                return self._velocity_to_endpoint(velocity=prediction, x_t=x_t, t=t)
            case PredictionKind.NOISE:
                return self._noise_to_endpoint(noise=prediction, x_t=x_t, t=t)

    def _velocity_to_endpoint(
        self,
        *,
        velocity: StateT,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        """x1 = (a*v - da_dt*x_t) / (a*db_dt - b*da_dt).

        Inverts x_t = a*x0 + b*x1, v = da_dt*x0 + db_dt*x1 for x1 via the
        schedule's Wronskian W = a*db_dt - b*da_dt (W ≡ 1 for
        `RectifiedFlowSchedule`, so this reduces to x1 = x_t + (1-t)*v). Unlike
        `_endpoint_to_velocity`/`_noise_to_velocity`, this is well-defined at
        both t=0 and t=1 — it only needs W ≠ 0, not a(t) or b(t) individually
        nonzero — so no interior-time restriction applies here.
        """
        a = self.schedule.a(t)
        b = self.schedule.b(t)
        da = self.schedule.da_dt(t)
        db = self.schedule.db_dt(t)
        wronskian = a * db - b * da

        if torch.any(wronskian.abs() < self.denominator_epsilon):
            raise ValueError(
                "Cannot convert velocity prediction to endpoint because the "
                "schedule's Wronskian a*db_dt - b*da_dt ≈ 0."
            )

        return (velocity * a - x_t * da) * (1.0 / wronskian)

    def _noise_to_endpoint(
        self,
        *,
        noise: StateT,
        x_t: StateT,
        t: Tensor,
    ) -> StateT:
        """x1 = (x_t - a*noise) / b."""
        a = self.schedule.a(t)
        b = self.schedule.b(t)

        if torch.any(b.abs() < self.denominator_epsilon):
            raise ValueError(
                "Cannot convert noise prediction to endpoint because b(t) ≈ 0. "
                "Use interior sampling times (t > ε)."
            )

        return (x_t - noise * a) * (1.0 / b)


class VelocityRegression[StateT: DDBatch]:
    """Shared regression primitive used by all fine-tuning algorithms.

    Predicts the path velocity from any native model output and computes
    the per-example squared error against a target velocity.
    """

    def __init__(
        self,
        *,
        geometry: LatentGeometry[StateT],
        converter: PredictionConverter[StateT],
    ) -> None:
        self.geometry = geometry
        self.converter = converter

    def predict(
        self,
        model: FlowModel[StateT],
        *,
        x_t: StateT,
        t: Tensor,
        conditioning: Conditioning,
    ) -> StateT:
        """Run the model and convert output to velocity."""
        native = model(x_t, t, conditioning=conditioning)
        return self.converter.to_velocity(
            prediction=native,
            kind=model.prediction_kind,
            x_t=x_t,
            t=t,
        )

    def per_example_error(
        self,
        predicted_velocity: StateT,
        target_velocity: StateT,
    ) -> Tensor:
        """Per-sample mean squared error, shape (n,).

        Uses "mean" reduction (divides by active dimensions) for training.
        Use geometry.squared_norm(..., reduction="sum") directly for log-probabilities.
        """
        return self.geometry.squared_norm(
            predicted_velocity - target_velocity, reduction="mean"
        )


# ---------------------------------------------------------------------------
# Adapters from BaseModel (old interface) to FlowModel (new interface)
# ---------------------------------------------------------------------------

_OUTPUT_TYPE_TO_KIND: dict[str, PredictionKind] = {
    "velocity": PredictionKind.VELOCITY,
    "endpoint": PredictionKind.ENDPOINT,
    "epsilon": PredictionKind.NOISE,
    "score": PredictionKind.VELOCITY,  # score ≈ velocity in this framework
}


class TorchFlowModelAdapter[D: DDBatch]:
    """Adapts a BaseModel[D] instance to the FlowModel[D] protocol.

    Translates the conditioning Mapping to keyword arguments for BaseModel.forward().
    """

    def __init__(self, base_model: BaseModel[D]) -> None:
        self._model = base_model
        kind = _OUTPUT_TYPE_TO_KIND.get(base_model.output_type)
        if kind is None:
            raise ValueError(
                f"Unknown output_type {base_model.output_type!r}. "
                f"Expected one of: {list(_OUTPUT_TYPE_TO_KIND)}"
            )
        self.prediction_kind = kind

    @property
    def device(self) -> torch.device:
        return self._model.device

    @property
    def parameters(self):
        return self._model.parameters

    def __call__(self, x_t: D, t: Tensor, *, conditioning: Conditioning) -> D:
        return self._model.forward(x_t, t, **conditioning)

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        return self._model.load_state_dict(state_dict)


class TorchBaseSampler[D: DDBatch]:
    """Adapts a BaseModel[D].sample_p0 to the BaseSampler[D] protocol."""

    def __init__(self, base_model: BaseModel[D]) -> None:
        self._model = base_model

    def sample(
        self,
        n: int,
        *,
        conditioning: Conditioning,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[D, Conditioning]:
        samples, extra_kwargs = self._model.sample_p0(n, **conditioning)
        merged = dict(conditioning) | extra_kwargs
        return samples.to(device), merged

    def sample_like(
        self,
        x_data: D,
        *,
        generator: torch.Generator | None = None,
    ) -> D:
        n = len(x_data)
        samples, _ = self._model.sample_p0(n)
        return samples.to(x_data.device)
