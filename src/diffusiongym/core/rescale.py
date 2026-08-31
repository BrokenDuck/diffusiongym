"""Temporal Score Rescaling: a local sampling temperature for affine flow models.

TSR (Xu et al., arXiv:2510.01184) multiplies the learned score by a
time-dependent scalar during sampling, and nothing else about the sampler
changes. Writing the forward process as `x_t = b(t) x_data + a(t) x_base` and
its signal-to-noise ratio as `SNR(t) = b(t)^2 / a(t)^2`,

    s~(x, t) = r_t * s(x, t),      r_t(k, sigma) = (SNR*sigma^2 + 1)
                                                  / (SNR*sigma^2/k + 1).

`r_t` runs monotonically from `1` at the noise end to `k` at the data end, with
the crossover at the noise level `a/b ~ sigma`. That schedule is the whole
point. Scaling the score by a *constant* — the obvious way to sample "colder"
or "hotter" — over-suppresses exploration where the model is still deciding
which mode to head for and under-suppresses it near the data, which biases
samples toward central modes and drops peripheral ones (the paper's Theorem C.1
shows no prior generates a temperature-scaled law that way). Leaving `r_t = 1`
at high noise keeps mode *selection* untouched and rescales only the local
spread once a mode has been chosen: for a Gaussian mixture the target is
`sum_m w_m N(mu_m, Sigma_m / k)` — same weights, rescaled covariances. It is
therefore not `p^(1/T)`, which reweights the modes as well.

`k > 1` sharpens (higher fidelity, less diversity), `k < 1` broadens, `k = 1` is
the exact identity.

**Velocity form.** All of this framework works in velocity space, so the score
rescaling is carried through the affine change of variables to

    v~ = (db/b) x_t + r_t * (v - (db/b) x_t),

i.e. the velocity is rescaled about the pure-scaling drift `(db/b) x_t` rather
than about zero. `db/b` is singular at `t = 0` for any schedule with `b(0) = 0`,
but `r_t - 1` vanishes there fast enough to cancel it, and `_coefficients`
below evaluates the cancelled form so no epsilon or interior-time restriction
is needed.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from diffusiongym.core.model import FlowModel, PredictionConverter, PredictionKind
from diffusiongym.core.schedule import AffineSchedule
from diffusiongym.types import DDBatch

__all__ = ["TemporalScoreRescaling"]

type Conditioning = Mapping[str, Any]


class TemporalScoreRescaling[StateT: DDBatch]:
    """A `FlowModel` wrapper applying TSR to whatever the inner model predicts.

    Declares `prediction_kind = VELOCITY` and returns velocity, so it drops into
    any sampler in place of the model it wraps: `VelocityRegression.predict`
    passes the output through unconverted, and `PredictionConverter.to_endpoint`
    is called with an explicit `VELOCITY` kind everywhere it matters.

    Sampling only. It holds no parameters and exposes no `state_dict`, because
    rescaling the field a *training* loss regresses onto would change what is
    being fit rather than how it is sampled.

    Parameters
    ----------
    model:
        The policy to sample from. Its native prediction kind is respected.
    schedule, converter:
        The environment's own affine schedule and prediction converter — pass
        `environment.forward_process.schedule` and
        `environment.regression.converter` so the rescaling is defined against
        the same path the model was trained on.
    k:
        The rescaling factor reached at the data end. `> 1` sharpens, `< 1`
        broadens, `1` is a no-op.
    sigma:
        The noise level at which rescaling engages, read as the assumed
        per-mode spread of the data. Larger `sigma` steers earlier in sampling.
    """

    prediction_kind = PredictionKind.VELOCITY

    def __init__(
        self,
        model: FlowModel[StateT],
        *,
        schedule: AffineSchedule,
        converter: PredictionConverter[StateT],
        k: float = 1.0,
        sigma: float = 1.0,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}.")
        self.model = model
        self.schedule = schedule
        self.converter = converter
        self.k = k
        self.sigma = sigma

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def is_identity(self) -> bool:
        """Whether this rescaling is the exact no-op, `k = 1`."""
        return self.k == 1.0

    def _coefficients(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """`(r_t - 1, (r_t - 1) * db/b)` at each time, both finite everywhere.

        Substituting `SNR = b^2/a^2` into `r_t` and cancelling the `a^2`:

            r_t - 1 = b^2 sigma^2 (1 - 1/k) / (b^2 sigma^2 / k + a^2)

        whose `b^2` numerator kills the `db/b` pole of the second coefficient
        one power over. The denominator is `sigma^2/k > 0` at the data end and
        `a(0)^2 = 1` at the noise end, so it never approaches zero either.
        """
        a, b = self.schedule.a(t), self.schedule.b(t)
        db = self.schedule.db_dt(t)
        scale = self.sigma**2 * (1.0 - 1.0 / self.k)
        denominator = b.square() * self.sigma**2 / self.k + a.square()
        return b.square() * scale / denominator, b * db * scale / denominator

    def __call__(
        self,
        x_t: StateT,
        t: Tensor,
        *,
        conditioning: Conditioning,
    ) -> StateT:
        velocity = self.converter.to_velocity(
            prediction=self.model(x_t, t, conditioning=conditioning),
            kind=self.model.prediction_kind,
            x_t=x_t,
            t=t,
        )
        if self.is_identity:
            return velocity
        gain, pull = self._coefficients(t)
        return velocity * (1.0 + gain) - x_t * pull
