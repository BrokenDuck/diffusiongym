r"""Environment where the base model predicts the endpoint :math:`\hat{x}_1(x, t)`."""

import torch

from diffusiongym.environments.base import Environment
from diffusiongym.types import DDMixin


class EndpointEnvironment[D: DDMixin](Environment[D]):
    r"""Environment where the base model predicts the endpoint :math:`\hat{x}_1(x, t)`."""

    def drift_from_prediction(
        self,
        x: D,
        t: torch.Tensor,
        prediction: D,
    ) -> D:
        alpha = self.scheduler.alpha(x, t)
        beta = self.scheduler.beta(x, t)
        kappa = self.scheduler.kappa(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        score_coefficient = eta + 0.5 * sigma.square()
        beta_sq = beta.square()

        return (kappa - score_coefficient / beta_sq) * x + (score_coefficient * alpha / beta_sq) * prediction

    def control_from_prediction_delta(
        self,
        x: D,
        t: torch.Tensor,
        prediction_delta: D,
    ) -> D:
        alpha = self.scheduler.alpha(x, t)
        beta = self.scheduler.beta(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        score_coefficient = eta + 0.5 * sigma.square()

        return (score_coefficient * alpha / (sigma * beta.square())) * prediction_delta
