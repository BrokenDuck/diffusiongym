r"""Environment where the base model predicts the score :math:`\nabla \log p_t(x)`."""

import torch

from diffusiongym.environments.base import Environment
from diffusiongym.types import DDMixin


class ScoreEnvironment[D: DDMixin](Environment[D]):
    r"""Environment where the base model predicts the score :math:`\nabla \log p_t(x)`."""

    def drift_from_prediction(
        self,
        x: D,
        t: torch.Tensor,
        prediction: D,
    ) -> D:
        kappa = self.scheduler.kappa(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        score_coefficient = eta + 0.5 * sigma.square()

        return kappa * x + score_coefficient * prediction

    def control_from_prediction_delta(
        self,
        x: D,
        t: torch.Tensor,
        prediction_delta: D,
    ) -> D:
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        score_coefficient = eta + 0.5 * sigma.square()

        return (score_coefficient / sigma) * prediction_delta
