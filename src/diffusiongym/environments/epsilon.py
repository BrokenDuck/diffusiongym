r"""Environment where the base model predicts the noise :math:`\epsilon(x, t)`."""

import torch

from diffusiongym.environments.base import Environment
from diffusiongym.types import DDBatch


class EpsilonEnvironment[D: DDBatch](Environment[D]):
    r"""Environment where the base model predicts the noise :math:`\epsilon(x, t)`."""

    def drift_from_prediction(
        self,
        x: D,
        t: torch.Tensor,
        prediction: D,
    ) -> D:
        beta = self.scheduler.beta(x, t)
        kappa = self.scheduler.kappa(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        score_coefficient = eta + 0.5 * sigma.square()

        return kappa * x - (score_coefficient / beta) * prediction

    def control_from_prediction_delta(
        self,
        x: D,
        t: torch.Tensor,
        prediction_delta: D,
    ) -> D:
        beta = self.scheduler.beta(x, t)
        eta = self.scheduler.eta(x, t)
        sigma = self.scheduler.sigma(x, t)

        score_coefficient = eta + 0.5 * sigma.square()

        return -(score_coefficient / (sigma * beta)) * prediction_delta
