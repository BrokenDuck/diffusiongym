"""Geometry and Gaussian measure on a DDBatch latent state."""

from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Generator, Tensor

from diffusiongym.types import DDBatch, DDTensor

NormReduction = Literal["mean", "sum"]


@runtime_checkable
class LatentGeometry[D: DDBatch](Protocol):
    """Geometry and Gaussian measure on a DDBatch representation.

    Arithmetic (add, subtract, scale, clone, detach, index, concat) lives on
    DDBatch itself. This protocol covers only the geometry-dependent operations:
    projection, constrained Gaussian sampling, norms, and effective dimensionality.
    """

    def project(self, x: D) -> D:
        """Project onto the modeled linear subspace (identity for unconstrained spaces)."""
        ...

    def standard_normal_like(
        self, x: D, *, generator: Generator | None = None
    ) -> D:
        """Sample N(0, I) in the modeled latent subspace."""
        ...

    def squared_norm(self, x: D, *, reduction: NormReduction) -> Tensor:
        """Return one value per batch element.

        Parameters
        ----------
        reduction:
            "mean" divides by active dimensions (used for training loss).
            "sum" returns the raw sum of squares (required for Gaussian log-probability).
        """
        ...

    def active_dimensions(self, x: D) -> Tensor:
        """Effective continuous dimensionality of each batch element, shape (batch,).

        Needed for normalized Gaussian log densities: log p = -d/2 * log(2π σ²) - ...
        """
        ...


class TensorGeometry:
    """Euclidean geometry for DDTensor states.

    No constraints or masks — identity projection. Covers toy vectors,
    image latents (SD3.5), and any unconstrained Euclidean flow model.
    """

    def project(self, x: DDTensor) -> DDTensor:
        return x

    def standard_normal_like(
        self, x: DDTensor, *, generator: Generator | None = None
    ) -> DDTensor:
        noise = torch.randn(
            x.data.shape,
            dtype=x.data.dtype,
            device=x.data.device,
            generator=generator,
        )
        return DDTensor(noise)

    def active_dimensions(self, x: DDTensor) -> Tensor:
        dimensions = x.data[0].numel()
        return torch.full(
            (len(x),),
            dimensions,
            dtype=torch.long,
            device=x.device,
        )

    def squared_norm(self, x: DDTensor, *, reduction: NormReduction) -> Tensor:
        sums = x.data.square().flatten(start_dim=1).sum(dim=1)
        match reduction:
            case "sum":
                return sums
            case "mean":
                return sums / self.active_dimensions(x).to(sums.dtype)
            case _:
                raise ValueError(f"Unsupported reduction: {reduction!r}.")
