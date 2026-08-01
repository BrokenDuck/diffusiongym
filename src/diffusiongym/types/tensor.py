from collections.abc import Iterable, Sequence
from typing import Self

import torch
from torch import Tensor

from diffusiongym.types.batch import BatchIndex, DDBatch, Scale, UnaryOp


class DDTensor(DDBatch):
    """A batched dense floating-point tensor state."""

    def __init__(self, data: Tensor) -> None:
        if not isinstance(data, Tensor):
            raise TypeError("DDTensor expects a torch.Tensor.")
        if data.ndim < 1:
            raise ValueError("DDTensor expects a tensor with a batch dimension.")
        if not data.dtype.is_floating_point:
            raise TypeError("DDTensor's state tensor must use a floating-point dtype.")
        self.data = data

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"shape={tuple(self.data.shape)}, "
            f"dtype={self.data.dtype}, "
            f"device={self.data.device})"
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def index_select(self, index: BatchIndex) -> Self:
        if isinstance(index, int):
            index = slice(index, index + 1)
        selected = self.data[index]
        if selected.ndim == self.data.ndim - 1:
            selected = selected.unsqueeze(0)
        return type(self)(selected)

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:  # ty: ignore[invalid-method-override]
        if not batches:
            raise ValueError("Cannot concatenate an empty sequence.")
        return cls(torch.cat([b.data for b in batches], dim=0))

    def all_tensors(self) -> Iterable[Tensor]:
        return (self.data,)

    def state_tensors(self) -> tuple[Tensor]:
        return (self.data,)

    def replace_state_tensors(self, tensors: Sequence[Tensor]) -> Self:
        if len(tensors) != 1:
            raise ValueError(f"DDTensor expects one state tensor, got {len(tensors)}.")
        return type(self)(tensors[0])

    def map_all_tensors(self, op: UnaryOp) -> Self:
        return type(self)(op(self.data))

    def assert_compatible(self, other: Self) -> None:  # ty: ignore[invalid-method-override]
        if self.data.shape != other.data.shape:
            raise ValueError(
                f"Tensor states have incompatible shapes: "
                f"{tuple(self.data.shape)} versus {tuple(other.data.shape)}."
            )

    def scale(self, coefficient: Scale) -> Self:
        if isinstance(coefficient, Tensor):
            coefficient = coefficient.to(device=self.data.device, dtype=self.data.dtype)
            if coefficient.ndim == 0:
                pass
            elif coefficient.shape == (len(self),):
                coefficient = coefficient.reshape(
                    len(self), *([1] * (self.data.ndim - 1))
                )
            else:
                raise ValueError(
                    f"A DDTensor scale must be scalar or have shape ({len(self)},), "
                    f"got {tuple(coefficient.shape)}."
                )
        return type(self)(self.data * coefficient)
