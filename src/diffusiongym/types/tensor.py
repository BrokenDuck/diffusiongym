from collections.abc import Sequence
from typing import Self

import torch

from diffusiongym.types.batch import BinaryOp, DDBatch, UnaryOp


class DDTensor(DDBatch):
    """A DDType wrapper around torch.Tensor."""

    def __init__(self, data: torch.Tensor):
        if not isinstance(data, torch.Tensor):
            raise TypeError("DDTensor expects a torch.Tensor")

        if data.ndim < 1:
            raise ValueError("DDTensor expects a tensor with at least 1 dimension")

        self.data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={tuple(self.data.shape)}, dtype={self.data.dtype}, device={self.data.device})"

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int | slice) -> Self:
        data_out = self.data[idx]

        if data_out.ndim < self.data.ndim:
            data_out = data_out.unsqueeze(0)

        return self.__class__(data_out)

    @classmethod
    def collate(cls, items: Sequence[Self]) -> Self:  # ty: ignore[invalid-method-override]
        if not items:
            raise ValueError("Cannot collate an empty sequence")

        tensors = [item.data for item in items]
        return cls(torch.cat(tensors, dim=0))

    def aggregate(self, reduction: str = "mean") -> torch.Tensor:
        dims = tuple(range(1, self.data.ndim))
        reducers = {
            "mean": torch.mean,
            "sum": torch.sum,
        }

        reducer = reducers.get(reduction, None)
        if reducer is None:
            raise ValueError(f"Unsupported reduction type: {reduction}")

        return reducer(self.data, dim=dims)

    def apply(self, op: UnaryOp) -> Self:
        return self.__class__(op(self.data))

    def combine(self, other: Self, op: BinaryOp) -> Self:  # ty:ignore[invalid-method-override]
        return self.__class__(op(self.data, other.data))
