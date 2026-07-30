from typing import Any

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from diffusiongym.types.batch import DDBatch
from diffusiongym.utils import index_dict

type DDDatasetD[D: DDBatch] = tuple[D, dict[str, Any], torch.Tensor]


class DDDataset[D: DDBatch](Dataset[DDDatasetD[D]]):
    """Dataset wrapper for diffusiongym batches."""

    def __init__(
        self,
        batches: list[D],
        kwargs: list[dict[str, Any]] | None,
        weights: list[torch.Tensor] | None,
    ):
        assert len(batches) != 0, "Batches list is empty."

        if kwargs is None:
            kwargs = [{}] * len(batches)

        assert len(batches) == len(kwargs), (
            "Kwargs should be the same length as batches."
        )

        all_kwargs = []
        for batch, kwarg in zip(batches, kwargs, strict=False):
            for i in range(len(batch)):
                all_kwargs.append(index_dict(kwarg, i))

        self.data = type(batches[0]).collate(batches)
        self.kwargs: dict = default_collate(all_kwargs)
        self.weights = (
            torch.ones(len(self.data)) if weights is None else torch.cat(weights, dim=0)
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[D, dict[str, Any], torch.Tensor]:  # ty:ignore[invalid-method-override]
        if self.kwargs is None:
            return self.data[idx], {}, self.weights[idx]

        return self.data[idx], index_dict(self.kwargs, idx), self.weights[idx]

    def collate(self, batch: list[DDDatasetD[D]]):
        data_batch, kwargs_batch, weight_batch = zip(*batch, strict=False)
        data_batch = type(data_batch[0]).collate(list(data_batch))
        kwargs_batch = default_collate(kwargs_batch)
        weight_batch = default_collate(weight_batch)
        return data_batch, kwargs_batch, weight_batch
