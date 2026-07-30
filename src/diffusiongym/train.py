"""Main training loop."""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from diffusiongym.base_models import BaseModel
from diffusiongym.types import DDMixin
from diffusiongym.utils import dict_to_device, index_dict


class DDDataset[D: DDMixin](Dataset[tuple[D, dict[str, Any], torch.Tensor]]):
    """Dataset wrapper for diffusiongym data."""

    def __init__(self, data: list[D], kwargs: list[dict[str, Any]] | None, weights: list[torch.Tensor] | None):
        if len(data) == 0:
            raise ValueError("Data list is empty.")

        # Combine all data into a single object
        self.data = type(data[0]).collate(data)

        if weights is None:
            self.weights = torch.ones(len(self.data))
        else:
            self.weights = torch.cat(weights, dim=0)

        if kwargs is None:
            kwargs = [{}] * len(data)

        all_kwargs = []
        for d, k in zip(data, kwargs, strict=False):
            for i in range(len(d)):
                all_kwargs.append(index_dict(k, i))

        self.kwargs: dict = default_collate(all_kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[D, dict[str, Any], torch.Tensor]:  # ty:ignore[invalid-method-override]
        if self.kwargs is None:
            return self.data[idx], {}, self.weights[idx]

        return self.data[idx], index_dict(self.kwargs, idx), self.weights[idx]

    def collate(self, batch):
        data_batch, kwargs_batch, weight_batch = zip(*batch, strict=False)
        data_batch = type(data_batch[0]).collate(list(data_batch))
        kwargs_batch = default_collate(kwargs_batch)
        weight_batch = default_collate(weight_batch)
        return data_batch, kwargs_batch, weight_batch


def train_base_model[D: DDMixin](
    base_model: BaseModel[D],
    opt: torch.optim.Optimizer,
    data: list[D],
    kwargs: list[dict] | None = None,
    weights: list[torch.Tensor] | None = None,
    steps: int = 1000,
    batch_size: int = 64,
    accumulate_steps: int = 1,
    pbar: bool = False,
) -> None:
    """Trains/fine-tunes a base model.

    Parameters
    ----------
    base_model : BaseModel[D]
        The model to train.
    opt : torch.optim.Optimizer
        Optimizer to use.
    data : list[D]
        The training data.
    kwargs : list[dict]
        Keyword arguments corresponding to the data.
    weights : list[torch.Tensor]
        Training weights for the data.
    steps : int
        Number of training steps.
    batch_size : int
        Batch size.
    accumulate_steps : int
        Number of gradient accumulation steps.
    pbar : bool, default: False
        Whether to display a tqdm progress bar or not.
    """
    dataset = DDDataset(data, kwargs, weights)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=0,
        pin_memory=False,
    )

    base_model.train()
    opt.zero_grad()

    # Create an iterator for the dataloader
    data_iter = iter(loader)

    iterator = range(steps)
    if pbar:
        iterator = tqdm(iterator)

    loss_sum = 0.0
    grad_norm_sum = 0.0
    n_steps = 0
    for _ in iterator:
        n_steps += 1

        # Get the next batch. If the loader is exhausted, restart it.
        try:
            x1_cpu, kwargs, weight = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x1_cpu, kwargs, weight = next(data_iter)

        x1_cpu: D
        x1 = x1_cpu.to(base_model.device)
        weight = weight.to(base_model.device)
        kwargs = dict_to_device(kwargs, base_model.device)

        loss = (weight * base_model.train_loss(x1, **kwargs)).mean()
        loss_sum += loss.item()

        loss = loss / accumulate_steps
        loss.backward()

        if n_steps % accumulate_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(base_model.parameters(), 0.1)
            grad_norm_sum += grad_norm.item()
            opt.step()
            opt.zero_grad()

        if isinstance(iterator, tqdm):
            iterator.set_postfix({"loss": loss_sum / n_steps, "grad_norm": grad_norm_sum / n_steps})

    base_model.eval()
