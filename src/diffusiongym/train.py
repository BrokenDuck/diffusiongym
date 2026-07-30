"""Main training loop."""

import itertools
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusiongym.base_models import BaseModel
from diffusiongym.types import DDBatch, DDDataset
from diffusiongym.utils import dict_to_device


def train_base_model[D: DDBatch](
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
    """Trains/fine-tunes a base model over the given data

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
    ### Load dataset ###

    dataset = DDDataset(data, kwargs, weights)
    loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        num_workers=0,
        pin_memory=False,
    )
    data_iter = itertools.islice(itertools.cycle(loader), steps)
    iterator = tqdm(data_iter) if pbar else data_iter

    ### Init training ####

    base_model.train()
    opt.zero_grad()

    loss_sum = 0.0
    grad_norm_sum = 0.0

    ### Training Loop ###

    batch: D
    kwarg: dict[str, Any]
    weight: torch.Tensor
    for i, (batch, kwarg, weight) in enumerate(iterator):
        x1 = batch.to(base_model.device)
        weight = weight.to(base_model.device)
        kwarg = dict_to_device(kwarg, base_model.device)

        ### Forward pass ###
        train_loss = base_model.train_loss(x1, **kwarg)
        loss = (weight * train_loss).mean() / accumulate_steps

        ### Backward pass ###

        loss.backward()

        if i % accumulate_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), 0.1)
            grad_norm_sum += grad_norm.item()
            opt.step()
            opt.zero_grad()

        ### Logging ###
        if pbar:
            loss_sum += loss.item()
            iterator.set_postfix(  # ty: ignore[unresolved-attribute]
                {"loss": loss_sum / i, "grad_norm": grad_norm_sum / i}
            )

    base_model.eval()
