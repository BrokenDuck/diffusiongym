"""Utility functions for diffusiongym."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager

import torch
from torch import nn

from diffusiongym.schedulers import NoiseSchedule
from diffusiongym.types import DDBatch


def identity_fn[T](x: T) -> T:
    """Identity function."""
    return x


def index_dict[T](d: T, start: int, end: int | None = None) -> T:
    """Recursively index into the leaves of a nested dictionary.

    Parameters
    ----------
    d : T
        Any value, if a dictionary, will be processed recursively.
    start : int
        The index to select from list/tensor leaves.
    end : Optional[int], optional
        The end index to select from list/tensor leaves, by default None.

    Returns
    -------
    T
        If d is a dictionary, returns a dictionary with the same keys and indexed leaves.
    """
    if end is None:
        idx = start
    else:
        idx = slice(start, end)

    if isinstance(d, dict):
        return {k: index_dict(v, start, end) for k, v in d.items()}  # ty:ignore[invalid-return-type]

    if isinstance(d, (list, tuple, torch.Tensor)):
        return d[idx]  # ty:ignore[invalid-return-type]

    if isinstance(d, (float, int, str)):
        return d

    raise TypeError(f"Unsupported leaf type: {type(d)}")


def append_dims(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """Match the number of dimensions of x to ndim by adding dimensions at the end.

    Parameters
    ----------
    x : torch.Tensor, shape (*shape)
        The input tensor.
    ndim : int
        The target number of dimensions.

    Returns
    -------
    x : torch.Tensor, shape (*shape, 1, ..., 1)
        The reshaped tensor with ndim dimensions.
    """
    if x.ndim > ndim:
        return x

    shape = x.shape + (1,) * (ndim - x.ndim)
    return x.view(shape)


def dict_to_device[T](d: T, device: torch.device | str) -> T:
    """Recursively move the leaves of a nested dictionary to a specified device.

    Parameters
    ----------
    d : T
        Any value, if a dictionary, will be processed recursively.
    device : torch.device
        The device to move tensor leaves to.

    Returns
    -------
    T
        If d is a dictionary, returns a dictionary with the same keys and device-moved leaves.
    """
    if isinstance(d, dict):
        return {k: dict_to_device(v, device) for k, v in d.items()}  # ty:ignore[invalid-return-type]

    if isinstance(d, list):
        return [dict_to_device(v, device) for v in d]  # ty:ignore[invalid-return-type]

    if isinstance(d, torch.Tensor):
        return d.to(device)  # ty:ignore[invalid-return-type]

    if isinstance(d, (float, int, str)):
        return d

    raise TypeError(f"Unsupported leaf type: {type(d)}")


@contextmanager
def temporary_workdir() -> Generator[str]:
    """Context manager that runs code in a fresh temporary directory.

    When exiting the context, it returns to the original working directory and deletes the temporary
    folder.
    """
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            yield tmp
        finally:
            os.chdir(old_cwd)


class ValuePolicy[D: DDBatch](nn.Module):
    r"""Policy based on a value function, :math:`u(x, t) = -\sigma(t) \nabla_x V(x, t)`.

    Parameters
    ----------
    value_network : nn.Module
        The value function network, :math:`V(x, t)`.

    noise_schedule : NoiseSchedule
        The noise schedule, :math:`\sigma(t)`.
    """

    def __init__(
        self, value_network: nn.Module, noise_schedule: NoiseSchedule[D]
    ) -> None:
        super().__init__()
        self.value_network = value_network
        self.noise_schedule = noise_schedule

    @torch.enable_grad()
    def forward(self, x: D, t: torch.Tensor, **kwargs) -> D:
        """Compute control action based on value function gradient."""
        x = x.requires_grad()
        value_pred = self.value_network(x, t, **kwargs)
        sigma = self.noise_schedule(x, t)
        control: D = -sigma * x.gradient(value_pred)
        return control
