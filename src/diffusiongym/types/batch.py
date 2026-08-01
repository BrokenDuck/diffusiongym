from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Self

import torch
from torch import Tensor

type UnaryOp = Callable[[Tensor], Tensor]
type BinaryOp = Callable[[Tensor, Tensor], Tensor]
type BatchIndex = int | slice | Tensor | Sequence[int]
type Scale = int | float | Tensor


class DDBatch(ABC):
    """A batch of continuous latent states with attached structure.

    Arithmetic acts only on dynamic state tensors. Structural tensors and
    metadata are preserved from `self`.
    """

    # ------------------------------------------------------------------
    # Batch / container
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def index_select(self, index: BatchIndex) -> Self:
        """Select complete batch elements, preserving a batch dimension."""
        ...

    def __getitem__(self, index: BatchIndex) -> Self:
        return self.index_select(index)

    @classmethod
    @abstractmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        """Concatenate complete batches."""
        ...

    # ------------------------------------------------------------------
    # Tensor access
    # ------------------------------------------------------------------

    @abstractmethod
    def all_tensors(self) -> Iterable[Tensor]:
        """All tensors, including structural tensors."""
        ...

    @abstractmethod
    def state_tensors(self) -> tuple[Tensor, ...]:
        """Floating-point tensors representing the continuous latent state."""
        ...

    @abstractmethod
    def replace_state_tensors(self, tensors: Sequence[Tensor]) -> Self:
        """Return the same structure with new dynamic state tensors."""
        ...

    @abstractmethod
    def map_all_tensors(self, op: UnaryOp) -> Self:
        """Apply an operation to all tensors (state and structural).

        Intended for device movement, cloning, and detaching only.
        """
        ...

    @abstractmethod
    def assert_compatible(self, other: Self) -> None:
        """Check that two states have identical batch structure."""
        ...

    # ------------------------------------------------------------------
    # Data-type-specific scalar broadcasting
    # ------------------------------------------------------------------

    @abstractmethod
    def scale(self, coefficient: Scale) -> Self:
        """Scale dynamic state tensors.

        A 1-D tensor of shape (batch_size,) is interpreted as one scalar
        per complete batch element and broadcast appropriately.
        """
        ...

    # ------------------------------------------------------------------
    # Generic state algebra (implemented in terms of abstract methods)
    # ------------------------------------------------------------------

    def map_state_tensors(self, op: UnaryOp) -> Self:
        return self.replace_state_tensors(tuple(op(x) for x in self.state_tensors()))

    def combine_state(self, other: Self, op: BinaryOp) -> Self:
        if type(other) is not type(self):
            raise TypeError(
                f"Cannot combine {type(self).__name__} and {type(other).__name__}."
            )
        self.assert_compatible(other)
        left = self.state_tensors()
        right = other.state_tensors()
        if len(left) != len(right):
            raise RuntimeError("Incompatible number of state tensors.")
        return self.replace_state_tensors(
            tuple(op(x, y) for x, y in zip(left, right, strict=True))
        )

    def __add__(self, other: Self) -> Self:
        return self.combine_state(other, torch.add)

    def __sub__(self, other: Self) -> Self:
        return self.combine_state(other, torch.sub)

    def __neg__(self) -> Self:
        return self.map_state_tensors(torch.neg)

    def __mul__(self, coefficient: Scale) -> Self:
        return self.scale(coefficient)

    def __rmul__(self, coefficient: Scale) -> Self:
        return self.scale(coefficient)

    def __truediv__(self, coefficient: Scale) -> Self:
        if isinstance(coefficient, Tensor):
            return self.scale(coefficient.reciprocal())
        if coefficient == 0:
            raise ZeroDivisionError("Cannot divide a latent state by zero.")
        return self.scale(1.0 / coefficient)

    def square(self) -> Self:
        return self.map_state_tensors(torch.square)

    # ------------------------------------------------------------------
    # Device and graph management
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        devices = {t.device for t in self.all_tensors()}
        if not devices:
            raise RuntimeError(f"No tensors found in {type(self).__name__}.")
        if len(devices) != 1:
            raise RuntimeError(
                f"Inconsistent devices in {type(self).__name__}: {devices}."
            )
        return next(iter(devices))

    def to(self, device: torch.device | str) -> Self:
        return self.map_all_tensors(lambda x: x.to(device))

    def cpu(self) -> Self:
        return self.to("cpu")

    def clone(self) -> Self:
        return self.map_all_tensors(torch.clone)

    def detach(self) -> Self:
        return self.map_all_tensors(torch.detach)

    # ------------------------------------------------------------------
    # Autograd over the continuous state only
    # ------------------------------------------------------------------

    def as_leaf(self, requires_grad: bool = True) -> Self:
        """Return a detached state whose dynamic tensors are autograd leaves."""
        return self.map_state_tensors(
            lambda x: x.detach().requires_grad_(requires_grad)
        )

    def gradient(
        self,
        outputs: Tensor,
        *,
        create_graph: bool = False,
        retain_graph: bool = False,
    ) -> Self:
        inputs = self.state_tensors()
        if not inputs:
            raise RuntimeError("The latent state has no dynamic tensors.")
        grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=(torch.ones_like(outputs) if outputs.ndim > 0 else None),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        completed = tuple(
            g if g is not None else torch.zeros_like(v)
            for v, g in zip(inputs, grads, strict=True)
        )
        return self.replace_state_tensors(completed)
