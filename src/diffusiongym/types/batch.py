from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Self

import torch

type UnaryOp = Callable[[torch.Tensor], torch.Tensor]
type BinaryOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class DDBatch(ABC):
    """Abstract data batch for common functionality."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> Self: ...

    @classmethod
    @abstractmethod
    def collate(cls, items: Sequence[Self]) -> Self: ...

    @abstractmethod
    def aggregate(self, reduction: str = "mean") -> torch.Tensor:
        """Reduce over all dimensions except batch (i.e., sum per sample).

        Parameters
        ----------
        reduction : str, default: "mean"
            Specifies the reduction to apply: "mean" or "sum".

        Returns
        -------
        Tensor of shape (len(self),)
        """
        ...

    @abstractmethod
    def apply(self, op: UnaryOp) -> Self:
        """Apply a function to the underlying tensor data (e.g., x -> -x)."""
        ...

    @abstractmethod
    def combine(self, other: Self, op: BinaryOp) -> Self:
        """Combine self with another instance element-wise (e.g., x + y).

        Note: `other` is guaranteed to be of the same type as `self` because the Mixin handles
        scalar broadcasting before calling this.
        """
        ...

    @property
    def device(self) -> torch.device:
        # Find the device of the underlying data and make sure all underlying data is on the same
        # device
        dev = None

        def get_tensor(x: torch.Tensor) -> torch.Tensor:
            nonlocal dev
            if dev is None:
                dev = x.device

            if dev != x.device:
                raise RuntimeError(f"Inconsistent devices found in {self.__class__}.")

            return x

        self.apply(get_tensor)

        if dev is None:
            raise RuntimeError(
                f"No tensors found in {self.__class__} to determine device."
            )

        return dev

    def to(self, device: torch.device | str) -> Self:
        return self.apply(lambda x: x.to(device))

    def cpu(self) -> Self:
        return self.apply(lambda x: x.cpu())

    def _binary_dispatch(
        self, other: Self | float | torch.Tensor, op: BinaryOp
    ) -> Self:
        if isinstance(other, torch.Tensor):
            return self.apply(lambda x: op(x, other))

        if isinstance(other, (int, float)):
            return self.apply(
                lambda x: op(x, torch.tensor(other, device=x.device, dtype=x.dtype))
            )

        if type(other) is self.__class__:
            return self.combine(other, op)

        raise TypeError(
            f"Unsupported operand type(s) for operation: {self.__class__} and {type(other)}"
        )

    def __add__(self, other) -> Self:
        return self._binary_dispatch(other, torch.add)

    def __sub__(self, other) -> Self:
        return self._binary_dispatch(other, torch.sub)

    def __mul__(self, other) -> Self:
        return self._binary_dispatch(other, torch.mul)

    def __truediv__(self, other) -> Self:
        return self._binary_dispatch(other, torch.div)

    def __neg__(self) -> Self:
        return self.apply(torch.neg)

    def __pow__(self, power) -> Self:
        return self.apply(lambda x: torch.pow(x, power))

    def __radd__(self, other) -> Self:
        return self._binary_dispatch(other, lambda x, y: torch.add(y, x))

    def __rsub__(self, other) -> Self:
        return self._binary_dispatch(other, lambda x, y: torch.sub(y, x))

    def __rmul__(self, other) -> Self:
        return self._binary_dispatch(other, lambda x, y: torch.mul(y, x))

    def square(self) -> Self:
        return self * self

    def randn_like(self) -> Self:
        def randn_like_to_float(x: torch.Tensor) -> torch.Tensor:
            if x.dtype.is_floating_point:
                return torch.randn_like(x)

            return x

        return self.apply(randn_like_to_float)

    def ones_like(self) -> Self:
        def ones_like_to_float(x: torch.Tensor) -> torch.Tensor:
            if x.dtype.is_floating_point:
                return torch.ones_like(x)

            return x

        return self.apply(ones_like_to_float)

    def zeros_like(self) -> Self:
        def zeros_like_to_float(x: torch.Tensor) -> torch.Tensor:
            if x.dtype.is_floating_point:
                return torch.zeros_like(x)

            return x

        return self.apply(zeros_like_to_float)

    def clone(self) -> Self:
        return self.apply(torch.clone)

    def requires_grad(self, mode: bool = True) -> Self:
        """In-place operation to start recording operations for autograd."""
        return self.apply(lambda x: x.requires_grad_(mode))

    def detach(self) -> Self:
        """Return a new instance detached from the current computation graph."""
        return self.apply(torch.detach)

    def gradient(
        self,
        outputs: torch.Tensor,
        create_graph: bool = False,
        retain_graph: bool = False,
    ) -> Self:
        """Compute the gradient of output w.r.t. self.

        Returns: An instance of Self containing the gradients.
        """
        # Gather inputs
        inputs = []

        def gather_inputs(x: torch.Tensor) -> torch.Tensor:
            inputs.append(x)
            return x

        self.apply(gather_inputs)

        # Compute gradients
        grad_outputs = torch.ones_like(outputs) if outputs.ndim > 0 else None
        raw_grads = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )

        # Inject gradients (this is in the same order as we gathered the inputs)
        grad_iter = iter(raw_grads)

        def inject_grads(x: torch.Tensor) -> torch.Tensor:
            g = next(grad_iter)
            return g if g is not None else torch.zeros_like(x)

        return self.apply(inject_grads)
