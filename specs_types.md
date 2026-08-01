I agree with **merging the basic vector-space algebra into the latent batch type**, but not with putting every latent-space concept into that class.

The most useful split is:

1. **`DDBatch`**: data representation, batching, device movement, indexing, and linear algebra on the modeled continuous state.
2. **`LatentGeometry[D]`**: projection, Gaussian noise, norms, and effective dimensionality.

This keeps common formulas readable:

```python
x_t = a_t * x_base + b_t * x_data
residual = predicted_velocity - target_velocity
```

while avoiding the assumption that every tensor stored in a graph is part of the continuous flow state.

That distinction is essential for PyG. A molecular batch contains tensors such as:

* coordinates and continuous atom/bond features: **dynamic**
* `edge_index`, `batch`, `ptr`, masks and graph sizes: **structural**

Your current `apply()` and `combine()` do not distinguish these. For example, adding two graph batches would also add their `edge_index` tensors. Dividing a batch would divide graph indices. `requires_grad()` would attempt to enable gradients on integer tensors. Those are serious correctness problems.

## Recommended division of responsibilities

| Operation                              | `DDBatch` | `LatentGeometry` |
| -------------------------------------- | --------: | ---------------: |
| Batch length                           |         ✓ |                  |
| Device movement                        |         ✓ |                  |
| Batch indexing                         |         ✓ |                  |
| Batch concatenation                    |         ✓ |                  |
| Addition/subtraction                   |         ✓ |                  |
| Multiplication by (t)-dependent scalar |         ✓ |                  |
| Clone/detach                           |         ✓ |                  |
| State-tensor gradients                 |         ✓ |                  |
| Projection to zero center of mass      |           |                ✓ |
| Standard Gaussian in constrained space |           |                ✓ |
| Squared norm                           |           |                ✓ |
| Active dimensions                      |           |                ✓ |
| Per-field metric weights               |           |                ✓ |
| Gaussian transition geometry           |           |                ✓ |

`project`, `active_dimensions`, `squared_norm`, and constrained `randn_like` should not be methods of the raw data container. The same PyG batch representation may be used with:

* unconstrained coordinates,
* zero-center-of-mass coordinates,
* different coordinate/feature loss weights,
* training mean-square geometry,
* Gaussian transition sum-square geometry.

Those are properties of the modeled latent distribution, not just the Python data type.

# Problems in the current abstraction

## 1. `apply()` is too broad

Currently, all tensor attributes receive the operation:

```python
return self.apply(lambda x: op(x, other))
```

That is acceptable for:

* `.to(device)`
* perhaps `.clone()`
* perhaps `.detach()`

It is not acceptable for:

* addition,
* subtraction,
* division,
* Gaussian noise,
* gradients,
* interpolation.

For a graph batch, arithmetic must affect only dynamic state fields.

You need two concepts:

```python
map_all_tensors(...)
map_state_tensors(...)
```

## 2. Batch-scalar broadcasting is data-type-specific

A schedule coefficient normally has shape `(batch_size,)`.

For an image tensor of shape `(B, C, H, W)`, it must become:

```python
coefficient.reshape(B, 1, 1, 1)
```

For a graph node field, a coefficient of shape `(B,)` must be expanded using the node-to-graph assignment:

```python
coefficient[data.batch]
```

For edge fields it must use the edge-to-graph assignment.

Your current implementation:

```python
torch.mul(x, other)
```

will not generally broadcast a `(B,)` tensor correctly over `(B, C, H, W)`, and cannot broadcast it correctly over graph nodes.

Therefore, scalar multiplication should be a specialized abstract operation.

## 3. `randn_like()` does not define the desired base distribution

For FlowMol, Gaussian coordinate noise may need:

* zero-center-of-mass projection,
* graph masks,
* possibly different treatment of node, edge, and coordinate fields.

A generic `torch.randn_like()` only produces ambient Gaussian noise. It does not necessarily sample the Gaussian distribution on the modeled subspace.

Keep an internal raw random operation if useful, but expose standard Gaussian sampling through the geometry.

## 4. `aggregate()` conflates different semantics

These are not the same operation:

```python
residual.square().aggregate("mean")
residual.square().aggregate("sum")
```

The first may be a training MSE. The second may be a Gaussian quadratic term.

For variable-size molecules, a mean must divide by each molecule's active dimensions, not by the total storage size or padded tensor size.

Replace generic `aggregate()` in algorithm-facing code with:

```python
geometry.squared_norm(x, reduction="mean")
geometry.squared_norm(x, reduction="sum")
```

## 5. `requires_grad()` and `gradient()` currently include integer metadata

This will fail or behave incorrectly for PyG tensors such as `edge_index`, `batch`, and `ptr`.

Autograd methods must operate only on floating dynamic state tensors.

## 6. `device` reconstructs the object unnecessarily

This:

```python
self.apply(get_tensor)
```

creates a new wrapper merely to inspect tensors. Expose an iterator over tensors instead.

## 7. `collate` is ambiguous

For a batch abstraction, `concat` is clearer than `collate`:

```python
DDTensor.concat([batch1, batch2])
DDGraphBatch.concat([batch1, batch2])
```

Dataset-level collation from individual `Data` objects can remain a PyG concern.

# Revised `DDBatch`

I would use the following interface.

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Self, TypeAlias

import torch
from torch import Tensor


UnaryOp: TypeAlias = Callable[[Tensor], Tensor]
BinaryOp: TypeAlias = Callable[[Tensor, Tensor], Tensor]
BatchIndex: TypeAlias = int | slice | Tensor | Sequence[int]
Scale: TypeAlias = int | float | Tensor


class DDBatch(ABC):
    """A batch of continuous latent states with attached structure.

    Arithmetic acts only on dynamic state tensors. Structural tensors and
    metadata are preserved from `self`.
    """

    # ------------------------------------------------------------------
    # Batch/container functionality
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int:
        ...

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
        """Floating tensors representing the continuous latent state."""
        ...

    @abstractmethod
    def replace_state_tensors(
        self,
        tensors: Sequence[Tensor],
    ) -> Self:
        """Return the same structure with new dynamic state tensors."""
        ...

    @abstractmethod
    def map_all_tensors(self, op: UnaryOp) -> Self:
        """Apply an operation to state and structural tensors.

        Intended for device movement, cloning, and detaching.
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

        A one-dimensional tensor of shape `(batch_size,)` is interpreted as
        one scalar per complete batch element.
        """
        ...

    # ------------------------------------------------------------------
    # Generic state algebra
    # ------------------------------------------------------------------

    def map_state_tensors(self, op: UnaryOp) -> Self:
        return self.replace_state_tensors(
            tuple(op(x) for x in self.state_tensors())
        )

    def combine_state(
        self,
        other: Self,
        op: BinaryOp,
    ) -> Self:
        if type(other) is not type(self):
            raise TypeError(
                f"Cannot combine {type(self).__name__} and "
                f"{type(other).__name__}."
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
        devices = {tensor.device for tensor in self.all_tensors()}

        if not devices:
            raise RuntimeError(
                f"No tensors found in {type(self).__name__}."
            )

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
            grad_outputs=(
                torch.ones_like(outputs) if outputs.ndim > 0 else None
            ),
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=True,
        )

        completed = tuple(
            grad if grad is not None else torch.zeros_like(value)
            for value, grad in zip(inputs, grads, strict=True)
        )

        return self.replace_state_tensors(completed)
```

This is narrower than your current class in useful ways:

* Adding two batches is supported.
* Scaling a state is supported.
* Arbitrary scalar addition is not supported because it has no clear flow-model interpretation.
* Elementwise multiplication of two complete states is not overloaded.
* Structural tensors are never modified by state algebra.
* Per-batch scalar broadcasting is delegated to the concrete data type.

# Revised `DDTensor`

```python
class DDTensor(DDBatch):
    """A batched dense continuous tensor state."""

    def __init__(self, data: Tensor):
        if not isinstance(data, Tensor):
            raise TypeError("DDTensor expects a torch.Tensor.")

        if data.ndim < 1:
            raise ValueError(
                "DDTensor expects a tensor with a batch dimension."
            )

        if not data.dtype.is_floating_point:
            raise TypeError(
                "DDTensor's state tensor must use a floating-point dtype."
            )

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

        # Tensor indices may remove the batch dimension in some cases.
        if selected.ndim == self.data.ndim - 1:
            selected = selected.unsqueeze(0)

        return type(self)(selected)

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:
        if not batches:
            raise ValueError("Cannot concatenate an empty sequence.")

        return cls(torch.cat([batch.data for batch in batches], dim=0))

    def all_tensors(self) -> tuple[Tensor]:
        return (self.data,)

    def state_tensors(self) -> tuple[Tensor]:
        return (self.data,)

    def replace_state_tensors(
        self,
        tensors: Sequence[Tensor],
    ) -> Self:
        if len(tensors) != 1:
            raise ValueError(
                f"DDTensor expects one state tensor, got {len(tensors)}."
            )

        return type(self)(tensors[0])

    def map_all_tensors(self, op: UnaryOp) -> Self:
        return type(self)(op(self.data))

    def assert_compatible(self, other: Self) -> None:
        if self.data.shape != other.data.shape:
            raise ValueError(
                "Tensor states have incompatible shapes: "
                f"{tuple(self.data.shape)} versus "
                f"{tuple(other.data.shape)}."
            )

    def scale(self, coefficient: Scale) -> Self:
        if isinstance(coefficient, Tensor):
            coefficient = coefficient.to(
                device=self.data.device,
                dtype=self.data.dtype,
            )

            if coefficient.ndim == 0:
                pass
            elif coefficient.shape == (len(self),):
                coefficient = coefficient.reshape(
                    len(self),
                    *([1] * (self.data.ndim - 1)),
                )
            else:
                raise ValueError(
                    "A DDTensor scale must be scalar or have shape "
                    f"({len(self)},), got {tuple(coefficient.shape)}."
                )

        return type(self)(self.data * coefficient)
```

The explicit scale handling solves the schedule-broadcasting problem.

# Geometry interface

```python
from typing import Generic, Literal, Protocol, TypeVar


D = TypeVar("D", bound=DDBatch)
NormReduction = Literal["mean", "sum"]


class LatentGeometry(Protocol[D]):
    """Geometry and Gaussian measure on a DDBatch representation."""

    def project(self, x: D) -> D:
        """Project onto the modeled linear subspace."""
        ...

    def standard_normal_like(
        self,
        x: D,
        *,
        generator: torch.Generator | None = None,
    ) -> D:
        """Sample N(0, I) in the modeled latent subspace."""
        ...

    def squared_norm(
        self,
        x: D,
        *,
        reduction: NormReduction,
    ) -> Tensor:
        """Return one value per complete batch element."""
        ...

    def active_dimensions(self, x: D) -> Tensor:
        """Effective continuous dimensionality of each batch element."""
        ...
```

## Dense tensor geometry

```python
class TensorGeometry(LatentGeometry[DDTensor]):
    def project(self, x: DDTensor) -> DDTensor:
        return x

    def standard_normal_like(
        self,
        x: DDTensor,
        *,
        generator: torch.Generator | None = None,
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

    def squared_norm(
        self,
        x: DDTensor,
        *,
        reduction: NormReduction,
    ) -> Tensor:
        sums = x.data.square().flatten(start_dim=1).sum(dim=1)

        match reduction:
            case "sum":
                return sums
            case "mean":
                return sums / self.active_dimensions(x).to(sums.dtype)
            case _:
                raise ValueError(f"Unsupported reduction: {reduction}.")
```

For a molecular graph geometry, these operations can use efficient PyG scatter operations:

```python
node_sums = scatter(
    node_values.square().sum(dim=-1),
    graph_batch,
    dim=0,
    reduce="sum",
)
```

Projection can similarly use per-graph means:

```python
center = scatter(
    positions,
    graph_batch,
    dim=0,
    reduce="mean",
)
centered = positions - center[graph_batch]
```

You therefore retain PyG's efficient kernels. The common abstraction does not require Python loops over graphs.

# PyG representation

For algorithms, wrap `torch_geometric.data.Batch`, not individual `Data` objects.

Conceptually:

```python
class DDGraphBatch(DDBatch):
    def __init__(
        self,
        data: torch_geometric.data.Batch,
        *,
        state_keys: tuple[str, ...],
    ):
        self.data = data
        self.state_keys = state_keys
```

`state_keys` might be:

```python
(
    "pos",
    "atom_features",
    "charges",
    "edge_features",
)
```

Structural attributes such as these are not part of the state algebra:

```python
(
    "edge_index",
    "batch",
    "ptr",
    "num_nodes",
)
```

Its implementation should delegate to PyG:

* `.to()` for device movement
* `.clone()` for cloning
* `Batch.from_data_list()` for constructing batches
* PyG graph-selection utilities for indexing
* `scatter` for per-graph reductions
* the existing `batch` vector for coefficient expansion

The important requirement is that `replace_state_tensors()` replaces only `state_keys` and preserves graph topology and batching metadata.

## Graph scaling

A graph batch needs to know whether each state field is:

* node-level,
* edge-level,
* graph-level.

A simple explicit field specification is sufficient:

```python
from dataclasses import dataclass
from typing import Literal


FieldLevel = Literal["node", "edge", "graph"]


@dataclass(frozen=True)
class GraphStateField:
    key: str
    level: FieldLevel
```

Then a batch coefficient (c\in\mathbb R^B) expands as:

```python
# Node field
expanded = c[data.batch]

# Edge field
edge_batch = data.batch[data.edge_index[0]]
expanded = c[edge_batch]

# Graph field
expanded = c
```

This is more reliable than attempting generic PyTorch broadcasting.

# Efficiency implications

The wrapper itself is not a meaningful performance problem.

An operation such as:

```python
x + y
```

still invokes one native `torch.add` per state field. It does not loop over tensor elements in Python. The dominant costs remain:

* neural-network forward and backward passes,
* graph message passing,
* sampling steps,
* reward evaluation.

The costly mistakes would instead be:

* converting PyG batches to lists of graphs inside every flow step,
* repeatedly unbatching and re-batching,
* transferring every trajectory step to CPU,
* cloning structural data unnecessarily,
* iterating over individual graphs for reductions.

A `DDGraphBatch` implementation should retain a native PyG `Batch` throughout a rollout and use vectorized PyG/scatter operations.

# Recommended decision

Use the following conceptual ownership:

```text
DDBatch
├── native storage
├── complete-batch indexing
├── batch concatenation
├── device/clone/detach
├── dynamic-state tensor access
├── state addition and subtraction
├── data-type-aware scalar broadcasting
└── state-only autograd

LatentGeometry[DDBatch]
├── projection
├── constrained Gaussian sampling
├── squared norm
├── effective dimensions
└── field weighting
```

So, **yes to merging basic algebra with the data type**, because broadcasting and structural preservation are inherently representation-specific.

But **no to merging the complete latent geometry with it**. In particular:

* `index_select()` belongs on `DDBatch`.
* `project()` belongs on `LatentGeometry`.
* `active_dimensions()` belongs on `LatentGeometry`.
* `squared_norm()` belongs on `LatentGeometry`.
* constrained `randn_like()` belongs on `LatentGeometry`.
* raw tensor traversal must distinguish all tensors from dynamic state tensors.

This is the smallest abstraction that remains correct for both dense image tensors and structured PyG molecular batches.
