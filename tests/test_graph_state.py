"""Conformance test: the framework on a ragged, graph-structured state.

`DDTensor` is dense and fixed-size, so it exercises none of the assumptions a
graph state would break. This module defines `SegmentBatch` — variable node
counts per sample, node features concatenated along dim 0, plus a structural
integer tensor — which is exactly the layout of a PyTorch Geometric `Batch`
(`x` / `edge_index` / `batch`). Everything here maps 1:1 onto a real PyG
subclass:

    SegmentBatch.state_tensors  ->  the continuous fields being generated
                                    (`data.pos`, `data.x`, ...)
    SegmentBatch.all_tensors    ->  those plus `edge_index`, `batch`, masks
    SegmentBatch.scale          ->  per-graph scalar broadcast to nodes
    SegmentGeometry.squared_norm->  segment reduction over nodes, per graph

If a change to `core/` or `trainers/` starts assuming dense fixed-size states,
these tests fail while the `DDTensor` ones keep passing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self

import pytest
import torch
from torch import Generator, Tensor

from diffusiongym.core import (
    AffineGaussianForwardProcess,
    DefaultEulerGaussianKernelFactory,
    EulerMaruyamaSampler,
    EulerODESampler,
    FlowEnvironment,
    IdentityCodec,
    MemorylessFlowSDE,
    PolicyBundle,
    PredictionConverter,
    PredictionKind,
    ProbabilityFlowODE,
    RectifiedFlowSchedule,
    RolloutRequest,
    RolloutStorage,
    VelocityRegression,
)
from diffusiongym.core.reward import RewardBatch
from diffusiongym.trainers import (
    ORWCFM,
    AdjointMatching,
    DiffusionNFT,
    FineTuningContext,
    FlowGRPO,
)
from diffusiongym.types import DDBatch

FEATURES = 2


# ---------------------------------------------------------------------------
# A ragged, graph-shaped state
# ---------------------------------------------------------------------------


class SegmentBatch(DDBatch):
    """Node features for a batch of variable-size graphs.

    nodes:  (total_nodes, FEATURES) continuous state — the only *dynamic* tensor
    sizes:  (num_graphs,) node count per graph — structural, integer
    degree: (total_nodes,) a stand-in for an integer structural field such as
            `edge_index`; it must survive device moves and cloning untouched and
            must never receive gradients or arithmetic.
    """

    def __init__(self, nodes: Tensor, sizes: Tensor, degree: Tensor) -> None:
        self.nodes = nodes
        self.sizes = sizes
        self.degree = degree

    # -- container ---------------------------------------------------------

    def __len__(self) -> int:
        return int(self.sizes.shape[0])

    @property
    def _ptr(self) -> Tensor:
        return torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.sizes.device),
             self.sizes.cumsum(0)]
        )

    def index_select(self, index) -> Self:
        if isinstance(index, int):
            index = [index]
        if isinstance(index, slice):
            index = list(range(*index.indices(len(self))))
        if isinstance(index, Tensor):
            index = index.tolist()
        ptr = self._ptr
        rows = torch.cat(
            [torch.arange(ptr[i], ptr[i + 1], device=self.nodes.device)
             for i in index]
        ) if index else torch.zeros(0, dtype=torch.long, device=self.nodes.device)
        picked = torch.tensor(index, dtype=torch.long, device=self.sizes.device)
        return type(self)(self.nodes[rows], self.sizes[picked], self.degree[rows])

    @classmethod
    def concat(cls, batches: Sequence[Self]) -> Self:  # ty: ignore[invalid-method-override]
        return cls(
            torch.cat([b.nodes for b in batches], dim=0),
            torch.cat([b.sizes for b in batches], dim=0),
            torch.cat([b.degree for b in batches], dim=0),
        )

    # -- tensor access -----------------------------------------------------

    def all_tensors(self) -> Iterable[Tensor]:
        return (self.nodes, self.sizes, self.degree)

    def state_tensors(self) -> tuple[Tensor, ...]:
        return (self.nodes,)

    def replace_state_tensors(self, tensors: Sequence[Tensor]) -> Self:
        (nodes,) = tensors
        return type(self)(nodes, self.sizes, self.degree)

    def map_all_tensors(self, op) -> Self:
        return type(self)(op(self.nodes), op(self.sizes), op(self.degree))

    def assert_compatible(self, other: Self) -> None:  # ty: ignore[invalid-method-override]
        if not torch.equal(self.sizes, other.sizes):
            raise ValueError("SegmentBatch states have different graph sizes.")

    # -- per-element scalar broadcast --------------------------------------

    def scale(self, coefficient) -> Self:
        """Broadcast a per-*graph* scalar onto nodes.

        This is the method that separates a graph state from a dense one: index
        0 of the state tensor is a node, not a batch element, so a coefficient
        of shape (num_graphs,) has to be expanded by the node counts.
        """
        if isinstance(coefficient, Tensor):
            coefficient = coefficient.to(
                device=self.nodes.device, dtype=self.nodes.dtype
            )
            if coefficient.ndim == 0:
                pass
            elif coefficient.shape == (len(self),):
                coefficient = coefficient.repeat_interleave(self.sizes).unsqueeze(-1)
            else:
                raise ValueError(
                    f"A SegmentBatch scale must be scalar or shape ({len(self)},), "
                    f"got {tuple(coefficient.shape)}."
                )
        return type(self)(self.nodes * coefficient, self.sizes, self.degree)


class SegmentGeometry:
    """LatentGeometry for SegmentBatch — reductions are per graph, not per row."""

    def project(self, x: SegmentBatch) -> SegmentBatch:
        return x

    def standard_normal_like(
        self, x: SegmentBatch, *, generator: Generator | None = None
    ) -> SegmentBatch:
        noise = torch.randn(
            x.nodes.shape, dtype=x.nodes.dtype, device=x.nodes.device,
            generator=generator,
        )
        return SegmentBatch(noise, x.sizes, x.degree)

    def active_dimensions(self, x: SegmentBatch) -> Tensor:
        return x.sizes * FEATURES

    def squared_norm(self, x: SegmentBatch, *, reduction: str) -> Tensor:
        per_node = x.nodes.square().sum(-1)
        graph_id = torch.arange(len(x), device=x.nodes.device).repeat_interleave(
            x.sizes
        )
        sums = torch.zeros(len(x), dtype=per_node.dtype, device=per_node.device)
        sums = sums.index_add_(0, graph_id, per_node)
        match reduction:
            case "sum":
                return sums
            case "mean":
                return sums / self.active_dimensions(x).to(sums.dtype)
            case _:
                raise ValueError(f"Unsupported reduction: {reduction!r}.")


class SegmentBaseSampler:
    """N(0, I) over nodes, with graph sizes drawn per sample."""

    def __init__(self, sizes: Sequence[int] = (3, 5, 4, 2)) -> None:
        self.sizes = list(sizes)

    def _structure(self, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
        sizes = torch.tensor(
            [self.sizes[i % len(self.sizes)] for i in range(n)], device=device
        )
        degree = torch.arange(int(sizes.sum()), device=device) % 3
        return sizes, degree

    def sample(self, n, *, conditioning, device, generator=None):
        sizes, degree = self._structure(n, device)
        nodes = torch.randn(
            int(sizes.sum()), FEATURES, device=device, generator=generator
        )
        return SegmentBatch(nodes, sizes, degree), conditioning

    def sample_like(self, x_data: SegmentBatch, *, generator=None) -> SegmentBatch:
        noise = torch.randn(
            x_data.nodes.shape, device=x_data.nodes.device, generator=generator
        )
        return SegmentBatch(noise, x_data.sizes, x_data.degree)


class SegmentFlowModel:
    """Per-node velocity MLP; time is broadcast from graphs onto nodes."""

    prediction_kind = PredictionKind.VELOCITY

    def __init__(self, device: torch.device) -> None:
        self._net = torch.nn.Sequential(
            torch.nn.Linear(FEATURES + 1, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, FEATURES),
        )
        torch.nn.init.zeros_(self._net[-1].weight)
        torch.nn.init.zeros_(self._net[-1].bias)
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def parameters(self):
        return self._net.parameters()

    def state_dict(self):
        return self._net.state_dict()

    def load_state_dict(self, sd):
        return self._net.load_state_dict(sd)

    def __call__(self, x_t: SegmentBatch, t: Tensor, *, conditioning) -> SegmentBatch:
        t_nodes = t.repeat_interleave(x_t.sizes).unsqueeze(-1)
        out = self._net(torch.cat([x_t.nodes, t_nodes], dim=-1))
        return SegmentBatch(out, x_t.sizes, x_t.degree)


class SegmentReward:
    """Mean of feature 0 over each graph's nodes — one scalar per graph."""

    @staticmethod
    def _value(x: SegmentBatch) -> Tensor:
        graph_id = torch.arange(len(x), device=x.nodes.device).repeat_interleave(
            x.sizes
        )
        totals = torch.zeros(len(x), device=x.nodes.device).index_add_(
            0, graph_id, x.nodes[:, 0]
        )
        return totals / x.sizes.to(totals.dtype)

    def __call__(self, *, sample, latent, conditioning) -> RewardBatch:
        return RewardBatch(rewards=self._value(sample))


class SegmentCost:
    """Differentiable terminal cost g(x) = -r(x), for Adjoint Matching."""

    def __call__(self, terminal_latent: SegmentBatch, *, conditioning) -> Tensor:
        return -SegmentReward._value(terminal_latent)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def geometry() -> SegmentGeometry:
    return SegmentGeometry()


@pytest.fixture
def schedule() -> RectifiedFlowSchedule:
    return RectifiedFlowSchedule()


@pytest.fixture
def graph_env(geometry, schedule) -> FlowEnvironment:
    base_sampler = SegmentBaseSampler()
    converter = PredictionConverter(geometry=geometry, schedule=schedule)
    return FlowEnvironment(
        geometry=geometry,
        base_sampler=base_sampler,
        forward_process=AffineGaussianForwardProcess(
            geometry=geometry, base_sampler=base_sampler, schedule=schedule
        ),
        regression=VelocityRegression(geometry=geometry, converter=converter),
        codec=IdentityCodec(),
        reward=SegmentReward(),
        terminal_cost=SegmentCost(),
    )


@pytest.fixture
def graph_context(graph_env, geometry) -> FineTuningContext:
    device = torch.device("cpu")
    train = SegmentFlowModel(device)
    rollout = SegmentFlowModel(device)
    rollout.load_state_dict(train.state_dict())
    reference = SegmentFlowModel(device)
    reference.load_state_dict(train.state_dict())
    return FineTuningContext(
        environment=graph_env,
        policies=PolicyBundle(train=train, rollout=rollout, reference=reference),
        optimizer=torch.optim.Adam(train.parameters(), lr=1e-3),
        ode_sampler=EulerODESampler(geometry),
        sde_sampler=EulerMaruyamaSampler(
            geometry, DefaultEulerGaussianKernelFactory(geometry)
        ),
    )


# ---------------------------------------------------------------------------
# The state itself
# ---------------------------------------------------------------------------


class TestSegmentBatch:
    @staticmethod
    def _make(n: int = 4) -> SegmentBatch:
        sampler = SegmentBaseSampler()
        batch, _ = sampler.sample(n, conditioning={}, device=torch.device("cpu"))
        return batch

    def test_length_counts_graphs_not_nodes(self):
        batch = self._make(4)
        assert len(batch) == 4
        assert batch.nodes.shape[0] == 3 + 5 + 4 + 2  # ragged on purpose

    def test_per_graph_scalar_broadcasts_over_nodes(self):
        batch = self._make(4)
        scaled = batch * torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat_interleave(batch.sizes)
        assert torch.allclose(scaled.nodes, batch.nodes * expected.unsqueeze(-1))

    def test_index_select_keeps_whole_graphs(self):
        batch = self._make(4)
        picked = batch[torch.tensor([0, 2])]
        assert len(picked) == 2
        assert picked.nodes.shape[0] == 3 + 4
        assert torch.equal(picked.sizes, torch.tensor([3, 4]))

    def test_concat_roundtrips(self):
        batch = self._make(4)
        rebuilt = SegmentBatch.concat([batch[torch.tensor([0, 1])],
                                       batch[torch.tensor([2, 3])]])
        assert torch.allclose(rebuilt.nodes, batch.nodes)
        assert torch.equal(rebuilt.sizes, batch.sizes)

    def test_arithmetic_leaves_structure_untouched(self):
        batch = self._make(4)
        combined = (batch + batch) - batch
        assert torch.allclose(combined.nodes, batch.nodes)
        # Structural fields must pass through unchanged, not be added together.
        assert torch.equal(combined.degree, batch.degree)
        assert torch.equal(combined.sizes, batch.sizes)

    def test_gradient_flows_only_through_state_tensors(self):
        batch = self._make(3).as_leaf(True)
        loss = batch.nodes.square().sum()
        grad = batch.gradient(loss)
        assert torch.allclose(grad.nodes, 2.0 * batch.nodes)
        assert not batch.degree.requires_grad

    def test_mismatched_structure_is_rejected(self):
        """Arithmetic between differently-shaped graphs must raise, not broadcast.

        This is the one failure mode a graph state has and a dense one does not:
        two batches can carry the same *total* node count while describing
        different graphs, in which case an elementwise add is silently wrong.
        """
        four = self._make(4)
        two = self._make(2)
        with pytest.raises(ValueError, match="different graph sizes"):
            _ = four + two

    def test_squared_norm_reduces_per_graph(self, geometry):
        batch = self._make(4)
        summed = geometry.squared_norm(batch, reduction="sum")
        assert summed.shape == (4,)
        assert summed[0].item() == pytest.approx(
            batch.nodes[:3].square().sum().item(), rel=1e-5
        )
        averaged = geometry.squared_norm(batch, reduction="mean")
        assert torch.allclose(averaged, summed / (batch.sizes * FEATURES))


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


class TestGraphPipeline:
    def test_forward_process_interpolates(self, graph_env):
        sampler = SegmentBaseSampler()
        x_data, _ = sampler.sample(4, conditioning={}, device=torch.device("cpu"))
        batch = graph_env.make_forward_batch(x_data, conditioning={})
        assert len(batch.x_t) == 4
        assert batch.x_t.nodes.shape == x_data.nodes.shape
        assert batch.t.shape == (4,)

    def test_ode_rollout(self, graph_env, geometry):
        sampler = EulerODESampler(geometry)
        rollout = sampler.rollout(
            environment=graph_env,
            model=SegmentFlowModel(torch.device("cpu")),
            dynamics=ProbabilityFlowODE(),
            n=4,
            conditioning={},
            request=RolloutRequest(time_grid=torch.linspace(0, 1, 4)),
        )
        assert len(rollout.terminal_latent) == 4
        assert rollout.reward is not None
        assert rollout.reward.rewards.shape == (4,)

    def test_sde_rollout_log_prob_is_per_graph(self, graph_env, geometry, schedule):
        sampler = EulerMaruyamaSampler(
            geometry, DefaultEulerGaussianKernelFactory(geometry)
        )
        rollout = sampler.rollout(
            environment=graph_env,
            model=SegmentFlowModel(torch.device("cpu")),
            dynamics=MemorylessFlowSDE(affine_schedule=schedule),
            n=4,
            conditioning={},
            request=RolloutRequest(
                time_grid=torch.linspace(0, 1, 5)[1:],
                storage=RolloutStorage(states=True, log_probs=True),
            ),
        )
        for step in rollout.steps:
            # One log-probability per graph, and graphs with more nodes are
            # higher-dimensional Gaussians — the kernel must use per-sample
            # active_dimensions rather than a single global d.
            assert step.log_prob is not None
            assert step.log_prob.shape == (4,)
            assert torch.isfinite(step.log_prob).all()


class TestGraphTrainers:
    """Every algorithm must complete collect() + update() on a ragged state."""

    def test_orwcfm(self, graph_context, schedule):
        algo = ORWCFM(temperature=1.0, steps_per_update=2, batch_size=4)
        metrics = _run(algo, graph_context, ProbabilityFlowODE(),
                       torch.linspace(0, 1, 4))
        assert "r_mean" in metrics

    def test_diffusion_nft(self, graph_context, schedule):
        algo = DiffusionNFT(beta=1.0, inner_epochs=2, batch_size=4)
        metrics = _run(algo, graph_context, ProbabilityFlowODE(),
                       torch.linspace(0, 1, 4))
        assert "r_mean" in metrics

    def test_flow_grpo(self, graph_context, schedule):
        algo = FlowGRPO(group_size=2, ppo_epochs=1, ppo_batch_size=8, beta_kl=1.0)
        metrics = _run(algo, graph_context,
                       MemorylessFlowSDE(affine_schedule=schedule),
                       torch.linspace(0, 1, 5)[1:])
        assert "kl" in metrics

    def test_adjoint_matching(self, graph_context, schedule):
        algo = AdjointMatching(train_steps_per_iter=2, train_batch_size=4)
        metrics = _run(algo, graph_context,
                       MemorylessFlowSDE(affine_schedule=schedule),
                       torch.linspace(0, 1, 5)[1:])
        assert "r_mean" in metrics


def _run(algo, context, dynamics, time_grid):
    algo.validate(context=context, dynamics=dynamics)
    experience = algo.collect(
        context=context, dynamics=dynamics, n=4,
        time_grid=time_grid, conditioning={},
    )
    metrics = algo.update(context=context, experience=experience)
    algo.synchronize_rollout_policy(context=context)
    return metrics
