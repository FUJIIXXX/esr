"""Synthetic hypergraph snapshots for mobile-edge task scheduling.

The generator intentionally keeps all domain quantities explicit instead of only
sampling anonymous node features. This makes experiments auditable: every GNN
input can be traced back to latency, throughput, energy, battery, compute and
minimum-task requirements from the proposal.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np
import torch

from esr.config import SchedulerConfig, SystemConfig


class NodeType(IntEnum):
    EDGE = 0
    MOBILE = 1
    TASK = 2


@dataclass(frozen=True)
class TaskRequirement:
    """Minimum resource demand for one task before splitting."""

    task_node: int
    data_mb: float
    cycles_g: float
    min_compute: float
    min_battery: float
    priority: int


@dataclass(frozen=True)
class DeviceState:
    """Observable state reported by one mobile device."""

    node_id: int
    edge_id: int
    compute: float
    battery: float
    bandwidth: float
    signal: float
    queue_load: float


@dataclass(frozen=True)
class HyperedgeRecord:
    """Typed hyperedge metadata retained next to the DHG structure."""

    vertices: list[int]
    weight: float
    kind: str
    task_node: int | None = None
    edge_node: int | None = None


@dataclass(frozen=True)
class MecaSnapshot:
    """A complete hypergraph scheduling observation."""

    node_features: torch.Tensor
    node_types: torch.Tensor
    devices: list[DeviceState]
    tasks: list[TaskRequirement]
    hyperedges: list[HyperedgeRecord]
    n_edges: int
    n_mobiles: int

    @property
    def n_nodes(self) -> int:
        return int(self.node_features.shape[0])

    def to_dhg_hypergraph(self):
        """Build a weighted ``dhg.Hypergraph`` from the stored hyperedges.

        DHG's ``Hypergraph`` constructor accepts ``num_v``, a list of hyperedges
        and optional hyperedge weights, matching the current DHG documentation.
        The import is local so pure scheduling tests can run without DHG.
        """

        import dhg

        return dhg.Hypergraph(
            self.n_nodes,
            [edge.vertices for edge in self.hyperedges],
            e_weight=[edge.weight for edge in self.hyperedges],
        )

    def task_node_ids(self) -> list[int]:
        return [task.task_node for task in self.tasks]

    def mobile_node_ids(self) -> list[int]:
        return [device.node_id for device in self.devices]


class MobileEdgeHypergraphBuilder:
    """Generate reproducible multi-edge MEC hypergraph snapshots."""

    def __init__(self, system: SystemConfig, scheduler: SchedulerConfig):
        self.system = system
        self.scheduler = scheduler
        self.rng = np.random.default_rng(system.seed)

    def build(self) -> MecaSnapshot:
        n_nodes = self.system.n_edges + self.system.n_mobiles + self.system.n_tasks
        node_types = torch.full((n_nodes,), NodeType.TASK, dtype=torch.long)
        node_types[: self.system.n_edges] = NodeType.EDGE
        node_types[self.system.n_edges : self.system.n_edges + self.system.n_mobiles] = NodeType.MOBILE

        devices = self._sample_devices()
        tasks = self._sample_tasks()
        features = self._build_features(n_nodes, devices, tasks)
        hyperedges = self._build_hyperedges(devices, tasks)

        return MecaSnapshot(
            node_features=features,
            node_types=node_types,
            devices=devices,
            tasks=tasks,
            hyperedges=hyperedges,
            n_edges=self.system.n_edges,
            n_mobiles=self.system.n_mobiles,
        )

    def _sample_devices(self) -> list[DeviceState]:
        devices: list[DeviceState] = []
        for idx in range(self.system.n_mobiles):
            node_id = self.system.n_edges + idx
            edge_id = idx % self.system.n_edges
            devices.append(
                DeviceState(
                    node_id=node_id,
                    edge_id=edge_id,
                    compute=float(self.rng.uniform(0.25, 1.0)),
                    battery=float(self.rng.uniform(0.15, 1.0)),
                    bandwidth=float(self.rng.uniform(0.20, 1.0)),
                    signal=float(self.rng.uniform(0.20, 1.0)),
                    queue_load=float(self.rng.uniform(0.0, 0.85)),
                )
            )
        return devices

    def _sample_tasks(self) -> list[TaskRequirement]:
        start = self.system.n_edges + self.system.n_mobiles
        tasks: list[TaskRequirement] = []
        for idx in range(self.system.n_tasks):
            data_mb = float(self.rng.uniform(0.1, 1.0))
            cycles_g = float(self.rng.uniform(0.1, 1.0))
            tasks.append(
                TaskRequirement(
                    task_node=start + idx,
                    data_mb=data_mb,
                    cycles_g=cycles_g,
                    min_compute=float(0.35 + 0.35 * cycles_g),
                    min_battery=float(0.20 + 0.25 * data_mb),
                    priority=int(self.rng.choice([0, 1], p=[0.7, 0.3])),
                )
            )
        return tasks

    def _build_features(
        self,
        n_nodes: int,
        devices: Iterable[DeviceState],
        tasks: Iterable[TaskRequirement],
    ) -> torch.Tensor:
        features = torch.zeros(n_nodes, self.system.feature_dim, dtype=torch.float32)
        features[: self.system.n_edges, 0] = 1.0
        features[: self.system.n_edges, 3] = 1.0
        features[: self.system.n_edges, 4] = 1.0

        for device in devices:
            row = features[device.node_id]
            row[1] = 1.0
            row[3] = device.compute
            row[4] = device.battery
            row[5] = device.bandwidth
            row[6] = device.signal
            row[7] = device.queue_load
            row[8] = 1.0 - device.queue_load

        for task in tasks:
            row = features[task.task_node]
            row[2] = 1.0
            row[9] = task.data_mb
            row[10] = task.cycles_g
            row[11] = task.min_compute
            row[12] = task.min_battery
            row[13] = float(task.priority)

        return features

    def _build_hyperedges(
        self, devices: list[DeviceState], tasks: list[TaskRequirement]
    ) -> list[HyperedgeRecord]:
        hyperedges: list[HyperedgeRecord] = []
        for edge_id in range(self.system.n_edges):
            domain = [device.node_id for device in devices if device.edge_id == edge_id]
            hyperedges.append(
                HyperedgeRecord(vertices=[edge_id, *domain], weight=1.0, kind="edge_device_domain", edge_node=edge_id)
            )

        for task in tasks:
            edge_node = int(self.rng.integers(0, self.system.n_edges))
            candidates = sorted(
                devices,
                key=lambda device: self.interaction_strength(task, device),
                reverse=True,
            )[: self.system.max_candidate_devices]
            phi = [self.interaction_strength(task, device) for device in candidates]
            hyperedges.append(
                HyperedgeRecord(
                    vertices=[edge_node, task.task_node, *[device.node_id for device in candidates]],
                    weight=float(np.mean(phi)),
                    kind="task_execution_candidate",
                    task_node=task.task_node,
                    edge_node=edge_node,
                )
            )
        return hyperedges

    def interaction_strength(self, task: TaskRequirement, device: DeviceState) -> float:
        alpha_l, alpha_t, alpha_e = self.scheduler.normalized_alphas()
        transmission_delay = task.data_mb / max(device.bandwidth * device.signal, 1e-6)
        compute_delay = task.cycles_g / max(device.compute * (1.0 - device.queue_load), 1e-6)
        latency_score = 1.0 / (1.0 + transmission_delay + compute_delay)
        throughput_score = device.bandwidth * device.compute * (1.0 - device.queue_load)
        energy_cost = task.data_mb * (1.0 - device.signal) + task.cycles_g * (1.0 - device.compute)
        energy_score = device.battery / (1.0 + energy_cost)
        return float(alpha_l * latency_score + alpha_t * throughput_score + alpha_e * energy_score)
