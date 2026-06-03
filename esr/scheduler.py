"""Greedy baseline scheduler over MEC hypergraph snapshots."""

from dataclasses import dataclass

from esr.config import SchedulerConfig
from esr.data import DeviceState, MecaSnapshot, TaskRequirement


@dataclass(frozen=True)
class Assignment:
    task_node: int
    device_nodes: list[int]
    granularity: str
    score: float
    priority: int
    fallback: str | None = None


class InteractionScheduler:
    """Resource-aware coarse/fine-grained scheduler.

    Rule implemented from the proposal:
    1. If a single device satisfies the task minimum resource demand and its
       matching score is above ``threshold``, choose coarse-grained execution.
    2. Otherwise split to Top-K feasible devices and maximize the summed score.
    3. If no local feasible device exists, high-priority tasks request edge-edge
       cooperation; ordinary tasks fall back to cloud.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config

    def schedule(self, snapshot: MecaSnapshot) -> list[Assignment]:
        devices_by_node = {device.node_id: device for device in snapshot.devices}
        task_order = sorted(snapshot.tasks, key=lambda task: (-task.priority, task.task_node))
        assignments: list[Assignment] = []

        for task in task_order:
            candidates = sorted(
                snapshot.devices,
                key=lambda device: self._interaction_strength(task, device),
                reverse=True,
            )
            feasible = [device for device in candidates if self._single_device_feasible(task, device)]
            best = feasible[0] if feasible else None
            best_score = self._interaction_strength(task, best) if best is not None else 0.0

            if best is not None and best_score >= self.config.threshold:
                chosen = [best]
                granularity = "coarse"
                score = best_score
            elif feasible:
                chosen = feasible[: self.config.fine_grained_top_k]
                granularity = "fine"
                score = sum(self._interaction_strength(task, device) for device in chosen)
            else:
                cooperation = task.priority > 0
                assignments.append(
                    Assignment(
                        task_node=task.task_node,
                        device_nodes=[],
                        granularity="remote",
                        score=0.0,
                        priority=task.priority,
                        fallback="edge_edge_cooperation" if cooperation else "cloud",
                    )
                )
                continue

            self._reserve(task, chosen)
            assignments.append(
                Assignment(
                    task_node=task.task_node,
                    device_nodes=[device.node_id for device in chosen if device.node_id in devices_by_node],
                    granularity=granularity,
                    score=float(score),
                    priority=task.priority,
                )
            )
        return assignments

    def _single_device_feasible(self, task: TaskRequirement, device: DeviceState) -> bool:
        available_compute = device.compute * (1.0 - device.queue_load)
        return available_compute >= task.min_compute and device.battery >= task.min_battery

    def _interaction_strength(self, task: TaskRequirement, device: DeviceState | None) -> float:
        if device is None:
            return 0.0
        alpha_l, alpha_t, alpha_e = self.config.normalized_alphas()
        transmission_delay = task.data_mb / max(device.bandwidth * device.signal, 1e-6)
        compute_delay = task.cycles_g / max(device.compute * (1.0 - device.queue_load), 1e-6)
        latency_score = 1.0 / (1.0 + transmission_delay + compute_delay)
        throughput_score = device.bandwidth * device.compute * (1.0 - device.queue_load)
        energy_cost = task.data_mb * (1.0 - device.signal) + task.cycles_g * (1.0 - device.compute)
        energy_score = device.battery / (1.0 + energy_cost)
        overload_penalty = self.config.overload_penalty * device.queue_load
        return float(max(0.0, alpha_l * latency_score + alpha_t * throughput_score + alpha_e * energy_score - overload_penalty))

    def _reserve(self, task: TaskRequirement, devices: list[DeviceState]) -> None:
        # The current immutable DeviceState records remain unchanged so each
        # generated snapshot is reproducible. This placeholder marks where a
        # simulator can later update queue/battery for multi-step RL rollouts.
        _ = (task, devices)


class CoarseOnlyScheduler(InteractionScheduler):
    """Baseline that never splits a task across multiple mobile devices."""

    def schedule(self, snapshot: MecaSnapshot) -> list[Assignment]:
        task_order = sorted(snapshot.tasks, key=lambda task: (-task.priority, task.task_node))
        assignments: list[Assignment] = []

        for task in task_order:
            feasible = [device for device in snapshot.devices if self._single_device_feasible(task, device)]
            if not feasible:
                assignments.append(self._remote_assignment(task))
                continue
            best = max(feasible, key=lambda device: self._interaction_strength(task, device))
            assignments.append(
                Assignment(
                    task_node=task.task_node,
                    device_nodes=[best.node_id],
                    granularity="coarse",
                    score=float(self._interaction_strength(task, best)),
                    priority=task.priority,
                )
            )
        return assignments

    def _remote_assignment(self, task: TaskRequirement) -> Assignment:
        return Assignment(
            task_node=task.task_node,
            device_nodes=[],
            granularity="remote",
            score=0.0,
            priority=task.priority,
            fallback="edge_edge_cooperation" if task.priority > 0 else "cloud",
        )


class RandomScheduler(InteractionScheduler):
    """Baseline that randomly chooses coarse or fine feasible assignments."""

    def __init__(self, config: SchedulerConfig, seed: int = 0):
        super().__init__(config)
        import random

        self.rng = random.Random(seed)

    def schedule(self, snapshot: MecaSnapshot) -> list[Assignment]:
        task_order = sorted(snapshot.tasks, key=lambda task: (-task.priority, task.task_node))
        assignments: list[Assignment] = []

        for task in task_order:
            feasible = [device for device in snapshot.devices if self._single_device_feasible(task, device)]
            if not feasible:
                assignments.append(
                    Assignment(
                        task_node=task.task_node,
                        device_nodes=[],
                        granularity="remote",
                        score=0.0,
                        priority=task.priority,
                        fallback="edge_edge_cooperation" if task.priority > 0 else "cloud",
                    )
                )
                continue

            use_fine = len(feasible) > 1 and self.rng.random() >= 0.5
            if use_fine:
                k = min(self.config.fine_grained_top_k, len(feasible))
                chosen = self.rng.sample(feasible, k=k)
                granularity = "fine"
                score = sum(self._interaction_strength(task, device) for device in chosen)
            else:
                chosen = [self.rng.choice(feasible)]
                granularity = "coarse"
                score = self._interaction_strength(task, chosen[0])

            assignments.append(
                Assignment(
                    task_node=task.task_node,
                    device_nodes=[device.node_id for device in chosen],
                    granularity=granularity,
                    score=float(score),
                    priority=task.priority,
                )
            )
        return assignments
