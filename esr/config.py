"""Configuration objects for the edge scheduling research prototype."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    """Static knobs for one simulated multi-edge MEC snapshot."""

    n_edges: int = 2
    n_mobiles: int = 12
    n_tasks: int = 8
    feature_dim: int = 16
    max_candidate_devices: int = 4
    seed: int = 7


@dataclass(frozen=True)
class SchedulerConfig:
    """Heuristic scheduler parameters that can later be emitted by an RL policy."""

    alpha_latency: float = 0.4
    alpha_throughput: float = 0.3
    alpha_energy: float = 0.3
    threshold: float = 0.62
    fine_grained_top_k: int = 3
    overload_penalty: float = 0.15

    def normalized_alphas(self) -> tuple[float, float, float]:
        total = self.alpha_latency + self.alpha_throughput + self.alpha_energy
        if total <= 0:
            return (1 / 3, 1 / 3, 1 / 3)
        return (
            self.alpha_latency / total,
            self.alpha_throughput / total,
            self.alpha_energy / total,
        )
