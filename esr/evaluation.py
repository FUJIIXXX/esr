"""Reward and metrics for policy-head training.

The scheduler contains discrete choices (coarse/fine/remote and Top-K device
selection), so the policy cannot be trained by ordinary backpropagation through
that scheduler. The first practical step is therefore an RL loop: sample
scheduler parameters, run the non-differentiable scheduler, convert the result
into a scalar reward, then update the policy with policy-gradient loss.
"""

from dataclasses import dataclass

from esr.config import RLConfig, SchedulerConfig
from esr.data import MecaSnapshot
from esr.scheduler import Assignment, InteractionScheduler


@dataclass(frozen=True)
class SchedulingMetrics:
    """Aggregated QoS counters for one scheduling rollout."""

    reward: float
    mean_score: float
    completion_rate: float
    remote_rate: float
    coarse_rate: float
    fine_rate: float
    priority_completion_rate: float


def config_from_action(
    alphas: tuple[float, float, float],
    threshold: float,
    base: SchedulerConfig,
) -> SchedulerConfig:
    """Create a scheduler config from an RL action while keeping fixed knobs."""

    alpha_latency, alpha_throughput, alpha_energy = alphas
    return SchedulerConfig(
        alpha_latency=float(alpha_latency),
        alpha_throughput=float(alpha_throughput),
        alpha_energy=float(alpha_energy),
        threshold=float(threshold),
        fine_grained_top_k=base.fine_grained_top_k,
        overload_penalty=base.overload_penalty,
    )


def evaluate_assignments(assignments: list[Assignment], reward_cfg: RLConfig) -> SchedulingMetrics:
    """Convert scheduler assignments into a scalar reward and diagnostics."""

    n_tasks = max(len(assignments), 1)
    completed = [item for item in assignments if item.granularity != "remote"]
    priority = [item for item in assignments if item.priority > 0]
    priority_completed = [item for item in priority if item.granularity != "remote"]
    coarse = [item for item in assignments if item.granularity == "coarse"]
    fine = [item for item in assignments if item.granularity == "fine"]
    remote = [item for item in assignments if item.granularity == "remote"]

    score_term = sum(item.score * (1.0 + reward_cfg.priority_weight * item.priority) for item in assignments)
    structure_term = reward_cfg.coarse_bonus * len(coarse) - reward_cfg.fine_penalty * len(fine)
    remote_term = -reward_cfg.remote_penalty * sum(1.0 + item.priority for item in remote)
    reward = (score_term + structure_term + remote_term) / n_tasks

    return SchedulingMetrics(
        reward=float(reward),
        mean_score=float(sum(item.score for item in assignments) / n_tasks),
        completion_rate=float(len(completed) / n_tasks),
        remote_rate=float(len(remote) / n_tasks),
        coarse_rate=float(len(coarse) / n_tasks),
        fine_rate=float(len(fine) / n_tasks),
        priority_completion_rate=float(len(priority_completed) / max(len(priority), 1)),
    )


def evaluate_policy_action(
    snapshot: MecaSnapshot,
    base_scheduler_cfg: SchedulerConfig,
    reward_cfg: RLConfig,
    alphas: tuple[float, float, float],
    threshold: float,
) -> tuple[list[Assignment], SchedulingMetrics]:
    """Run the scheduler with one policy action and return reward metrics."""

    scheduler_cfg = config_from_action(alphas=alphas, threshold=threshold, base=base_scheduler_cfg)
    assignments = InteractionScheduler(scheduler_cfg).schedule(snapshot)
    return assignments, evaluate_assignments(assignments, reward_cfg)
