import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from esr.config import SchedulerConfig, SystemConfig
from esr.data import MobileEdgeHypergraphBuilder
from esr.scheduler import InteractionScheduler


def test_builder_creates_weighted_task_hyperedges():
    cfg = SystemConfig(seed=3, n_tasks=8)
    snapshot = MobileEdgeHypergraphBuilder(cfg, SchedulerConfig()).build()

    task_edges = [edge for edge in snapshot.hyperedges if edge.kind == "task_execution_candidate"]

    assert len(task_edges) == cfg.n_tasks
    assert all(0.0 <= edge.weight <= 1.0 for edge in task_edges)
    assert snapshot.node_features.shape == (snapshot.n_nodes, cfg.feature_dim)


def test_scheduler_prioritizes_tasks_and_returns_assignments():
    cfg = SystemConfig(seed=11, n_tasks=5)
    snapshot = MobileEdgeHypergraphBuilder(cfg, SchedulerConfig()).build()
    assignments = InteractionScheduler(SchedulerConfig(threshold=0.95)).schedule(snapshot)

    assert len(assignments) == cfg.n_tasks
    assert [assignment.priority for assignment in assignments] == sorted(
        [assignment.priority for assignment in assignments], reverse=True
    )
    assert all(assignment.granularity in {"coarse", "fine", "remote"} for assignment in assignments)
