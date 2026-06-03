import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from esr.config import SchedulerConfig, SystemConfig
from esr.data import MobileEdgeHypergraphBuilder
from esr.evaluation import RLConfig, evaluate_assignments
from esr.scheduler import CoarseOnlyScheduler, RandomScheduler
from experiments.compare_baselines import run_baselines


def test_simple_baselines_return_one_assignment_per_task():
    cfg = SchedulerConfig()
    snapshot = MobileEdgeHypergraphBuilder(SystemConfig(seed=31, n_tasks=4), cfg).build()

    for scheduler in (CoarseOnlyScheduler(cfg), RandomScheduler(cfg, seed=31)):
        assignments = scheduler.schedule(snapshot)
        metrics = evaluate_assignments(assignments, RLConfig())
        assert len(assignments) == len(snapshot.tasks)
        assert 0.0 <= metrics.completion_rate <= 1.0


def test_compare_baselines_generates_three_methods():
    rows = run_baselines(num_seeds=2, start_seed=50)
    methods = {row["method"] for row in rows}

    assert methods == {"greedy", "coarse_only", "random"}
    assert len(rows) == 2 * 3
