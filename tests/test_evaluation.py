import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")

from esr.config import RLConfig, SchedulerConfig, SystemConfig
from esr.data import MobileEdgeHypergraphBuilder
from esr.evaluation import config_from_action, evaluate_policy_action


def test_policy_action_evaluation_returns_metrics():
    base_cfg = SchedulerConfig()
    snapshot = MobileEdgeHypergraphBuilder(SystemConfig(seed=21), base_cfg).build()
    scheduler_cfg = config_from_action((0.2, 0.5, 0.3), 0.4, base_cfg)

    assignments, metrics = evaluate_policy_action(
        snapshot=snapshot,
        base_scheduler_cfg=scheduler_cfg,
        reward_cfg=RLConfig(),
        alphas=(0.2, 0.5, 0.3),
        threshold=0.4,
    )

    assert len(assignments) == len(snapshot.tasks)
    assert 0.0 <= metrics.completion_rate <= 1.0
    assert 0.0 <= metrics.remote_rate <= 1.0
    assert metrics.coarse_rate + metrics.fine_rate + metrics.remote_rate == pytest.approx(1.0)
