"""Run the 1-2 week minimum validation experiment.

Pipeline:
    DHG weighted hypergraph -> HGNN encoder -> fixed greedy scheduler

Install runtime dependencies first:
    pip install torch dhg numpy
"""

from esr.config import SchedulerConfig, SystemConfig
from esr.data import MecaSnapshot, MobileEdgeHypergraphBuilder
from esr.model import HypergraphStateEncoder, SchedulerPolicyHead
from esr.scheduler import Assignment, InteractionScheduler


def _format_assignment(item: Assignment, snapshot: MecaSnapshot) -> str:
    """Render one scheduler decision with human-readable node labels."""

    task_idx = item.task_node - snapshot.n_edges - snapshot.n_mobiles
    if item.fallback:
        target = item.fallback
    else:
        device_labels = [f"m{node_id - snapshot.n_edges}" for node_id in item.device_nodes]
        target = f"{item.device_nodes} ({', '.join(device_labels)})"
    granularity_text = {
        "coarse": "coarse/粗粒度：一个端设备执行完整任务",
        "fine": "fine/细粒度：多个端设备协同执行拆分后的子任务",
        "remote": "remote/远端：本边缘域端设备资源不足",
    }[item.granularity]
    return (
        f"task_node={item.task_node} (t{task_idx}) priority={item.priority} "
        f"{granularity_text} -> {target} score={item.score:.3f}"
    )


def format_validation_report(
    snapshot: MecaSnapshot,
    node_embeddings_shape: tuple[int, ...],
    graph_state_shape: tuple[int, ...],
    policy_alphas: list[float],
    policy_threshold: float,
    scheduler_threshold: float,
    assignments: list[Assignment],
) -> str:
    """Create an interpretable text report for the minimum validation run."""

    lines = [
        "=== ESR minimum validation ===",
        "[Graph] nodes = edge nodes + mobile/device nodes + task nodes",
        f"        nodes={snapshot.n_nodes} = {snapshot.n_edges} edge + {snapshot.n_mobiles} mobile + {len(snapshot.tasks)} task",
        f"        hyperedges={len(snapshot.hyperedges)} = {snapshot.n_edges} edge-device domain + {len(snapshot.tasks)} task-candidate hyperedges",
        "[GNN] node_embeddings shape = one embedding vector per node after DHG HGNNConv",
        f"      node_embeddings={node_embeddings_shape}",
        "[RL state] graph_state shape = mean(edge embeddings) || mean(mobile embeddings) || mean(task embeddings)",
        f"           graph_state={graph_state_shape}",
        "[Policy head demo] random/untrained actor output; not a trained decision yet",
        f"                   alphas(latency, throughput, energy)={[round(x, 3) for x in policy_alphas]}",
        f"                   learned_threshold_candidate={policy_threshold:.3f}",
        f"[Scheduler baseline] fixed_threshold={scheduler_threshold:.3f}; current assignments below use the fixed baseline, not the untrained policy threshold",
        "[Assignments] priority=1 is served before priority=0; score is summed interaction strength Φ(t,S)",
    ]
    lines.extend(_format_assignment(item, snapshot) for item in assignments)
    return "\n".join(lines)


def run() -> None:
    system_cfg = SystemConfig(n_edges=2, n_mobiles=10, n_tasks=6, feature_dim=16, seed=42)
    scheduler_cfg = SchedulerConfig(threshold=0.62)
    snapshot = MobileEdgeHypergraphBuilder(system_cfg, scheduler_cfg).build()
    hypergraph = snapshot.to_dhg_hypergraph()

    encoder = HypergraphStateEncoder(in_dim=system_cfg.feature_dim, hidden_dim=64, out_dim=64, num_layers=2)
    policy = SchedulerPolicyHead(state_dim=64 * 3)
    encoder.eval()
    policy.eval()

    node_embeddings, graph_state = encoder(snapshot.node_features, hypergraph, snapshot.node_types)
    policy_params = policy(graph_state)
    assignments = InteractionScheduler(scheduler_cfg).schedule(snapshot)

    print(
        format_validation_report(
            snapshot=snapshot,
            node_embeddings_shape=tuple(node_embeddings.shape),
            graph_state_shape=tuple(graph_state.shape),
            policy_alphas=policy_params["alphas"].detach().cpu().numpy().tolist(),
            policy_threshold=float(policy_params["threshold"]),
            scheduler_threshold=scheduler_cfg.threshold,
            assignments=assignments,
        )
    )


if __name__ == "__main__":
    run()
