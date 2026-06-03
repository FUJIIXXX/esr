"""Run the 1-2 week minimum validation experiment.

Pipeline:
    DHG weighted hypergraph -> HGNN encoder -> fixed greedy scheduler

Install runtime dependencies first:
    pip install torch dhg numpy
"""

from esr.config import SchedulerConfig, SystemConfig
from esr.data import MobileEdgeHypergraphBuilder
from esr.model import HypergraphStateEncoder, SchedulerPolicyHead
from esr.scheduler import InteractionScheduler


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

    print("=== ESR minimum validation ===")
    print(f"nodes={snapshot.n_nodes}, hyperedges={len(snapshot.hyperedges)}")
    print(f"node_embeddings={tuple(node_embeddings.shape)}, graph_state={tuple(graph_state.shape)}")
    print(f"policy alphas={policy_params['alphas'].detach().cpu().numpy().round(3).tolist()}")
    print(f"policy threshold={float(policy_params['threshold']):.3f}")
    for item in assignments:
        target = item.fallback if item.fallback else item.device_nodes
        print(f"task={item.task_node} priority={item.priority} {item.granularity} -> {target} score={item.score:.3f}")


if __name__ == "__main__":
    run()
