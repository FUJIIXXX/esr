"""Train the HGNN policy head with a minimal actor-critic loop.

This script answers the next research step after minimum validation: the policy
head is trained to output dynamic scheduler weights and a threshold. It is still
a simulator prototype, but it contains all algorithmic pieces needed before
moving to PPO/SAC and real traces.

Run after installing dependencies:
    python -m experiments.train_policy --episodes 200
"""

import argparse
import random

import numpy as np
import torch

from esr.config import RLConfig, SchedulerConfig, SystemConfig
from esr.data import MobileEdgeHypergraphBuilder
from esr.evaluation import evaluate_policy_action
from esr.model import HypergraphStateEncoder, SchedulerActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESR scheduler policy head with actor-critic.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    system_template = SystemConfig(n_edges=2, n_mobiles=12, n_tasks=8, feature_dim=16, seed=args.seed)
    base_scheduler_cfg = SchedulerConfig(threshold=0.62)
    rl_cfg = RLConfig(episodes=args.episodes, learning_rate=args.learning_rate, seed=args.seed)

    encoder = HypergraphStateEncoder(in_dim=system_template.feature_dim, hidden_dim=64, out_dim=64, num_layers=2)
    actor_critic = SchedulerActorCritic(state_dim=64 * 3)
    optimizer = torch.optim.Adam([*encoder.parameters(), *actor_critic.parameters()], lr=rl_cfg.learning_rate)

    running_reward = 0.0
    for episode in range(1, rl_cfg.episodes + 1):
        system_cfg = SystemConfig(
            n_edges=system_template.n_edges,
            n_mobiles=system_template.n_mobiles,
            n_tasks=system_template.n_tasks,
            feature_dim=system_template.feature_dim,
            max_candidate_devices=system_template.max_candidate_devices,
            seed=args.seed + episode,
        )
        snapshot = MobileEdgeHypergraphBuilder(system_cfg, base_scheduler_cfg).build()
        hypergraph = snapshot.to_dhg_hypergraph()

        _, graph_state = encoder(snapshot.node_features, hypergraph, snapshot.node_types)
        action = actor_critic.sample_action(graph_state)
        alphas = tuple(float(x) for x in action["alphas"].detach().cpu().tolist())
        threshold = float(action["threshold"].detach().cpu())
        _, metrics = evaluate_policy_action(snapshot, base_scheduler_cfg, rl_cfg, alphas, threshold)

        reward = torch.tensor(metrics.reward, dtype=action["value"].dtype, device=action["value"].device)
        advantage = reward - action["value"]
        actor_loss = -(action["log_prob"] * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy_loss = -action["entropy"]
        loss = actor_loss + rl_cfg.value_coef * critic_loss + rl_cfg.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *actor_critic.parameters()], max_norm=1.0)
        optimizer.step()

        running_reward = 0.95 * running_reward + 0.05 * metrics.reward if episode > 1 else metrics.reward
        if episode == 1 or episode % args.log_every == 0:
            print(
                f"episode={episode:04d} reward={metrics.reward:.3f} running_reward={running_reward:.3f} "
                f"completion={metrics.completion_rate:.2f} remote={metrics.remote_rate:.2f} "
                f"coarse={metrics.coarse_rate:.2f} fine={metrics.fine_rate:.2f} "
                f"alphas={[round(x, 3) for x in alphas]} threshold={threshold:.3f}"
            )

    print("Training finished. Next: save checkpoints, evaluate on held-out seeds, then replace this loop with PPO/SAC.")


if __name__ == "__main__":
    train()
