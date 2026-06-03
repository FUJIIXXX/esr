"""DHG-based hypergraph encoder and policy heads."""

import torch
import torch.nn as nn

from dhg.nn import HGNNConv


class HypergraphStateEncoder(nn.Module):
    """HGNN encoder that produces node embeddings and a fixed graph state.

    ``HGNNConv`` is the canonical DHG layer for the AAAI 2019 HGNN operator.
    The readout uses masked mean-pooling per node type, so the RL policy always
    receives a fixed-size vector even when device/task counts change.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = [in_dim, *[hidden_dim] * max(0, num_layers - 1), out_dim]
        self.layers = nn.ModuleList(
            HGNNConv(
                dims[idx],
                dims[idx + 1],
                use_bn=idx < num_layers - 1,
                drop_rate=dropout,
                is_last=idx == num_layers - 1,
            )
            for idx in range(num_layers)
        )
        self.out_dim = out_dim

    def forward(self, node_features: torch.Tensor, hypergraph, node_types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = node_features
        for layer in self.layers:
            x = layer(x, hypergraph)
        graph_state = self._typed_readout(x, node_types)
        return x, graph_state

    def _typed_readout(self, x: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
        pooled = []
        for node_type in (0, 1, 2):
            mask = node_types == node_type
            if mask.any():
                pooled.append(x[mask].mean(dim=0))
            else:
                pooled.append(torch.zeros(self.out_dim, dtype=x.dtype, device=x.device))
        return torch.cat(pooled, dim=-1)


class SchedulerPolicyHead(nn.Module):
    """Small actor head that emits dynamic scheduler parameters.

    Output semantics:
    - ``alphas`` are normalized weights for latency, throughput and energy.
    - ``threshold`` is in [0, 1] and can replace the fixed heuristic threshold.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 3)
        self.threshold_head = nn.Linear(hidden_dim, 1)

    def forward(self, graph_state: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(graph_state)
        return {
            "alphas": torch.softmax(self.alpha_head(h), dim=-1),
            "threshold": torch.sigmoid(self.threshold_head(h)).squeeze(-1),
        }
