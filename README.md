# ESR Hypergraph Scheduling Prototype

This repository contains a minimum validation system for mobile-edge computing
(MEC) task scheduling with weighted hypergraphs and DeepHypergraph (DHG).

## Research pipeline

1. Generate a multi-edge MEC snapshot with edge nodes, mobile nodes and task nodes.
2. Build weighted hyperedges for edge-device domains and task execution candidates.
3. Encode the weighted hypergraph with DHG `HGNNConv`.
4. Convert dynamic-size hypergraphs into a fixed-size state vector with typed pooling.
5. Run a resource-aware greedy scheduler as the first fixed-rule baseline.
6. Use the provided policy head later to learn latency/throughput/energy weights and the threshold with RL.

## Install

```bash
python -m pip install -r requirements.txt
```

DHG currently documents `pip install dhg` as the stable installation path. Its
latest README states that DHG is built on PyTorch and supports high-order
message passing for hypergraph neural networks.

## Run the minimum validation

```bash
python hg.py
```

or:

```bash
python -m experiments.minimal_validation
```

## Suggested next experiments

- Replace synthetic sampling with traces from a real MEC/IoT dataset.
- Add a simulator step that updates queue length, energy and deadline misses.
- Train `SchedulerPolicyHead` with PPO/SAC where the action controls
  `(alpha_latency, alpha_throughput, alpha_energy, threshold)`.
- Compare coarse-only, fine-only, greedy, GNN+greedy and GNN+RL variants.
