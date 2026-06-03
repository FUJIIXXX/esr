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

The default `hg.py` mode now runs validation, evaluates the three simple baselines over 10 seeds and saves a reward plot. Generated files are written to `results/baselines/`:

- `baseline_metrics.csv`
- `baseline_summary.txt`
- `baseline_reward.png`

For validation only, run:

```bash
python hg.py --mode validate
```

or call the module directly:

```bash
python -m experiments.minimal_validation
```


## How to read the validation output

- `nodes=a = b edge + c mobile + d task` means the hypergraph contains all three node types from the proposal.
- `hyperedges` contains edge-device domain hyperedges plus task-candidate hyperedges. A task-candidate hyperedge connects one edge node, one task node and several candidate mobile devices.
- `node_embeddings=(N, 64)` means DHG produced one 64-dimensional vector for every graph node.
- `graph_state=(192,)` is `64 * 3`: typed mean pooling over edge, mobile and task node embeddings. This fixed-length vector is what the future RL policy consumes.
- `policy alphas` and `learned_threshold_candidate` are emitted by an untrained policy head in this minimum demo; they prove the interface works, but they are not optimized yet.
- The printed assignments still use the fixed greedy scheduler threshold from `SchedulerConfig`. `fine` means multiple devices are selected for a split task; `coarse` means one device handles the whole task; `remote` means cloud or edge-edge fallback.
- `score` is the summed interaction strength `Φ(t,S)` over selected devices, computed from normalized latency, throughput and energy signals.


## Train the policy head

The scheduler is non-differentiable because it sorts devices, chooses Top-K
candidates and switches among `coarse` / `fine` / `remote`. Therefore the policy
head should not be trained with ordinary supervised backpropagation through the
scheduler. The included first training step is actor-critic RL:

```bash
python -m experiments.train_policy --episodes 200
```

What is trained:

- HGNN encoder: converts the weighted hypergraph into node embeddings and a fixed graph state.
- Stochastic actor: samples `(alpha_latency, alpha_throughput, alpha_energy)` from a Dirichlet distribution and `threshold` from a Beta distribution.
- Critic: estimates the scalar reward baseline for lower-variance policy-gradient updates.

Reward currently combines completed-task interaction score, priority-task bonus,
a small coarse-grained bonus, a fine-grained split penalty and a remote fallback
penalty. This is only the first simulator reward; for a paper, the next changes
should add deadlines, energy budgets, queue updates and held-out trace evaluation.

## What remains beyond training the policy head

Training the policy head is necessary, but it is not the only open problem. A
strong paper implementation should also add:

- a time-stepped simulator that updates queue length, residual energy and task arrivals;
- a subtask decomposition model instead of only selecting Top-K devices;
- deadline/energy/cost constraints in the reward and metrics;
- more baselines such as fine-only, earliest-deadline-first, local-only, cloud-only, GNN+greedy and GNN+RL;
- train/validation/test seeds or real MEC traces for statistically reliable comparison;
- visualization beyond line charts, such as confidence intervals, bar charts for average QoS and ablation plots for reward weights.


## Compare baselines and plot results

The repository now includes three schedulers for quick comparison:

- `greedy`: the current interaction-strength baseline that can choose coarse, fine or remote execution.
- `coarse_only`: a simple baseline that always tries to put each task on one feasible device.
- `random`: a weak baseline that randomly chooses feasible coarse/fine assignments.

Run the comparison over multiple random seeds and save plain-text results:

```bash
python -m experiments.compare_baselines --num-seeds 30
```

This creates:

- `results/baselines/baseline_metrics.csv` with per-seed metrics.
- `results/baselines/baseline_summary.txt` with mean metrics per method.

Then plot a line chart, for example reward over seeds:

```bash
python -m experiments.plot_baselines --metric reward
```

You can also use the all-in-one entrypoint, which runs validation, comparison and plotting together:

```bash
python hg.py --mode all --num-seeds 30 --metric reward
```

You can also plot `completion_rate`, `remote_rate`, `coarse_rate`, `fine_rate` or `priority_completion_rate` by changing `--metric`.

## Suggested next experiments

- Replace synthetic sampling with traces from a real MEC/IoT dataset.
- Add a simulator step that updates queue length, energy and deadline misses.
- Train `SchedulerPolicyHead` with PPO/SAC where the action controls
  `(alpha_latency, alpha_throughput, alpha_energy, threshold)`.
- Compare coarse-only, fine-only, greedy, GNN+greedy and GNN+RL variants.
