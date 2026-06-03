"""Compare simple scheduling baselines and save results for plotting.

The script writes two plain-text files under ``results/baselines`` by default:
- ``baseline_metrics.csv``: per-seed metrics for each method.
- ``baseline_summary.txt``: mean metrics per method.

Run after installing runtime dependencies:
    python -m experiments.compare_baselines --num-seeds 30
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from esr.config import RLConfig, SchedulerConfig, SystemConfig
from esr.data import MobileEdgeHypergraphBuilder
from esr.evaluation import SchedulingMetrics, evaluate_assignments
from esr.scheduler import CoarseOnlyScheduler, InteractionScheduler, RandomScheduler

METRIC_NAMES = (
    "reward",
    "mean_score",
    "completion_rate",
    "remote_rate",
    "coarse_rate",
    "fine_rate",
    "priority_completion_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ESR scheduling baselines.")
    parser.add_argument("--num-seeds", type=int, default=30)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"))
    return parser.parse_args()


def metric_row(method: str, seed: int, metrics: SchedulingMetrics) -> dict[str, str | float | int]:
    row: dict[str, str | float | int] = {"method": method, "seed": seed}
    for name in METRIC_NAMES:
        row[name] = getattr(metrics, name)
    return row


def run_baselines(num_seeds: int, start_seed: int) -> list[dict[str, str | float | int]]:
    scheduler_cfg = SchedulerConfig(threshold=0.62)
    reward_cfg = RLConfig()
    rows: list[dict[str, str | float | int]] = []

    for offset in range(num_seeds):
        seed = start_seed + offset
        system_cfg = SystemConfig(seed=seed)
        snapshot = MobileEdgeHypergraphBuilder(system_cfg, scheduler_cfg).build()
        baselines = {
            "greedy": InteractionScheduler(scheduler_cfg),
            "coarse_only": CoarseOnlyScheduler(scheduler_cfg),
            "random": RandomScheduler(scheduler_cfg, seed=seed),
        }
        for method, scheduler in baselines.items():
            assignments = scheduler.schedule(snapshot)
            rows.append(metric_row(method, seed, evaluate_assignments(assignments, reward_cfg)))
    return rows


def write_outputs(rows: list[dict[str, str | float | int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_metrics.csv"
    summary_path = output_dir / "baseline_summary.txt"

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["method", "seed", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, str | float | int]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    lines = ["method " + " ".join(f"mean_{name}" for name in METRIC_NAMES)]
    for method in sorted(grouped):
        method_rows = grouped[method]
        means = [sum(float(row[name]) for row in method_rows) / len(method_rows) for name in METRIC_NAMES]
        lines.append(method + " " + " ".join(f"{value:.6f}" for value in means))
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"Saved per-seed metrics to {csv_path}")
    print(f"Saved summary metrics to {summary_path}")


def main() -> None:
    args = parse_args()
    rows = run_baselines(num_seeds=args.num_seeds, start_seed=args.start_seed)
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
