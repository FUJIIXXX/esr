"""Plot baseline comparison curves from saved text/CSV data.

Run after ``experiments.compare_baselines``:
    python -m experiments.plot_baselines --input results/baselines/baseline_metrics.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ESR baseline comparison curves.")
    parser.add_argument("--input", type=Path, default=Path("results/baselines/baseline_metrics.csv"))
    parser.add_argument("--metric", type=str, default="reward")
    parser.add_argument("--output", type=Path, default=Path("results/baselines/baseline_reward.png"))
    return parser.parse_args()


def load_series(path: Path, metric: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        if metric not in reader.fieldnames:
            raise ValueError(f"Metric {metric!r} not found in {path}; available columns: {reader.fieldnames}")
        for row in reader:
            series[row["method"]].append((int(row["seed"]), float(row[metric])))
    return series


def plot_metric(input_path: Path, metric: str, output_path: Path) -> None:
    """Create and save a line chart for one metric from baseline CSV results."""

    import matplotlib.pyplot as plt

    series = load_series(input_path, metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for method, points in sorted(series.items()):
        points = sorted(points)
        seeds = [seed for seed, _ in points]
        values = [value for _, value in points]
        plt.plot(seeds, values, marker="o", linewidth=1.5, label=method)
    plt.xlabel("seed")
    plt.ylabel(metric)
    plt.title(f"Baseline comparison: {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    plot_metric(input_path=args.input, metric=args.metric, output_path=args.output)


if __name__ == "__main__":
    main()
