"""Convenient command-line entrypoint for ESR experiments.

By default this runs the minimum validation, compares simple baselines and
creates a reward line chart so first-time users get both text output and a plot.
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run ESR validation, baseline comparison and plotting.")
    parser.add_argument(
        "--mode",
        choices=("all", "validate", "compare", "plot"),
        default="all",
        help="Experiment stage to run. Default 'all' validates, compares baselines and plots reward.",
    )
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of random seeds for baseline comparison.")
    parser.add_argument("--start-seed", type=int, default=100, help="First seed used by baseline comparison.")
    parser.add_argument("--metric", type=str, default="reward", help="Metric to plot from saved baseline data.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines"), help="Directory for CSV/TXT/PNG outputs.")
    return parser.parse_args()


def run_comparison(output_dir: Path, num_seeds: int, start_seed: int) -> Path:
    from experiments.compare_baselines import run_baselines, write_outputs

    rows = run_baselines(num_seeds=num_seeds, start_seed=start_seed)
    write_outputs(rows, output_dir)
    return output_dir / "baseline_metrics.csv"


def run_plot(input_path: Path, metric: str, output_dir: Path) -> None:
    from experiments.plot_baselines import plot_metric

    output_path = output_dir / f"baseline_{metric}.png"
    plot_metric(input_path=input_path, metric=metric, output_path=output_path)


def main() -> None:
    args = parse_args()
    metrics_path = args.output_dir / "baseline_metrics.csv"

    if args.mode in {"all", "validate"}:
        from experiments.minimal_validation import run as run_minimal_validation

        run_minimal_validation()

    if args.mode in {"all", "compare"}:
        print("\n=== ESR baseline comparison ===")
        metrics_path = run_comparison(args.output_dir, args.num_seeds, args.start_seed)

    if args.mode in {"all", "plot"}:
        print("\n=== ESR baseline plot ===")
        if not metrics_path.exists():
            metrics_path = run_comparison(args.output_dir, args.num_seeds, args.start_seed)
        try:
            run_plot(metrics_path, args.metric, args.output_dir)
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("matplotlib is not installed; run `python -m pip install -r requirements.txt` and retry plotting.")


if __name__ == "__main__":
    main()
