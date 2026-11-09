"""Save a critical-difference diagram of pipeline ranks across datasets."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from moabb.analysis.plotting import plot_critical_difference

from fnirsatlas.benchmark import results_dir
from fnirsatlas.results import load_results


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/cd_diagram.py --evaluation CrossSubject "
            "--pipelines Windowed_LDA eegnet fnirsnet23 --seeds 0 1 2 3 4"
        ),
    )
    parser.add_argument("--evaluation", required=True, help="e.g. CrossSubject")
    parser.add_argument(
        "--pipelines", nargs="+", required=True, help="Pipeline file stems"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--paradigm", default="FNIRSClassification")
    parser.add_argument("--datasets", nargs="+", help="Default: all of the paradigm")
    parser.add_argument("--exclude-datasets", nargs="+")
    parser.add_argument("--score", default="score", help="e.g. score_kappa")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--results", type=Path, help="Default: MOABB results root")
    parser.add_argument("--output", type=Path, help="Output image path")
    return parser.parse_args(argv)


def main(argv=None):
    """Load benchmark results and save their critical-difference diagram."""
    args = parse_args(argv)
    results = load_results(
        args.paradigm,
        args.evaluation,
        args.pipelines,
        args.seeds,
        args.datasets,
        args.exclude_datasets,
        args.results,
    )
    results = results.assign(score=results[args.score])
    fig = plot_critical_difference(results, alpha=args.alpha)
    output = args.output or (
        results_dir() / "benchmark" / f"cd_diagram_{args.evaluation}_{args.score}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
