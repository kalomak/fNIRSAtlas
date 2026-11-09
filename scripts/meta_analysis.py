"""Compare two pipelines with MOABB's per-dataset tests and meta-analysis."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from moabb.analysis import compute_dataset_statistics, find_significant_differences
from moabb.analysis.plotting import meta_analysis_plot

from fnirsatlas.benchmark import results_dir
from fnirsatlas.results import load_results


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/meta_analysis.py --evaluation CrossSubject "
            "--pipelines Windowed_LDA eegnet --seeds 0 1 2 3 4"
        ),
    )
    parser.add_argument("--evaluation", required=True, help="e.g. CrossSubject")
    parser.add_argument(
        "--pipelines", nargs=2, required=True, help="Two pipeline file stems"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--paradigm", default="FNIRSClassification")
    parser.add_argument("--datasets", nargs="+", help="Default: all of the paradigm")
    parser.add_argument("--exclude-datasets", nargs="+")
    parser.add_argument("--score", default="score", help="e.g. score_kappa")
    parser.add_argument("--results", type=Path, help="Default: MOABB results root")
    parser.add_argument("--output", type=Path, help="Output image path")
    return parser.parse_args(argv)


def main(argv=None):
    """Load benchmark results, print the meta-analysis and save its plot."""
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
    first, second = args.pipelines
    # MOABB averages each subject over sessions and seeds before testing.
    stats = compute_dataset_statistics(results.assign(score=results[args.score]))
    p_values, effects = find_significant_differences(stats)
    pair = stats[(stats["pipe1"] == first) & (stats["pipe2"] == second)]

    print(f"{first} > {second} on {args.score}, one-sided per dataset")
    print(pair[["dataset", "nsub", "smd", "p"]].to_string(index=False))
    print(
        f"\nCombined: SMD={effects.loc[first, second]:.3f}, "
        f"p={p_values.loc[first, second]:.3g}"
    )
    fig = meta_analysis_plot(stats, first, second)
    output = args.output or (
        results_dir()
        / "benchmark"
        / f"meta_analysis_{first}_vs_{second}_{args.evaluation}_{args.score}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
