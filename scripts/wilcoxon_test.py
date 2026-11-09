"""Compare two pipelines across datasets with a Wilcoxon signed-rank test."""

import argparse
from pathlib import Path

from scipy.stats import wilcoxon

from fnirsatlas.results import load_results, score_matrix


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/wilcoxon_test.py --evaluation CrossSubject "
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
    return parser.parse_args(argv)


def main(argv=None):
    """Load benchmark results and print the Wilcoxon signed-rank test."""
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
    scores = score_matrix(results, args.score)[[first, second]]
    scores["diff"] = scores[first] - scores[second]
    statistic, p_value = wilcoxon(scores[first], scores[second])
    print(f"Wilcoxon signed-rank test on {args.score} dataset means")
    print(
        scores.sort_values("diff", ascending=False).to_string(
            float_format="{:.1f}".format
        )
    )
    wins = (scores["diff"] > 0).sum()
    losses = (scores["diff"] < 0).sum()
    print(
        f"\n{first} vs {second}: mean diff={scores['diff'].mean():.2f}, "
        f"wins/losses={wins}/{losses}, W={statistic:.1f}, p={p_value:.3g}, "
        f"n={len(scores)}"
    )


if __name__ == "__main__":
    main()
