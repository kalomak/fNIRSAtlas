"""Compare 3+ pipelines across datasets with a Friedman test and Nemenyi post-hoc."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

from fnirsatlas.results import load_results, score_matrix


def significance_stars(p_value, alpha=0.05):
    """Return ``***``, ``**``, ``*`` or ``""`` for p below alpha/10, alpha/5, alpha."""
    for divisor, stars in [(10, "***"), (5, "**"), (1, "*")]:
        if p_value < alpha / divisor:
            return stars
    return ""


def nemenyi(mean_ranks, n_datasets, alpha=0.05):
    """Return pairwise Nemenyi p-values and the critical difference.

    Demšar (2006), Statistical Comparisons of Classifiers over Multiple Data Sets.
    """
    n_pipelines = len(mean_ranks)
    standard_error = np.sqrt(n_pipelines * (n_pipelines + 1) / (6 * n_datasets))
    ranks = mean_ranks.to_numpy()
    differences = np.abs(np.subtract.outer(ranks, ranks))
    p_values = studentized_range.sf(
        differences / standard_error * np.sqrt(2), n_pipelines, np.inf
    )
    critical_value = studentized_range.ppf(1 - alpha, n_pipelines, np.inf) / np.sqrt(2)
    p_values = pd.DataFrame(p_values, mean_ranks.index, mean_ranks.index)
    return p_values, critical_value * standard_error


def pipeline_summary(scores):
    """Return mean score and mean rank per pipeline, best rank first."""
    return pd.DataFrame(
        {
            "mean_score": scores.mean(),
            "mean_rank": scores.rank(axis=1, ascending=False).mean(),
        }
    ).sort_values(["mean_rank", "mean_score"], ascending=[True, False])


def posthoc_rows(scores, p_values, alpha):
    """Return Nemenyi rows for every pipeline pair, higher mean score first."""
    order = scores.mean().sort_values(ascending=False).index
    return pd.DataFrame(
        [
            {
                "pipeline_1": first,
                "pipeline_2": second,
                "diff": (scores[first] - scores[second]).mean(),
                "p": p_values.loc[first, second],
                "sig": significance_stars(p_values.loc[first, second], alpha),
            }
            for first, second in combinations(order, 2)
        ]
    )


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/friedman_test.py --evaluation CrossSubject "
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
    parser.add_argument("--show-blocks", action="store_true", help="List the datasets")
    return parser.parse_args(argv)


def main(argv=None):
    """Load benchmark results and print the Friedman and Nemenyi results."""
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
    scores = score_matrix(results, args.score)
    if scores.shape[1] < 3:
        raise ValueError("Friedman test needs at least 3 pipelines")
    statistic, p_value = friedmanchisquare(*scores.T.to_numpy())
    summary = pipeline_summary(scores)
    p_values, critical_difference = nemenyi(
        summary["mean_rank"], len(scores), args.alpha
    )

    print(f"Friedman test on {args.score} dataset means (alpha={args.alpha})")
    print(f"Blocks: {len(scores)}")
    print(summary.to_string(float_format=lambda value: f"{value:.2f}"))
    stars = significance_stars(p_value, args.alpha)
    print(f"\nFriedman: chi2={statistic:.3f}, p={p_value:.3g} {stars}")
    print(f"Nemenyi critical difference: {critical_difference:.3f}\n")
    rows = posthoc_rows(scores, p_values, args.alpha)
    print(rows.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if args.show_blocks:
        print("\nBlocks:")
        for dataset in scores.index:
            print(f"- {dataset}")


if __name__ == "__main__":
    main()
