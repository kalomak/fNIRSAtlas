"""Run the configurable fNIRS benchmark."""

import argparse

from moabb.datasets.base import CacheConfig

from fnirsatlas.benchmark import (
    DEFAULT_PIPELINES,
    EVALUATIONS,
    PARADIGMS,
    fnirs_benchmark,
)


def parse_args(argv=None):
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/run_fnirs_benchmark.py "
            "--datasets Bnci2014FnirsMA --seed 1"
        ),
    )
    parser.add_argument("--contexts", help="Paradigm context YAML file")
    parser.add_argument("-d", "--datasets", nargs="+", help="Dataset codes")
    parser.add_argument("--exclude-datasets", nargs="+", help="Excluded dataset codes")
    parser.add_argument(
        "--evaluations", nargs="+", choices=EVALUATIONS, help="Evaluation types"
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Fit declared parameters instead of searching a pipeline param_grid",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Fold count where supported (default: 5)",
    )
    parser.add_argument("--optuna", action="store_true", help="Use Optuna search")
    parser.add_argument(
        "--output", help="Analysis directory (default: <results>/benchmark)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite results")
    parser.add_argument(
        "--paradigms", nargs="+", choices=PARADIGMS, help="fNIRS paradigms"
    )
    parser.add_argument(
        "-p",
        "--pipelines",
        nargs="+",
        default=DEFAULT_PIPELINES,
        metavar="FILE",
        help=f"Pipeline definition files (default: {' '.join(DEFAULT_PIPELINES)})",
    )
    parser.add_argument("--plot", action="store_true", help="Create analysis plots")
    parser.add_argument(
        "--results", help="MOABB result directory (default: MOABB_RESULTS)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Evaluation random seed")
    parser.add_argument("--suffix", default="", help="Result filename suffix")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the fNIRS benchmark CLI."""
    args = parse_args(argv)
    return fnirs_benchmark(
        pipelines=args.pipelines,
        evaluations=args.evaluations,
        paradigms=args.paradigms,
        results=args.results,
        overwrite=args.overwrite,
        output=args.output,
        suffix=args.suffix,
        n_jobs=args.n_jobs,
        plot=args.plot,
        contexts=args.contexts,
        include_datasets=args.datasets,
        exclude_datasets=args.exclude_datasets,
        n_splits=args.n_splits,
        cache_config=CacheConfig(use=True, save_epochs=True, save_array=True),
        optuna=args.optuna,
        random_state=args.seed,
        no_grid=args.no_grid,
    )


if __name__ == "__main__":
    main()
