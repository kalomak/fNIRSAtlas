"""Run and plot learning curves with MOABB's ``LearningCurveSplitter``.

Each split trains on a fixed number of trials per class and tests on a held-out
share of the data. Results are saved by MOABB with the ``data_size`` and
``permutation`` of each split.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from moabb.datasets.base import CacheConfig
from moabb.evaluations.splitters import LearningCurveSplitter

from fnirsatlas.benchmark import (
    DEFAULT_PIPELINES,
    EVALUATIONS,
    PARADIGMS,
    _seed_configs,
    _seed_suffix,
    load_pipeline_file,
    results_dir,
    select_datasets,
)
from fnirsatlas.results import subject_mean


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/learning_curve.py --datasets "
            "Bnci2014FnirsMA --sizes 5 10 15 --n-perms 10 5 2"
        ),
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--pipelines", nargs="+", default=DEFAULT_PIPELINES)
    parser.add_argument("--evaluation", default="WithinSession", choices=EVALUATIONS)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[5, 10, 15], help="Trials per class"
    )
    parser.add_argument(
        "--n-perms",
        nargs="+",
        type=int,
        default=[10],
        help="Permutations, one value or one per size",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Held-out share of the data"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paradigm", default="FNIRSClassification")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--results", type=Path, help="Default: MOABB results root")
    parser.add_argument("--output", type=Path, help="Output image path")
    return parser.parse_args(argv)


def run(args):
    """Run the evaluation and return MOABB's results."""
    paradigm = PARADIGMS[args.paradigm]()
    configs = [config for path in args.pipelines for config in load_pipeline_file(path)]
    pipelines = {
        config["name"]: config["pipeline"]
        for config in _seed_configs(configs, args.seed)
    }
    n_perms = args.n_perms[0] if len(args.n_perms) == 1 else np.array(args.n_perms)
    evaluation = EVALUATIONS[args.evaluation](
        paradigm=paradigm,
        datasets=select_datasets(paradigm, args.datasets),
        hdf5_path=str(args.results or results_dir()),
        suffix=_seed_suffix("learning-curve", args.seed),
        overwrite=args.overwrite,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        cache_config=CacheConfig(use=True, save_epochs=True, save_array=True),
        cv_class=LearningCurveSplitter,
        cv_kwargs={
            "data_size": {"policy": "per_class", "value": np.array(args.sizes)},
            "n_perms": n_perms,
            "test_size": args.test_size,
            "random_state": args.seed,
        },
    )
    return evaluation.process(pipelines)


def plot(results):
    """Plot the subject-mean score against trials per class, one panel per dataset."""
    scores = subject_mean(results, ["dataset", "pipeline", "data_size"]).mul(100)
    names = scores.index.unique("dataset")
    fig, axes = plt.subplots(
        1, len(names), figsize=(4 * len(names), 3), sharey=True, squeeze=False
    )
    n_classes = results.groupby("dataset")["n_classes"].first()
    for ax, name in zip(axes[0], names, strict=True):
        for pipeline, curve in scores[name].groupby("pipeline"):
            curve = curve.droplevel("pipeline")
            ax.plot(curve.index / n_classes[name], curve, marker="o", label=pipeline)
        ax.set_title(name)
        ax.set_xlabel("Training trials per class")
    axes[0, 0].set_ylabel("Score (%)")
    axes[0, -1].legend(fontsize=7)
    return fig


def main(argv=None):
    """Run the learning curves and save their plot."""
    args = parse_args(argv)
    results = run(args)
    fig = plot(results)
    output = args.output or (
        results_dir() / "benchmark" / f"learning_curve_{args.evaluation}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
