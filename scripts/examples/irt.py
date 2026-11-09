"""Fit an item response theory (IRT) model to benchmark results with py-irt.

Pipelines are the respondents and (dataset, subject) pairs the items, so
datasets with more subjects weigh more in the abilities. A pipeline answers
an item correctly when its subject-mean score reaches the dataset's
threshold, chosen so that the share of correct answers is closest to the mean
score (Polo et al. 2024). Prints pipeline abilities and the mean item
difficulty and discrimination per dataset.

py-irt is not a dependency and needs Python below 3.12. Run in an isolated
environment::

    uv run --isolated --python 3.12 --extra data --with py-irt==0.7.1 \
        python scripts/examples/irt.py --evaluation CrossSubject \
        --pipelines Windowed_LDA eegnet fnirsnet23
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from py_irt.config import IrtConfig  # ty: ignore[unresolved-import]
from py_irt.dataset import Dataset  # ty: ignore[unresolved-import]
from py_irt.training import IrtModelTrainer  # ty: ignore[unresolved-import]

from fnirsatlas.results import load_results


def binarize(scores):
    """Return 1 where a score reaches the threshold that best keeps the mean."""
    values = scores.to_numpy()
    candidates = np.unique(values)
    shares = np.array([(values >= value).mean() for value in candidates])
    threshold = candidates[np.argmin(np.abs(shares - values.mean()))]
    return (scores >= threshold).astype(int)


def responses(results, score):
    """Return binary responses, one row per pipeline and one column per item."""
    means = results.groupby(["dataset", "subject", "pipeline"])[score].mean()
    binary = means.groupby("dataset", group_keys=False).apply(binarize)
    wide = binary.unstack("pipeline").T
    wide.columns = [f"{dataset}|{subject}" for dataset, subject in wide.columns]
    return wide


def fit(wide, model, epochs):
    """Fit py-irt and return pipeline abilities and per-item parameters."""
    data = Dataset.from_pandas(
        wide.rename_axis("pipeline").reset_index(),
        subject_column="pipeline",
        item_columns=list(wide.columns),
    )
    config = IrtConfig(model_type=model, epochs=epochs, log_every=epochs, seed=0)
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=data)
    trainer.train(epochs=epochs, device="cpu")
    params = trainer.last_params
    pipelines = [params["subject_ids"][i] for i in range(len(params["ability"]))]
    items = [params["item_ids"][i] for i in range(len(params["diff"]))]
    ability = pd.Series(params["ability"], index=pipelines, name="ability")
    item_params = pd.DataFrame(
        {"difficulty": params["diff"], "discrimination": params["disc"]},
        index=items,
    )
    item_params["dataset"] = item_params.index.str.split("|").str[0]
    return ability, item_params


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument("--model", default="2pl", help="py-irt model, e.g. 1pl")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--results", type=Path, help="Default: MOABB results root")
    return parser.parse_args(argv)


def main(argv=None):
    """Fit the IRT model and print its parameters."""
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
    ability, item_params = fit(responses(results, args.score), args.model, args.epochs)
    print(ability.sort_values(ascending=False).to_string())
    datasets = item_params.groupby("dataset")[["difficulty", "discrimination"]].mean()
    print(datasets.sort_values("difficulty").to_string(float_format="{:.2f}".format))


if __name__ == "__main__":
    main()
