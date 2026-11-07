#!/usr/bin/env python3
"""Run subject-count learning curves."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import GroupKFold

from fnirsatlas.classification import get_baseline_model
from fnirsatlas.config import (
    get_results_cache_dir,
    get_paper_models,
    get_processed_datasets_dir,
)
from fnirsatlas.io.dataset import fNIRSAtlasDataset
from fnirsatlas.utils import CONFIG_DIR, dict_to_json


MODELS = get_paper_models()
PARAMS_PATH = CONFIG_DIR / "best_params_cross.yaml"
METRICS = {
    "accuracy": accuracy_score,
    "kappa": cohen_kappa_score,
    "balanced_accuracy_adjusted": partial(balanced_accuracy_score, adjusted=True),
}


def run(dataset, model, seed, fold, sizes, n_folds, device, root):
    """Run all training sizes for one dataset, seed, and outer fold."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in [0, {n_folds - 1}]")
    size_key = "-".join(map(str, sizes))
    name = f"seed-{seed}-fold-{fold}-sizes-{size_key}-cv-{n_folds}.json"
    path = root / dataset / model / name
    if path.exists():
        return

    data = fNIRSAtlasDataset(get_processed_datasets_dir() / f"{dataset}_processed.npz")
    data.set_task()
    X, y, groups = data.as_tuple()[:3]
    splits = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train, test = list(splits.split(X, y, groups))[fold]
    subjects = np.unique(groups[train])
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
    subjects = subjects[rng.permutation(len(subjects))]
    largest = max(sizes)
    if len(subjects) < largest:
        raise ValueError(f"{dataset} has only {len(subjects)} training subjects")

    params = yaml.safe_load(PARAMS_PATH.read_text())["params"][model]
    rows = []
    for size in sizes:
        if largest % size:
            raise ValueError(f"training size {size} does not divide {largest}")
        for subset, selected in enumerate(np.split(subjects[:largest], largest // size)):
            fit = train[np.isin(groups[train], selected)]
            clf = get_baseline_model(
                model_name=model, random_state=seed, classification_setting="cross",
                X=X[fit], y=y[fit], info=data.info, device=device,
            )
            clf.set_params(**params)
            clf.fit(X[fit], y[fit])
            y_pred = clf.predict(X[test])
            rows.append({
                "dataset": dataset, "model": model, "seed": seed, "fold": fold,
                "size": str(size), "subset": subset, "n_train": len(selected),
                **{name: metric(y[test], y_pred) for name, metric in METRICS.items()},
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict_to_json(rows)))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--folds", nargs="+", type=int, help="Outer folds to run (default: all)"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--root", type=Path, default=get_results_cache_dir() / "learning_curve"
    )
    args = parser.parse_args()
    for fold in args.folds or range(args.n_folds):
        run(args.dataset, args.model, args.seed, fold, args.sizes,
            args.n_folds, args.device, args.root)


if __name__ == "__main__":
    main()
