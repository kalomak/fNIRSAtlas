"""Compare within-subject accuracy under shuffled and blocked CV.

Every CV variant uses one fixed hyperparameter configuration per model from
`fnirsatlas/configs/best_params_within.yaml` (`scripts/best_params.py -c within`). Blocked
CV splitters come from the `fnirsatlas/configs/blocked_cv_gap*.yaml` overlays.

Examples:
    uv run python scripts/checks/blocked_cv.py -m lda -d bnci --seeds 0 --cv blocked_gap1
    uv run python scripts/checks/blocked_cv.py --summary -m lda -d bnci --seeds 0
    uv run python scripts/checks/blocked_cv.py --balance -d cockx2023
"""

import argparse
import json
import logging

import numpy as np
import yaml

from fnirsatlas.classification import (
    cross_validate,
    get_baseline_model,
    get_within_subject_splits,
)
from fnirsatlas.config import get_cv_config, get_results_cache_dir
from fnirsatlas.dataset_catalog import get_available_datasets
from fnirsatlas.io.dataset import fNIRSAtlasDataset
from fnirsatlas.logging import setup_logging
from fnirsatlas.utils import CONFIG_DIR, dict_to_json, map_values_to_sequential

setup_logging()
logger = logging.getLogger(__name__)

CV_CONFIGS = {
    "shuffled": None,
    "blocked_gap0": str(CONFIG_DIR / "blocked_cv_gap0.yaml"),
    "blocked_gap1": str(CONFIG_DIR / "blocked_cv_gap1.yaml"),
}
PARAMS_PATH = CONFIG_DIR / "best_params_within.yaml"


def iter_tasks(dataset):
    """Yield (description, dataset) with each classification task applied."""
    data = fNIRSAtlasDataset(dataset)
    for task in data.get_tasks():
        data.set_task({**task, "mapping": map_values_to_sequential(task["mapping"])})
        yield task["description"], data


def result_path(cv, model, dataset, seed):
    """Return the result file of one run."""
    return get_results_cache_dir() / "blocked_cv" / cv / model / dataset / f"{seed}.json"


def run_task(data, model, seed, spec, params, n_jobs):
    """Return per-subject fold scores of one task with fixed parameters."""
    clf_kwargs = {
        "model_name": model,
        "X": data.X,
        "y": data.y,
        "info": data.info,
        "classification_setting": "within",
        "random_state": seed,
    }
    acc, kappa = [], []
    for subject in np.unique(data.groups):
        mask = data.groups == subject
        splits = get_within_subject_splits(data.y[mask], seed, spec)
        if splits is None:
            logger.warning("Skipping subject %s: fewer than 2 splits", subject)
            continue
        results = cross_validate(
            data.X[mask], data.y[mask], splits, get_baseline_model, n_jobs=n_jobs,
            clf_kwargs=clf_kwargs, classification_setting="within",
            estimator_params=params,
        )
        acc.append(results["test_accuracy"])
        kappa.append(results["test_kappa"])
    return {"within_acc": acc, "within_kappa": kappa}


def run(model, dataset, seed, cv, n_jobs):
    """Run all tasks of one dataset with one CV variant and save the result."""
    path = result_path(cv, model, dataset, seed)
    if path.exists():
        logger.info("Result exists: %s", path)
        return
    params = yaml.safe_load(PARAMS_PATH.read_text())["params"][model]
    spec = get_cv_config(CV_CONFIGS[cv])["within"]
    tasks = {
        description: run_task(data, model, seed, spec, params, n_jobs)
        for description, data in iter_tasks(dataset)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict_to_json({"params": params, "tasks": tasks})))
    logger.info("Saved %s", path)


def mean_accuracy(path):
    """Return mean subject accuracy per task, or None if the result is missing."""
    if not path.exists():
        return None
    tasks = json.loads(path.read_text())["tasks"]
    return {
        task: float(np.mean([np.mean(folds) for folds in r["within_acc"]]))
        for task, r in tasks.items()
    }


def print_summary(model, datasets, seeds):
    """Print accuracy per task and seed for each CV variant."""
    print(f"| Dataset | Task | Seed | {' | '.join(CV_CONFIGS)} |")
    print("|---|---|---:|" + "---:|" * len(CV_CONFIGS))
    for dataset in datasets:
        for seed in seeds:
            accs = [mean_accuracy(result_path(cv, model, dataset, seed)) for cv in CV_CONFIGS]
            tasks = next((acc for acc in accs if acc is not None), {})
            for task in tasks:
                cols = " | ".join("-" if acc is None else f"{acc[task]:.4f}" for acc in accs)
                print(f"| {dataset} | {task} | {seed} | {cols} |")


def fold_balance(y, groups, spec):
    """Return class-balance diagnostics of within-subject folds."""
    classes = np.unique(y)
    tv, sizes = [], []
    missing_test = missing_train = 0
    for subject in np.unique(groups):
        subject_y = y[groups == subject]
        full = np.array([np.mean(subject_y == c) for c in classes])
        for train, test in get_within_subject_splits(subject_y, 0, spec) or []:
            test_counts = np.array([np.sum(subject_y[test] == c) for c in classes])
            train_counts = np.array([np.sum(subject_y[train] == c) for c in classes])
            missing_test += np.any(test_counts == 0)
            missing_train += np.any(train_counts == 0)
            tv.append(np.abs(test_counts / len(test) - full).sum() / 2)
            sizes.append(len(test))
    return (
        f"{min(sizes)}-{max(sizes)} | {missing_test}/{len(sizes)} | "
        f"{missing_train}/{len(sizes)} | {np.mean(tv):.3f} | {np.max(tv):.3f}"
    )


def print_balance(datasets):
    """Print blocked CV fold balance per dataset task and gap."""
    print("| Dataset | Task | CV | Test n | Missing test | Missing train | Mean TV | Max TV |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for dataset in datasets:
        for description, data in iter_tasks(dataset):
            for cv, config in CV_CONFIGS.items():
                if config is None:
                    continue
                row = fold_balance(data.y, data.groups, get_cv_config(config)["within"])
                print(f"| {dataset} | {description} | {cv} | {row} |", flush=True)


def main():
    """Run one model/dataset/seed/CV variant, or print a table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-m", "--model", default="lda", help="Model name (default: lda)")
    parser.add_argument(
        "-d", "--datasets", nargs="+", default=get_available_datasets(),
        help="Datasets (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[0], help="Seeds (default: 0)"
    )
    parser.add_argument(
        "--cv", choices=CV_CONFIGS, default="shuffled",
        help="CV variant to run (default: shuffled)",
    )
    parser.add_argument(
        "-j", "--n_jobs", type=int, default=1, help="Parallel CV folds (default: 1)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--summary", action="store_true",
        help="Print saved accuracy per task, seed and CV variant instead of running",
    )
    mode.add_argument(
        "--balance", action="store_true",
        help="Print blocked CV fold class balance per task instead of running",
    )
    args = parser.parse_args()
    if args.summary:
        print_summary(args.model, args.datasets, args.seeds)
    elif args.balance:
        print_balance(args.datasets)
    else:
        for dataset in args.datasets:
            for seed in args.seeds:
                run(args.model, dataset, seed, args.cv, args.n_jobs)


if __name__ == "__main__":
    main()
