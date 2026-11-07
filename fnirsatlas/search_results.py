import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from fnirsatlas.config import get_param_grid
from fnirsatlas.utils import build_result_path

logger = logging.getLogger(__name__)


class GridSearchResults:
    def __init__(self, npz_paths: List[Path], metric: str = "accuracy"):
        self.npz_paths = npz_paths
        self.metric = metric
        self.data = self._load()

    def _load(self) -> pd.DataFrame:
        all_rows = []

        for npz_path in self.npz_paths:
            data = np.load(npz_path, allow_pickle=True)
            cv_results = {key: data[key] for key in data.files}

            params = cv_results["params"]
            n_params = len(params)

            for i in range(n_params):
                row = {}
                param_dict = params[i]
                for k, v in param_dict.items():
                    row[f"param_{k}"] = tuple(v) if isinstance(v, list) else v

                fold_idx = int(npz_path.stem.split("_")[-1])
                row["fold"] = fold_idx

                score_key = f"mean_test_{self.metric}"
                std_key = f"std_test_{self.metric}"
                rank_key = f"rank_test_{self.metric}"

                if score_key in cv_results:
                    row["mean_score"] = cv_results[score_key][i]
                if std_key in cv_results:
                    row["std_score"] = cv_results[std_key][i]
                if rank_key in cv_results:
                    row["rank"] = cv_results[rank_key][i]

                split_cols = [
                    c
                    for c in cv_results
                    if c.startswith(f"split") and f"test_{self.metric}" in c
                ]
                for split_col in split_cols:
                    parts = split_col.split("_")
                    split_idx = int(parts[0].replace("split", ""))
                    row[f"split_{split_idx}"] = cv_results[split_col][i]

                all_rows.append(row)

        return pd.DataFrame(all_rows)

    def get_pivot_table(self, agg: str = "mean") -> pd.DataFrame:
        param_cols = [c for c in self.data.columns if c.startswith("param_")]

        if agg == "mean":
            agg_dict = {
                col: "mean"
                for col in self.data.columns
                if col.startswith("mean_") or col.startswith("split_")
            }
            agg_dict["rank"] = "mean"
            agg_dict["std_score"] = "mean"
        else:
            agg_dict = {
                col: agg
                for col in self.data.columns
                if col.startswith("mean_") or col.startswith("split_")
            }

        grouped = (
            self.data.groupby(param_cols + ["dataset"]).agg(agg_dict).reset_index()
        )

        pivot = grouped.pivot(index=param_cols, columns="dataset", values="mean_score")
        return pivot.round(4)

    def get_best_params(self) -> dict:
        best_rows = self.data.loc[self.data.groupby("dataset")["rank"].idxmin()]
        param_cols = [c for c in best_rows.columns if c.startswith("param_")]

        result = {}
        for _, row in best_rows.iterrows():
            dataset = row["dataset"]
            params = {c.replace("param_", ""): row[c] for c in param_cols}
            result[dataset] = params

        return result


def load_search_results(
    results_dir: Path,
    pattern: str,
    metric: str = "accuracy",
) -> GridSearchResults:
    npz_files = sorted(results_dir.glob(pattern))
    if not npz_files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")
    return GridSearchResults(npz_files, metric=metric)


def _load_block_search_rows(task_results: dict, metric: str) -> list:
    """Load rows from a block-strategy 'search' dict.

    scores shape: [subject, candidate, fold] (folds may vary per subject)
    Returns one row per candidate with mean_score averaged over subjects and folds.
    """
    search = task_results["search"]
    params = search["candidate_params"]
    scores_key = f"candidate_{metric}"
    if scores_key not in search:
        raise KeyError(f"No '{scores_key}' key in block search results")
    raw_scores = search[scores_key]  # list[subject][candidate][fold]
    rows = []
    for cand_ix, param_dict in enumerate(params):
        row = {
            f"param_{k}": tuple(v) if isinstance(v, list) else v
            for k, v in param_dict.items()
        }
        subject_means = [np.mean(raw_scores[s][cand_ix]) for s in range(len(raw_scores))]
        row["fold"] = 0
        row["mean_score"] = float(np.mean(subject_means))
        rows.append(row)
    return rows


def _load_nested_search_rows(task_results: dict, metric: str) -> list:
    """Load rows from a nested-strategy 'search_results' list."""
    raw = task_results["search_results"]
    all_rows = []
    for fold_data in raw:
        fold_idx = fold_data.get("fold")
        params = fold_data.get("params", [])
        for i, param_dict in enumerate(params):
            row = {
                f"param_{k}": tuple(v) if isinstance(v, list) else v
                for k, v in param_dict.items()
            }
            row["fold"] = fold_idx
            score_key = f"mean_test_{metric}"
            std_key = f"std_test_{metric}"
            rank_key = f"rank_test_{metric}"
            if score_key in fold_data:
                row["mean_score"] = fold_data[score_key][i]
            if std_key in fold_data:
                row["std_score"] = fold_data[std_key][i]
            if rank_key in fold_data:
                row["rank"] = fold_data[rank_key][i]
            split_cols = [
                c for c in fold_data if c.startswith("split") and f"test_{metric}" in c
            ]
            for split_col in split_cols:
                parts = split_col.split("_")
                split_idx = int(parts[0].replace("split", ""))
                row[f"split_{split_idx}"] = fold_data[split_col][i]
            all_rows.append(row)
    return all_rows


def load_search_results_from_json(
    task_results: dict,
    metric: str = "accuracy",
) -> "GridSearchResults":
    """Load GridSearchResults from a task results dict.

    Handles both 'search_results' (nested strategy) and 'search' (block strategy).
    """
    if "search" in task_results:
        all_rows = _load_block_search_rows(task_results, metric)
    elif "search_results" in task_results:
        all_rows = _load_nested_search_rows(task_results, metric)
    else:
        raise KeyError("No 'search_results' or 'search' key found in task results")

    result = GridSearchResults.__new__(GridSearchResults)
    result.npz_paths = []
    result.metric = metric
    result.data = pd.DataFrame(all_rows)
    return result


def normalize_param_value(value):
    """Convert values to JSON-stable Python types."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, tuple):
        return [normalize_param_value(v) for v in value]
    if isinstance(value, list):
        return [normalize_param_value(v) for v in value]
    return value


def order_param_cols(param_cols):
    """Keep learning rate as the first parameter level when present."""
    lr_cols = [col for col in param_cols if col.endswith("__lr")]
    other_cols = [col for col in param_cols if col not in lr_cols]
    return lr_cols + other_cols


def filter_to_configured_grid(data, model):
    """Drop rows with parameter values outside the configured grid."""
    param_grid = get_param_grid(model)
    if not param_grid:
        return data

    mask = pd.Series(True, index=data.index)
    for param_name, allowed_values in param_grid.items():
        param_col = f"param_{param_name}"
        if param_col not in data.columns:
            continue
        # Compare values, not JSON strings, so float columns match int grid values.
        allowed = [normalize_param_value(value) for value in allowed_values]
        mask &= data[param_col].map(lambda value: normalize_param_value(value) in allowed)

    filtered = data.loc[mask].copy()
    dropped = len(data) - len(filtered)
    if dropped:
        logger.warning(
            "Dropped %s out-of-grid search rows for model %s", dropped, model
        )
    return filtered


def _load_task_dicts(results_path, dataset):
    """Load task results dict, falling back to split files if needed."""
    if results_path.exists():
        return json.loads(results_path.read_text()).get(dataset, {})
    split_files = sorted(results_path.parent.glob("results_split*.json"))
    if not split_files:
        raise FileNotFoundError(results_path)
    merged = {}
    for split_path in split_files:
        split_data = json.loads(split_path.read_text()).get(dataset, {})
        for task, task_results in split_data.items():
            if task not in merged:
                merged[task] = dict(task_results)
            elif "search_results" in task_results:
                merged[task]["search_results"] += task_results["search_results"]
    return merged


def load_search_data(
    datasets, model, seeds, classification_setting, search_strategy, metric="kappa"
):
    """Load in-grid search rows for all datasets and seeds, or None if absent."""
    all_results = {}

    for dataset in datasets:
        for seed in seeds:
            results_path = build_result_path(
                classification_setting,
                model,
                dataset,
                [seed],
                search_strategy=search_strategy,
            )
            try:
                task_dicts = _load_task_dicts(results_path, dataset)
                combined_rows = []
                for task_results in task_dicts.values():
                    try:
                        gs_results = load_search_results_from_json(
                            task_results, metric=metric
                        )
                        combined_rows.append(gs_results.data)
                    except KeyError:
                        continue
                if not combined_rows:
                    logger.warning(f"No search results found in {results_path}")
                    continue
                data = pd.concat(combined_rows, ignore_index=True)
                data = filter_to_configured_grid(data, model)
                if data.empty:
                    logger.warning(
                        "No in-grid search results left in %s for model %s",
                        results_path,
                        model,
                    )
                    continue
                data["dataset"] = dataset
                data["seed"] = seed
                all_results[f"{dataset}_{seed}"] = data
            except FileNotFoundError:
                logger.warning(
                    f"No results file found for dataset {dataset} with seed {seed}"
                )

    if not all_results:
        return None
    return pd.concat(list(all_results.values()), ignore_index=True)


def get_param_cols(combined_data):
    """Return ordered `param_*` columns of search rows."""
    return order_param_cols(
        [c for c in combined_data.columns if c.startswith("param_")]
    )


def dataset_mean_scores(combined_data, param_cols):
    """Mean score per dataset and parameter combination."""
    return (
        combined_data.groupby(["dataset", *param_cols])["mean_score"]
        .mean()
        .reset_index()
    )


def best_params(combined_data):
    """Parameter combination with the highest mean score over datasets.

    Datasets are weighted equally, as in the heatmap's `Mean` row.
    Returns a dict of estimator parameter names to values.
    """
    param_cols = get_param_cols(combined_data)
    scores = dataset_mean_scores(combined_data, param_cols)
    means = scores.groupby(param_cols)["mean_score"].mean()
    best = means.idxmax()
    best = best if isinstance(best, tuple) else (best,)
    return {
        col.removeprefix("param_"): normalize_param_value(value)
        for col, value in zip(param_cols, best)
    }
