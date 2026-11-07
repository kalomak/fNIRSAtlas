import os
import sys
import threading
from collections import defaultdict
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"
MONTAGES_DIR = Path(__file__).parent / "montages"


def resolve_path(path_str: str) -> Path:
    """Return a path with `~` expanded. Relative paths use the current directory."""
    return Path(path_str).expanduser()


def deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override_dict taking precedence."""
    result = base_dict.copy()

    if not override_dict:
        return result

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def cluster_numbers(numbers, threshold=20):
    numbers = sorted(numbers)
    clusters = []
    current_cluster = [numbers[0]]
    for n in numbers[1:]:
        if n - current_cluster[-1] <= threshold:
            current_cluster.append(n)
        else:
            clusters.append(current_cluster)
            current_cluster = [n]
    clusters.append(current_cluster)
    return clusters


def cluster_means(numbers, threshold=20):
    clusters = cluster_numbers(numbers, threshold)
    return [np.mean(cluster) for cluster in clusters]


def check_duplicates(X):
    """Check if there are duplicate channels in data of shape (samples, ...)"""
    n = X.shape[0]
    first = X[None, 0].reshape(1, -1)
    rest = X[1:].reshape(n - 1, -1)
    matches = (first == rest).all(axis=1)
    return np.sum(matches, axis=0)


def override_raw_sfreq(raw, factor):
    """Overrides sfreq in info and annotation onset times"""
    with raw.info._unlock():
        raw.info["sfreq"] = raw.info["sfreq"] * factor
    raw.annotations.onset = raw.annotations.onset * (1 / factor)
    return raw


def check_data_exists(dataset_name: str) -> dict:
    """
    Check if processed and raw data exists for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to check

    Returns
    -------
    dict
        Dictionary with keys:
        - processed_exists: bool
        - raw_exists: bool
        - processed_path: Path
        - raw_path: Path
    """
    processed_path = get_processed_data_path(dataset_name)
    processed_exists = processed_path.exists()

    try:
        raw_path = get_raw_data_path(dataset_name)
        raw_exists = raw_path.is_file() or (raw_path.is_dir() and any(raw_path.iterdir()))
    except (ValueError, FileNotFoundError):
        raw_path = None
        raw_exists = False

    return {
        "processed_exists": processed_exists,
        "raw_exists": raw_exists,
        "processed_path": processed_path,
        "raw_path": raw_path,
    }


def get_raw_data_path(dataset_name: str) -> Path:
    """Return the raw data path for a dataset."""
    from fnirsatlas.config import get_dataset_path

    return get_dataset_path(dataset_name.split("__")[0])


def get_processed_data_path(dataset_name: str) -> Path:
    """Return the processed dataset path for a dataset."""
    from fnirsatlas.config import get_processed_datasets_dir

    return get_processed_datasets_dir() / f"{dataset_name}_processed.npz"


def map_values_to_sequential(d):
    """Map dictionary values to sequential integers 0,1,2... while preserving order."""
    unique_values = list(dict.fromkeys(d.values()))
    return {k: unique_values.index(v) for k, v in d.items()}


def normalize_seeds(seeds, default=42):
    if seeds is None:
        logger.debug(f"No seeds provided, using default seed: {default}")
        return [int(default)]
    if isinstance(seeds, (int, np.integer)):
        return [int(seeds)]
    return [int(s) for s in seeds]


def format_seeds(seeds, default=42):
    seeds_list = normalize_seeds(seeds, default=default)
    return "-".join(str(s) for s in seeds_list)


def _metric_rows(values):
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        return [[float(values)]]
    if not values:
        return []
    if isinstance(values[0], list):
        return [[float(v) for v in row] for row in values]
    return [[float(v)] for v in values]


def metric_to_subject_scores(values):
    rows = _metric_rows(values)
    return np.array([np.mean(row) for row in rows], dtype=float)


def metric_to_flat_scores(values):
    rows = _metric_rows(values)
    return np.array([value for row in rows for value in row], dtype=float)


def build_result_path(
    setting,
    model,
    dataset,
    seeds,
    split_ix=None,
    search_strategy="none",
):
    """Build a result file path from individual components."""
    from fnirsatlas.config import get_results_cache_dir

    search_strategy = search_strategy or "none"
    seed_str = format_seeds(seeds)
    split_suffix = f"_split{split_ix}" if split_ix is not None else ""
    filename = f"results{split_suffix}.json"
    return (
        get_results_cache_dir()
        / setting
        / search_strategy
        / model
        / dataset
        / seed_str
        / filename
    )


def build_output_path(args):
    """Build output path for classification results from args."""
    if args.output:
        return Path(args.output)
    if args.datasets:
        dataset = "_".join(args.datasets)
    elif args.suite:
        dataset = args.suite
    else:
        dataset = "all"
    return build_result_path(
        args.classification_setting,
        args.model,
        dataset,
        args.seed,
        args.split_ix,
        getattr(args, "search_strategy", None) or "none",
    )


def aggregate_classification_results(seed_results):
    """Aggregate classification results across multiple seeds."""
    aggregated_results = {}

    for task_desc in seed_results[0].keys():
        task_results = [sr[task_desc] for sr in seed_results]

        aggregated_task = {
            "n_classes": task_results[0]["n_classes"],
            "n_epochs": task_results[0]["n_epochs"],
            "mapping": task_results[0]["mapping"],
        }

        # Aggregate metrics
        for metric in [
            "cross_acc",
            "cross_kappa",
            "within_acc",
            "within_kappa",
        ]:
            if metric in task_results[0]:
                values = [
                    np.mean(metric_to_subject_scores(tr[metric])) for tr in task_results
                ]
                aggregated_task[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "values": values,
                }

        aggregated_results[task_desc] = aggregated_task

    return aggregated_results


def dict_to_json(data):
    """Convert any dictionary to JSON-serializable format."""
    if isinstance(data, dict):
        return {str(k): dict_to_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [dict_to_json(item) for item in data]
    elif hasattr(data, "dtype"):  # numpy arrays and scalars
        if data.ndim == 0:  # scalar
            return data.item()
        else:  # array
            return data.tolist()
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
        return data.item()
    else:
        return data


def get_non_overlapping_clf_tasks(task_ids: list[str]):
    """
    Returns a filtered list of task_ids that have no overlapping data with other tasks.
    We assume that all classification tasks of the same dataset have overlapping data, so we just use the first task.
    """
    from fnirsatlas.config import get_classification_tasks

    dataset_to_tasks = defaultdict(list)
    for task_id in task_ids:
        dataset_name, task_desc = task_id.rsplit("__", 1)
        dataset_to_tasks[dataset_name].append(task_desc)

    filtered = []
    for dataset_name, tasks in dataset_to_tasks.items():
        if len(tasks) == 1:
            default_task = tasks[0]
        else:
            default_task = get_classification_tasks(dataset_name)[0]["description"]
        filtered.append(f"{dataset_name}__{default_task}")
    return filtered


def filter_independent_task_ids(task_ids: list[str]):
    """Keep independent task ids and drop configured overlapping variants."""
    from fnirsatlas.config import load_config

    ordered_ids = sorted(task_ids)
    config_dropped_ids = []
    drops = load_config().get("task_selection", {}).get("drop_if_present", {})
    for keep_dataset, drop_prefixes in drops.items():
        if not isinstance(drop_prefixes, list):
            raise ValueError(
                f"task_selection.drop_if_present[{keep_dataset}] must be a list"
            )
        dataset_ids = [task_id.rsplit("__", 1)[0] for task_id in ordered_ids]
        if keep_dataset in dataset_ids:
            dropped_now = [
                task_id
                for task_id in ordered_ids
                if any(task_id.startswith(prefix) for prefix in drop_prefixes)
            ]
            config_dropped_ids.extend(dropped_now)
            ordered_ids = [
                task_id
                for task_id in ordered_ids
                if not any(task_id.startswith(prefix) for prefix in drop_prefixes)
            ]

    filtered_set = set(get_non_overlapping_clf_tasks(ordered_ids))
    filtered_ids = [task_id for task_id in ordered_ids if task_id in filtered_set]
    dropped_ids = [
        task_id
        for task_id in ordered_ids
        if task_id not in filtered_set and task_id not in config_dropped_ids
    ]
    dropped_ids = config_dropped_ids + dropped_ids
    return filtered_ids, dropped_ids


def get_tqdm_env_kwargs():
    kwargs = {}

    def _get_from_thread_or_env(var, default=None):
        if hasattr(threading.current_thread(), var.lower()):
            return getattr(threading.current_thread(), var.lower())
        elif os.environ.get(var.upper()):
            return os.environ.get(var.upper())
        return default

    kwargs["position"] = int(_get_from_thread_or_env("tqdm_position", 0))
    kwargs["leave"] = bool(_get_from_thread_or_env("tqdm_leave", True))
    kwargs["file"] = _get_from_thread_or_env("tqdm_file", None)
    if kwargs["file"] == "sys.stdout":
        kwargs["file"] = sys.stdout
    return kwargs


def to_plain(obj):
    """Convert nested structures to JSON-serializable plain Python types.

    - numpy scalars -> Python scalars
    - tuples -> lists
    - dict keys -> str
    """
    if type(obj) in (str, int, float, bool) or obj is None:
        return obj
    # Keep numpy arrays as-is; convert numpy scalars to Python scalars
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    # Try to coerce int/float-like custom objects (e.g., mne NamedInt) to plain scalars
    for caster in (int, float):
        try:
            return caster(obj)
        except Exception:
            pass
    return repr(obj)


def escape_latex(text):
    """Escape special LaTeX characters."""
    special_chars = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "^": r"\textasciicircum{}",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        # "[": r"\[",
        # "]": r"\]",
        "~": r"\textasciitilde{}",
        "×": r"\texttimes{}",
        # "\\": r"\textbackslash{}",
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


def get_datasets_from_args(datasets_arg, suite_arg):
    """Get dataset list from command line arguments."""
    from fnirsatlas.parsers import get_suite_datasets, get_available_datasets

    if datasets_arg:
        return datasets_arg
    elif suite_arg:
        return get_suite_datasets(suite_arg)
    else:
        return get_available_datasets()


def order_tasks_by_suite(task_keys):
    """Order task keys by suite order."""
    from fnirsatlas.parsers import SUITE_MAP, get_suite_for_dataset

    suite_order = list(SUITE_MAP.keys())
    suite_to_tasks = {suite: [] for suite in suite_order}
    suite_to_tasks[None] = []

    for task_key in task_keys:
        parts = task_key.split("__")
        dataset_id = "__".join(parts[:-1]) if len(parts) >= 2 else task_key
        suite = get_suite_for_dataset(dataset_id)
        suite_to_tasks[suite].append(task_key)

    ordered = []
    for suite in suite_order:
        ordered.extend(sorted(suite_to_tasks[suite]))
    ordered.extend(sorted(suite_to_tasks[None]))
    return ordered, suite_to_tasks


def recursive_update(dict1, dict2):
    """Recursive update of nested dictionaries."""
    for k, v in dict2.items():
        if k in dict1 and isinstance(dict1[k], dict) and isinstance(v, dict):
            recursive_update(dict1[k], v)
        else:
            dict1[k] = v
    return dict1
