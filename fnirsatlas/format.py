"""Benchmark utilities for classification results formatting and aggregation."""

import numpy as np
from fnirsatlas.utils import metric_to_subject_scores


# Column width for formatted metric values, e.g. "100.0±10.0".
VALUE_WIDTH = 10

PAPER_MODEL_LABELS = {
    "lda": "LDA",
    "logreg": "LR",
    "svm": "SVM",
    "rf": "RF",
    "knn": "kNN",
    "ann": "ANN",
    "cnn2a": "CNN",
    "lstm": "LSTM",
    "eegnet": "EEGNet",
    "dcn": "DCN",
    "aeon_multirockethydra": "MRH",
    "fnirst": "fNIRS-T",
    "fnirsnet23": "fNIRSNet23",
    "fnirsnet24": "fNIRSNet24",
}


def format_dataset_name(name: str) -> str:
    """Format dataset name for display."""
    parts = name.split("__")
    return (
        f"{parts[0].capitalize()} ({parts[1]})" if len(parts) > 1 else name.capitalize()
    )


def format_score(val, width):
    """Format score for display."""
    return f"{val * 100:>{width}.1f}" if not np.isnan(val) else "nan"


def _extract_metric(task_results, metric_name):
    """Extract metric mean and std over score rows (subjects or folds, times seeds)."""
    if metric_name not in task_results:
        return np.nan, np.nan

    values = task_results[metric_name]
    if isinstance(values, list):
        v = metric_to_subject_scores(values) * 100
        return np.mean(v), np.std(v)
    if isinstance(values, dict):
        return values["mean"] * 100, values["std"] * 100
    return values * 100, 0


def _format_metric_value(value, std, has_multiple_seeds):
    """Format a single metric value with optional std."""
    if np.isnan(value):
        return "nan"
    elif has_multiple_seeds and std > 0:
        return f"{value:.1f}±{std:.1f}"
    else:
        return f"{value:.1f}"


def _create_summary_row(dataset_name, task_desc, task_results, metrics):
    """Create a summary row for a dataset-task combination."""
    metric_data = {}
    for metric in metrics:
        value, std = _extract_metric(task_results, metric)
        metric_data[metric] = {"value": value, "std": std}

    has_multiple_seeds = any(
        not np.isnan(metric_data[m]["std"]) and metric_data[m]["std"] > 0 for m in metrics
    )

    return {
        "dataset": format_dataset_name(dataset_name),
        "task": task_desc,
        "n_classes": task_results["n_classes"],
        "n_epochs": task_results["n_epochs"],
        "has_multiple_seeds": has_multiple_seeds,
        **metric_data,
    }


def _get_metric_header(metric):
    """Get formatted header for a metric."""
    headers = {
        "cross_acc": "Cross%",
        "within_acc": "Within%",
        "cross_kappa": "Cross κ",
        "within_kappa": "Within κ",
    }
    return headers.get(metric, metric)


def _print_header(model_name, metrics):
    """Print the table header."""
    model_label = f" {model_name}" if model_name else ""
    print(f"\nSummary{model_label}:")
    metric_headers = " ".join(f"{_get_metric_header(metric):>{VALUE_WIDTH}}" for metric in metrics)
    print(
        f"{'dataset':<16} {'task':<30} {'#classes':>8} {'#epochs':>8} {metric_headers}"
    )
    print("-" * 111)


def _print_row(row, metrics):
    """Print a single data row."""
    formatted_metrics = {
        metric: _format_metric_value(
            row[metric]["value"], row[metric]["std"], row["has_multiple_seeds"]
        )
        for metric in metrics
    }

    metric_values = " ".join(f"{formatted_metrics[metric]:>{VALUE_WIDTH}}" for metric in metrics)
    print(
        f"{row['dataset']:<16} {row['task'][:28]:<30} {row['n_classes']:>8} "
        f"{row['n_epochs']:>8} {metric_values}"
    )


def _print_means(summary, metrics):
    """Print the mean row."""
    if not summary:
        return

    # Calculate means
    mean_data = {}
    for metric in metrics:
        values = [
            row[metric]["value"]
            for row in summary
            if not np.isnan(row[metric]["value"])
        ]
        stds = [row[metric]["std"] for row in summary if not np.isnan(row[metric]["std"])]

        if values:
            mean_data[metric] = {
                "value": np.mean(values),
                "std": np.mean(stds) if stds else 0,
            }
        else:
            mean_data[metric] = {"value": np.nan, "std": np.nan}

    has_any_multiple_seeds = any(row["has_multiple_seeds"] for row in summary)

    # Format and print means
    mean_formatted = {
        metric: _format_metric_value(
            mean_data[metric]["value"],
            mean_data[metric]["std"],
            has_any_multiple_seeds,
        )
        for metric in metrics
    }

    print("-" * 111)
    metric_values = " ".join(f"{mean_formatted[metric]:>{VALUE_WIDTH}}" for metric in metrics)
    print(f"{'MEAN':<16} {'':<30} {'':>8} {'':>8} {metric_values}")


def get_task_labels(dataset, task_dict):
    """Get formatted task labels for a dataset."""
    task_labels = []
    for task_key in task_dict.keys():
        dataset_name = format_dataset_name(dataset)
        label = f"{dataset_name} ({task_key})"
        task_labels.append((f"{dataset}__{task_key}", label))
    return task_labels


def format_benchmark_table(results_dict, setting="cross"):
    """Format and print benchmark results table with percentages and 1 decimal place."""
    metrics = ["cross_acc", "within_acc", "cross_kappa", "within_kappa"]
    if setting == "cross":
        metrics = [m for m in metrics if "cross" in m]
    if setting == "within":
        metrics = [m for m in metrics if "within" in m]

    model_name = results_dict.pop("model_name", "")

    # Create summary rows
    summary = [
        _create_summary_row(dataset_name, task_desc, task_results, metrics)
        for dataset_name, dataset_results in results_dict.items()
        for task_desc, task_results in dataset_results.items()
    ]

    # Print table
    _print_header(model_name, metrics)
    for row in summary:
        _print_row(row, metrics)
    _print_means(summary, metrics)


METRIC_LABELS = {
    "cross_kappa": "Cross $\\kappa$",
    "within_kappa": "Within $\\kappa$",
    "cross_acc": "Cross Acc",
    "within_acc": "Within Acc",
}


def format_metric(metric):
    return METRIC_LABELS.get(metric, metric)


def format_paper_model_name(model_name: str) -> str:
    """Format a paper model name for compact display."""
    return PAPER_MODEL_LABELS.get(model_name, model_name)


def _collect_all_tasks(results_dicts):
    """Return ordered list of (dataset, task) pairs across all models."""
    seen = {}
    for rd in results_dicts:
        for key, val in rd.items():
            if key == "model_name":
                continue
            for task_desc in val.keys():
                seen[(key, task_desc)] = val[task_desc]
    return seen


LATEX_INFO_COLUMNS = {
    "dataset": lambda dataset, task_desc, results: format_dataset_name(dataset),
    "task": lambda dataset, task_desc, results: task_desc,
    "classes": lambda dataset, task_desc, results: str(results["n_classes"]),
    "epochs": lambda dataset, task_desc, results: str(results["n_epochs"]),
}


def format_model_comparison_table(
    results_dicts, metric="cross_acc", latex=False, columns=tuple(LATEX_INFO_COLUMNS)
):
    """Print one-metric table with columns per model.

    Args:
        results_dicts: list of result dicts from Results.gather_results
        metric: single metric name to display
        latex: if True, format as LaTeX rows with & and \\
        columns: keys of LATEX_INFO_COLUMNS shown before the model columns in LaTeX rows
    """
    models = [rd["model_name"] for rd in results_dicts]
    task_data = _collect_all_tasks(results_dicts)
    model_lookup = {rd["model_name"]: rd for rd in results_dicts}
    col_means = {m: [] for m in models}

    if not latex:
        model_headers = " ".join(f"{m:>{VALUE_WIDTH}}" for m in models)
        print(f"\n{metric} comparison:")
        print(f"{'dataset':<16} {'task':<30} {'#cls':>5} {'#ep':>6} {model_headers}")
        print("-" * (16 + 1 + 30 + 1 + 5 + 1 + 6 + 1 + (VALUE_WIDTH + 1) * len(models)))

    for (dataset, task_desc), ref_results in task_data.items():
        n_classes = ref_results["n_classes"]
        n_epochs = ref_results["n_epochs"]
        raw_vals = []
        values = []
        for model in models:
            rd = model_lookup.get(model, {})
            task_results = rd.get(dataset, {}).get(task_desc)
            if task_results and metric in task_results:
                val, std = _extract_metric(task_results, metric)
                col_means[model].append(val)
                raw_vals.append(val)
                values.append(_format_metric_value(val, std, std > 0))
            else:
                raw_vals.append(np.nan)
                values.append("N/A")
        if latex:
            best = np.nanmax(raw_vals) if any(not np.isnan(v) for v in raw_vals) else None
            bold_values = [
                rf"\textbf{{{v}}}" if best is not None and rv == best and not np.isnan(rv) else v
                for v, rv in zip(values, raw_vals)
            ]
            info_cols = [LATEX_INFO_COLUMNS[c](dataset, task_desc, ref_results) for c in columns]
            cols = info_cols + bold_values
            print(" & ".join(cols) + r" \\")
        else:
            model_cols = " ".join(f"{v:>{VALUE_WIDTH}}" for v in values)
            print(
                f"{format_dataset_name(dataset):<16} {task_desc[:28]:<30} "
                f"{n_classes:>5} {n_epochs:>6} {model_cols}"
            )

    mean_cols_vals = [
        f"{np.mean(col_means[m]):.1f}" if col_means[m] else "N/A"
        for m in models
    ]
    if latex:
        print(r"\midrule")
        mean_nums = [float(v) if v != "N/A" else np.nan for v in mean_cols_vals]
        best_mean = np.nanmax(mean_nums) if any(not np.isnan(v) for v in mean_nums) else None
        bold_means = [
            rf"\textbf{{{v}}}" if best_mean is not None and float(v) == best_mean and v != "N/A" else v
            for v in mean_cols_vals
        ]
        print(" & ".join(["Mean"] + [""] * (len(columns) - 1) + bold_means) + r" \\")
    else:
        print("-" * (16 + 1 + 30 + 1 + 5 + 1 + 6 + 1 + (VALUE_WIDTH + 1) * len(models)))
        mean_cols = " ".join(f"{v:>{VALUE_WIDTH}}" for v in mean_cols_vals)
        print(f"{'MEAN':<16} {'':<30} {'':>5} {'':>6} {mean_cols}")
