"""
Model comparison utilities for statistical testing between models.
"""

import numpy as np
from pathlib import Path
import json
import logging
import pandas as pd
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel
from fnirsatlas.stats import corrected_t_test, bootstrap_se, cohens_d
from fnirsatlas.format import format_dataset_name
from fnirsatlas.config import get_cv_config
from fnirsatlas.utils import metric_to_subject_scores

logger = logging.getLogger(__name__)


def extract_model_metrics(results_dict, metric="cross_acc"):
    """Extract metrics and sample counts for statistical comparison.

    Returns:
        Tuple of (metrics, sample_counts) where both are dicts keyed by
        task key. sample_counts values are n_epochs from the results.
    """
    metrics = {}
    sample_counts = {}

    for dataset_name, dataset_results in results_dict.items():
        if not isinstance(dataset_results, dict):
            continue
        for task_desc, task_results in dataset_results.items():
            key = f"{dataset_name}__{task_desc}"
            if metric in task_results:
                metrics[key] = metric_to_subject_scores(task_results[metric]).tolist()
            if "n_epochs" in task_results:
                sample_counts[key] = task_results["n_epochs"]

    return metrics, sample_counts


def meta_effect(metrics_a, metrics_b, sample_counts, epsilon=1e-8):
    """Compute meta-effect: weighted average Cohen's d across tasks.

    The standardized mean difference is combined with a weighting
    given by the square root of the number of samples (n_epochs).

    Args:
        metrics_a: dict from extract_model_metrics (task_key -> list of scores)
        metrics_b: dict from extract_model_metrics (task_key -> list of scores)
        sample_counts: dict (task_key -> n_epochs)
        epsilon: small constant for numerical stability in Cohen's d
    Returns:
        Weighted average Cohen's d across all common tasks.
    """
    common_tasks = set(metrics_a) & set(metrics_b) & set(sample_counts)
    if not common_tasks:
        raise ValueError("No common tasks between the two metric dicts")
    weighted_d, total_w = 0.0, 0.0
    for task in common_tasks:
        a, b = np.array(metrics_a[task]), np.array(metrics_b[task])
        d = cohens_d(a, b, epsilon=epsilon)
        w = np.sqrt(sample_counts[task])
        weighted_d += w * d
        total_w += w
    return weighted_d / total_w


def compare_models(results_dicts, metric="cross_acc", alpha=0.05, one_sided=False):
    """Compare multiple models using corrected t-test.

    Args:
        results_dicts: List of results dictionaries
        metric: Metric to compare (default: "cross_acc")
        alpha: Significance level (default: 0.05)
        one_sided: If True, test if model1 > model2 (default: False)
    """
    model_names = []
    all_metrics = {}

    # Load all results
    for i, results in enumerate(results_dicts):
        # Get model name from results or generate one
        model_name = results.get("model_name", f"model_{i}")
        model_names.append(model_name)
        metrics, _ = extract_model_metrics(results, metric)
        all_metrics[model_name] = metrics

    # Find common tasks across all models
    common_tasks = set.intersection(
        *[set(metrics.keys()) for metrics in all_metrics.values()]
    )

    if not common_tasks:
        logger.error("No common tasks found across models")
        return {}

    # Perform pairwise comparisons
    comparisons = {}
    n_models = len(model_names)

    # Collect all p-values for FDR correction
    all_p_values = []
    p_value_info = []  # Store (comparison_key, task) tuples for mapping back

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1, model2 = model_names[i], model_names[j]
            comparison_key = f"{model1}_vs_{model2}"

            task_results = {}
            for task in common_tasks:
                values1 = all_metrics[model1][task]
                values2 = all_metrics[model2][task]
                diff_values = [v1 - v2 for v1, v2 in zip(values1, values2)]

                # Choose test based on metric type
                if metric.startswith("within_"):
                    alternative = "greater" if one_sided else "two-sided"
                    t_stat, p_value = ttest_rel(
                        values1, values2, alternative=alternative
                    )
                else:
                    cv_config = get_cv_config()
                    n_splits = cv_config["cross"]["kwargs"]["n_splits"]
                    n_test = 1 / n_splits
                    n_train = 1 - n_test
                    t_stat, p_value = corrected_t_test(
                        diff_values,
                        n_train=len(values1) * n_train,
                        n_test=len(values1) * n_test,
                        two_tailed=not one_sided,
                    )

                # Store p-value for FDR correction
                all_p_values.append(p_value)
                p_value_info.append((comparison_key, task))

                # Scale values to percentages for consistent reporting
                v1_scaled = np.array(values1) * 100
                v2_scaled = np.array(values2) * 100
                diff_scaled = np.array(diff_values) * 100

                task_results[task] = {
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "mean_diff": float(np.mean(diff_scaled)),
                    "se_diff": float(bootstrap_se(diff_scaled)),
                    "model1_mean": float(np.mean(v1_scaled)),
                    "model1_se": float(bootstrap_se(v1_scaled)),
                    "model2_mean": float(np.mean(v2_scaled)),
                    "model2_se": float(bootstrap_se(v2_scaled)),
                }

            comparisons[comparison_key] = task_results

    # Apply FDR correction to all p-values
    if all_p_values:
        _, p_values_corrected, _, _ = multipletests(
            all_p_values, alpha=alpha, method="fdr_bh"
        )

        # Update p-values and significance in comparisons
        for idx, (comparison_key, task) in enumerate(p_value_info):
            corrected_p_value = p_values_corrected[idx]
            comparisons[comparison_key][task]["p_value_corrected"] = float(
                corrected_p_value
            )
            comparisons[comparison_key][task]["significant"] = bool(
                corrected_p_value < alpha
            )

    return comparisons


def create_comparison_dataframe(
    comparison_key, task_results, alpha=0.05, metric="cross_acc"
):
    """Create pandas DataFrame for a single comparison result."""
    rows = []

    # Extract model names from comparison key
    if "_vs_" in comparison_key:
        model1, model2 = comparison_key.split("_vs_")
    else:
        model1, model2 = "Model1", "Model2"

    for task, results in task_results.items():
        # Format task name similar to summary table
        # Task keys are in format "dataset__taskdesc"
        parts = task.split("__")
        if len(parts) >= 2:
            dataset_id = "__".join(parts[:-1])
            clf_task_desc = parts[-1]
            dataset = format_dataset_name(dataset_id)
        else:
            dataset = task
            clf_task_desc = ""

        # Truncate task description to fit in 30 characters
        if len(clf_task_desc) > 28:
            clf_task_desc = clf_task_desc[:25] + "..."

        significance = ""
        p_value = results.get("p_value_corrected", results["p_value"])

        if results["significant"]:
            if p_value < alpha / 10:
                significance = "***"
            elif p_value < alpha / 5:
                significance = "**"
            else:
                significance = "*"

        mean_diff = results["mean_diff"]
        se_diff = results["se_diff"]
        model1_mean = results["model1_mean"]
        model1_se = results["model1_se"]
        model2_mean = results["model2_mean"]
        model2_se = results["model2_se"]
        p_original = results["p_value"]
        p_corrected = results.get("p_value_corrected", p_original)

        # Format values with ± se (already in percentages)
        model1_formatted = f"{model1_mean:.1f}±{model1_se:.1f}"
        model2_formatted = f"{model2_mean:.1f}±{model2_se:.1f}"
        diff_formatted = f"{mean_diff:.1f}±{se_diff:.1f}"

        rows.append(
            {
                "Dataset": dataset,
                "Task": clf_task_desc,
                "Model1_Perf": model1_formatted,
                "Model2_Perf": model2_formatted,
                "Diff": diff_formatted,
                "P_Orig": p_original,
                "P_FDR": p_corrected,
                "Significance": significance,
                "mean_diff": mean_diff,
            }
        )

    df = pd.DataFrame(rows)
    # Sort by p-value corrected (most significant first), break ties with mean difference
    df = df.sort_values(["P_FDR", "mean_diff"], ascending=[True, False])
    return df, model1, model2


def format_comparison_table_console(df, model1, model2, alpha=0.05, metric="cross_acc"):
    """Format comparison table for console output."""
    if df is None or df.empty:
        return

    model1_short = model1.split()[0].upper()
    model2_short = model2.split()[0].upper()

    print(f"\n{model1} vs {model2} {metric} comparison (α = {alpha})")
    print("=" * 120)
    print(
        f"{'Dataset':<16} {'Task':<30} {model1_short:>12} {model2_short:>12} {'Diff':>12} {'P-Orig':>8} {'P-FDR':>8} {'Sig':>3}"
    )
    print("-" * 120)

    # Print sorted rows
    for _, row in df.iterrows():
        print(
            f"{row['Dataset']:<16} {row['Task']:<30} {row['Model1_Perf']:>12} {row['Model2_Perf']:>12} {row['Diff']:>12} {row['P_Orig']:>8.3f} {row['P_FDR']:>8.3f} {row['Significance']:>3}"
        )


def format_comparison_table_latex(
    df, model1, model2, alpha=0.05, metric="cross_acc", output_path=None
):
    """Format comparison table for LaTeX output."""
    if df is None or df.empty:
        return

    # Prepare data for LaTeX table
    rows = []
    for _, row in df.iterrows():
        significance = row["Significance"]
        if significance:
            diff_formatted = row["Diff"] + significance
        else:
            diff_formatted = row["Diff"]

        rows.append(
            [
                row["Dataset"],
                row["Task"],
                row["Model1_Perf"],
                row["Model2_Perf"],
                diff_formatted,
                f"{row['P_FDR']:.3f}",
            ]
        )

    # Create DataFrame
    cols = ["Dataset", "Task", model1, model2, "Diff", "$p$"]
    df_latex = pd.DataFrame(rows, columns=cols)

    # Generate LaTeX using pandas styler
    latex_output = df_latex.style.to_latex()
    latex_output = latex_output.replace(r"[c]", "[t]")  # Match kappa_table.py format

    # Write to file if path provided, otherwise print
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(latex_output)
        logger.info(f"LaTeX table saved to {output_path}")
    else:
        print(latex_output)


def format_comparison_table(
    comparison_key, task_results, alpha=0.05, metric="cross_acc", output_path=None
):
    """Format and print comparison results."""
    df, model1, model2 = create_comparison_dataframe(
        comparison_key, task_results, alpha, metric
    )

    if output_path:
        format_comparison_table_latex(df, model1, model2, alpha, metric, output_path)
    else:
        format_comparison_table_console(df, model1, model2, alpha, metric)
