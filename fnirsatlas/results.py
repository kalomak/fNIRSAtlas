import json
from pathlib import Path
from typing import List
import numpy as np
from fnirsatlas.parsers import get_suite_datasets, get_available_datasets, SUITE_MAP
from fnirsatlas.format import format_dataset_name
from fnirsatlas.stats import bootstrap_se, bootstrap_ci
from fnirsatlas.utils import (
    build_result_path,
    metric_to_subject_scores,
    normalize_seeds,
)
from fnirsatlas.logging import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)


class Results:
    """Class for loading and processing classification results."""

    def __init__(
        self,
        dataset,
        metric="cross_acc",
        model="lda",
        seeds=None,
        search_strategy="none",
        classification_setting=None,
        accept_partial=True,
    ):
        """Initialize Results with configuration."""
        self.dataset = dataset
        self.model = model
        self.seeds = normalize_seeds(seeds)
        self.metric = metric
        self.classification_setting = (
            classification_setting or Results._infer_classification_settings(metric)
        )
        self.search_strategy = search_strategy
        self.accept_partial = accept_partial
        self.file_path = None

    def __str__(self):
        """String representation of Results object."""
        return (
            "Results(dataset={dataset}, metric={metric}, model={model}, seeds={seeds})"
        ).format(
            dataset=self.dataset,
            metric=self.metric,
            model=self.model,
            seeds=self.seeds,
        )

    # TODO: remove this function, no more useful
    def get_file_path(self, file_setting=None):
        if file_setting is None:
            file_setting = self.classification_setting
        return Results.get_results_file_path(
            self.dataset,
            file_setting,
            self.model,
            self.seeds,
            self.search_strategy,
        )

    @staticmethod
    def get_results_file_path(dataset, setting, model, seeds, search_strategy="none"):
        return build_result_path(
            setting,
            model,
            dataset,
            seeds,
            search_strategy=search_strategy,
        )

    def _get_result_from_path(self, path):
        if not path.exists():
            return None
        result = json.load(open(path))
        if self.dataset not in result:
            return None
        task_dict = result[self.dataset]
        for task, r in task_dict.items():
            if self.metric in r.keys():
                return task_dict
        return None

    def load_single_seed(self, path):
        result = self._get_result_from_path(path)
        if result is not None:
            return result, path
        task_dict = self._load_split_files(path)
        return task_dict, f"merged(split files for {self.dataset})"

    def _load_multiple_seeds(self) -> dict:
        """Load and combine results from multiple seed files.

        Each seed file can fallback to split files if not found.
        """
        task_dict = None
        loaded_seeds = []
        for seed in self.seeds:
            single_seed_path = Results.get_results_file_path(
                self.dataset,
                self.classification_setting,
                self.model,
                [seed],
                self.search_strategy,
            )
            try:
                result_dict, _ = self.load_single_seed(single_seed_path)
            except FileNotFoundError:
                if not self.accept_partial:
                    raise
                continue
            if task_dict is None:
                task_dict = result_dict
            else:
                for task_name in task_dict:
                    for metric_key in [
                        "cross_acc",
                        "cross_kappa",
                        "within_acc",
                        "within_kappa",
                    ]:
                        if (
                            metric_key in task_dict[task_name]
                            and metric_key in result_dict[task_name]
                        ):
                            task_dict[task_name][metric_key] += result_dict[task_name][
                                metric_key
                            ]
            loaded_seeds.append(seed)
        if task_dict is None:
            raise FileNotFoundError(
                f"No results found for {self.dataset} {self.model} seeds {self.seeds}"
            )
        if len(loaded_seeds) != len(self.seeds):
            logger.warning(
                "Loaded partial results for %s %s: %s/%s seeds (%s)",
                self.dataset,
                self.model,
                len(loaded_seeds),
                len(self.seeds),
                loaded_seeds,
            )
        return task_dict

    def load(self, accept_partial=None):
        if accept_partial is not None:
            self.accept_partial = accept_partial
        if len(self.seeds) == 1:
            path = self.get_file_path()
            self.task_dict, self.file_path = self.load_single_seed(path)
            logger.info(f"Loaded results from: {self.file_path}")
        else:
            self.task_dict = self._load_multiple_seeds()
            self.file_path = f"merged(single seed files: {self.seeds})"
            logger.info(
                f"Loaded results from individual seed files for seeds {self.seeds}"
            )
        self.task2metric = {
            task: metric_to_subject_scores(r[self.metric])
            for task, r in self.task_dict.items()
        }
        return self

    def _load_split_files(self, base_path=None) -> dict:
        """Load split-specific files and combine them into full results."""
        task_dict = None
        split_ix = 0
        if base_path is None:
            base_path = self.get_file_path()

        def is_metric(k, v):
            return k.startswith(("cross_", "within_")) and isinstance(v, list)

        max_splits = 1000
        while True and split_ix < max_splits:
            split_path = base_path.parent / f"results_split{split_ix}.json"
            split_res = self._get_result_from_path(split_path)
            if split_res is None:
                if split_ix == 0:
                    raise FileNotFoundError(f"No split files found for {self.dataset}")
                break
            if task_dict is None:
                task_dict = split_res
            else:
                for task_name in task_dict:
                    for metric_key, metric_val in task_dict[task_name].items():
                        if is_metric(metric_key, metric_val):
                            task_dict[task_name][metric_key] += split_res[task_name][
                                metric_key
                            ]
            logger.info(f"Loaded split {split_ix} from: {split_path}")
            split_ix += 1
        return task_dict

    def compute_stats(self, return_ci=False, confidence_level=0.95):
        """Compute statistics for all tasks.

        Args:
            return_ci: If True, return confidence intervals instead of standard error
            confidence_level: Confidence level for CI/SE calculation

        Returns:
            Dictionary with task statistics
        """
        stats = {}
        for task, cv_scores in self.task2metric.items():
            stats[task] = {}
            mu = np.mean(cv_scores)
            stats[task]["mean"] = float(mu)
            if return_ci:
                ci = bootstrap_ci(cv_scores, confidence_level=confidence_level)
                stats[task]["ci_lower"] = float(ci[0])
                stats[task]["ci_upper"] = float(ci[1])
            else:
                se = bootstrap_se(cv_scores, confidence_level=confidence_level)
                stats[task]["se"] = float(se)
        return stats

    @staticmethod
    def gather_results(
        models,
        datasets,
        metric="cross_acc",
        seeds=None,
        search_strategy="none",
        classification_setting=None,
        accept_partial=True,
    ):
        """Gather results from multiple models and datasets into a list of result dictionaries.

        Args:
            models: List of model names to compare
            datasets: List of dataset names to include
        Returns:
            List of result dictionaries, one per model, with model_name included
        """
        results_dicts = []
        for model in models:
            model_results = {}
            for dataset in datasets:
                try:
                    results = Results(
                        dataset=dataset,
                        metric=metric,
                        model=model,
                        seeds=seeds,
                        search_strategy=search_strategy,
                        classification_setting=classification_setting,
                        accept_partial=accept_partial,
                    )
                    results.load()
                    model_results[dataset] = results.task_dict
                except FileNotFoundError:
                    logger.warning(f"No results found for {dataset} {model} {metric}")
                    continue
                except Exception as e:
                    logger.error(f"Error loading results for {dataset} {model}: {e}")
                    continue
            if model_results:
                model_results["model_name"] = model
                results_dicts.append(model_results)
            else:
                logger.warning(f"No results found for model {model}")
        return results_dicts

    @staticmethod
    def save_to_separate_files(
        results_dict,
        setting,
        seeds=None,
        model_name=None,
        search_strategy="none",
        override=False,
    ):
        """Save results_dict with multiple datasets to separate json files"""
        info = {"wrote": [], "skipped": []}
        model_name = model_name or results_dict.pop("model_name", None)
        if model_name is None:
            raise ValueError("Provide model_name")
        seed_list = normalize_seeds(seeds)
        for dataset_name, d in results_dict.items():
            new_path = Results.get_results_file_path(
                dataset_name,
                setting,
                model_name,
                seed_list,
                search_strategy,
            )
            if not new_path.exists() or override:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                dd = {dataset_name: d}
                new_path.write_text(json.dumps(dd, indent=2))
                info["wrote"].append(new_path)
            else:
                info["skipped"].append(new_path)
        return info

    @staticmethod
    def _infer_classification_settings(metric):
        """Infer classification setting from metric name."""
        if metric.startswith("cross_"):
            return "cross"
        elif metric.startswith("within_"):
            return "within"
        return None
