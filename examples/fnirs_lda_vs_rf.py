"""
Compare LDA and Random Forest
=============================

Run a cross-subject comparison with meta-analysis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from moabb.analysis import compute_dataset_statistics, find_significant_differences
from moabb.analysis import plotting as moabb_plt
from moabb.datasets.base import CacheConfig
from moabb.evaluations import CrossSubjectEvaluation
from moabb.pipelines.utils import parse_pipelines_from_directory
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import (
    Bnci2014FnirsMA,
    FNIRSClassification,
    Gao2023FnirsSqueezing,
    Luke2021FnirsTapping,
    Nemo2024FnirsImagery,
    Nemo2024FnirsPerception,
    Shin2016FnirsMA,
    Shin2016FnirsMI,
)
from fnirsatlas.pipelines import WindowedFeatureExtractor

RESULTS_PATH = Path(__file__).resolve().parents[1]
CACHE_CONFIG = CacheConfig(use=True, save_epochs=True, save_array=True)
RF_CONFIG = parse_pipelines_from_directory(
    str(RESULTS_PATH / "pipelines" / "Windowed_RF.yml")
)[0]

DATASETS = {
    "bnci": Bnci2014FnirsMA,
    "luke": Luke2021FnirsTapping,
    "empe": Nemo2024FnirsPerception,
    "afim": Nemo2024FnirsImagery,
    "gao": Gao2023FnirsSqueezing,
    "shin_ma": lambda: Shin2016FnirsMA(accept=True),
    "shin_mi": lambda: Shin2016FnirsMI(accept=True),
}

PIPELINES = {
    "Windowed + LDA": make_pipeline(
        WindowedFeatureExtractor(),
        StandardScaler(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    ),
    RF_CONFIG["name"]: RF_CONFIG["pipeline"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="Datasets to run (default: all)",
    )
    return parser.parse_args()


def make_datasets(names=None):
    if names is None:
        names = list(DATASETS.keys())
    return [DATASETS[n]() for n in names]


def run(datasets):
    paradigm = FNIRSClassification()
    pipeline_names = set(PIPELINES.keys())
    dataset_codes = {d.code for d in datasets}
    all_results = []
    for dataset in datasets:
        evaluation = CrossSubjectEvaluation(
            paradigm=paradigm,
            datasets=[dataset],
            hdf5_path=str(RESULTS_PATH),
            cache_config=CACHE_CONFIG,
        )
        if not evaluation.is_valid(dataset):
            continue
        results = evaluation.process(PIPELINES)
        all_results.append(results)
    results = pd.concat(all_results, ignore_index=True)
    # Filter to only the pipelines/datasets we ran (cache may include others)
    mask = results["pipeline"].isin(pipeline_names) & results["dataset"].isin(
        dataset_codes
    )
    return results[mask].reset_index(drop=True)


def main():
    args = parse_args()
    datasets = make_datasets(args.datasets)
    results = run(datasets)
    # Per-dataset scores
    summary = (
        results.groupby(["dataset", "pipeline"])["score"].agg(["mean", "std"]).round(3)
    )
    print(summary.to_string())
    # Meta-analysis
    stats = compute_dataset_statistics(results)
    P, T = find_significant_differences(stats)
    print("\nMeta-effect (SMD):")
    print(T.to_string())
    print("\nP-values:")
    print(P.to_string())
    # Plots
    fig = moabb_plt.meta_analysis_plot(
        stats, "Windowed + LDA", "Windowed + Random Forest"
    )
    fig.savefig(RESULTS_PATH / "lda_vs_rf_meta.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
