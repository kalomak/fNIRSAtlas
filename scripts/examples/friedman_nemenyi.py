"""Compare fNIRS pipelines with Friedman and post-hoc Nemenyi tests."""

import numpy as np
import pandas as pd
from moabb.evaluations import CrossSubjectEvaluation
from scipy.stats import friedmanchisquare, studentized_range
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fnirsatlas import FingerTapping
from fnirsatlas.pipelines import WindowedFeatureExtractor

ALPHA = 0.05
PIPELINES = {
    "Mean + LDA": make_pipeline(
        WindowedFeatureExtractor(),
        StandardScaler(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    ),
    "Mean/std + LDA": make_pipeline(
        WindowedFeatureExtractor(features=["mean", "std"]),
        StandardScaler(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    ),
    "Mean + SVM-RBF": make_pipeline(
        WindowedFeatureExtractor(), StandardScaler(), SVC(kernel="rbf")
    ),
}


def nemenyi(mean_ranks, n_datasets):
    """Return pairwise Nemenyi p-values and the critical difference."""
    n_pipelines = len(mean_ranks)
    standard_error = np.sqrt(n_pipelines * (n_pipelines + 1) / (6 * n_datasets))
    ranks = mean_ranks.to_numpy()
    differences = np.abs(np.subtract.outer(ranks, ranks))
    p_values = studentized_range.sf(
        differences / standard_error * np.sqrt(2), n_pipelines, np.inf
    )
    critical_value = studentized_range.ppf(1 - ALPHA, n_pipelines, np.inf) / np.sqrt(2)
    return pd.DataFrame(p_values, mean_ranks.index, mean_ranks.index), (
        critical_value * standard_error
    )


def main():
    """Run the benchmark and statistical comparison."""
    paradigm = FingerTapping()
    evaluation = CrossSubjectEvaluation(paradigm=paradigm, datasets=paradigm.datasets)
    results = evaluation.process(PIPELINES)
    scores = results[results["pipeline"].isin(PIPELINES)].pivot_table(
        index="dataset", columns="pipeline", values="score", aggfunc="mean"
    )
    if scores.isna().any().any():
        raise ValueError("Every pipeline must have a score for every dataset")

    statistic, p_value = friedmanchisquare(*(scores[name] for name in PIPELINES))
    mean_ranks = scores.rank(axis=1, ascending=False).mean().sort_values()
    nemenyi_p, critical_difference = nemenyi(mean_ranks, len(scores))
    print(f"Friedman: chi2={statistic:.3f}, p={p_value:.4g}")
    print("Mean ranks:\n", mean_ranks)
    print(f"Nemenyi critical difference (alpha={ALPHA}): {critical_difference:.3f}")
    print("Nemenyi p-values:\n", nemenyi_p)


if __name__ == "__main__":
    main()
