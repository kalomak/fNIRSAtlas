"""
=============================
Compare LDA and Random Forest
=============================

Compare two pipelines with a cross-subject evaluation on several datasets and
combine the per-dataset results in a meta-analysis.
"""

# Authors: Kalle Mäkelä
#
# License: MIT

from moabb.analysis import compute_dataset_statistics, find_significant_differences
from moabb.analysis import plotting as moabb_plt
from moabb.evaluations import CrossSubjectEvaluation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fnirsatlas import (
    Bak2019FnirsTapping,
    Bnci2014FnirsMA,
    FNIRSClassification,
    Gao2023FnirsSqueezing,
    Luke2021FnirsTapping,
    Nemo2024FnirsPerception,
    Shin2016FnirsMA,
    Shin2018FnirsWG,
)
from fnirsatlas.pipelines import WindowedFeatureExtractor

# %%
# Define the pipelines
# --------------------
#
# Both pipelines extract windowed features from each epoch.

pipelines = {
    "Windowed + LDA": make_pipeline(
        WindowedFeatureExtractor(),
        StandardScaler(),
        LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"),
    ),
    "Windowed + Random Forest": make_pipeline(
        WindowedFeatureExtractor(n_windows=6),
        RandomForestClassifier(n_estimators=100),
    ),
}

# %%
# Evaluate
# --------
#
# The evaluation skips datasets that do not fit the paradigm or evaluation.
# Results are cached across runs and returned for every dataset already
# evaluated with these pipelines, so they are filtered to the selected datasets.

datasets = [
    Bnci2014FnirsMA(),
    Luke2021FnirsTapping(),
    Nemo2024FnirsPerception(),
    Gao2023FnirsSqueezing(),
    Shin2016FnirsMA(accept=True),
    Shin2018FnirsWG(),
    Bak2019FnirsTapping(),
]
evaluation = CrossSubjectEvaluation(paradigm=FNIRSClassification(), datasets=datasets)
results = evaluation.process(pipelines)
results = results[results["dataset"].isin([dataset.code for dataset in datasets])]

summary = results.groupby(["dataset", "pipeline"])["score"].agg(["mean", "std"])
print(summary.round(3).to_string())

# %%
# Meta-analysis
# -------------
#
# The standardized mean difference (SMD) between the pipelines is computed per
# dataset and combined into a meta-effect.

stats = compute_dataset_statistics(results)
p_values, effects = find_significant_differences(stats)
print(effects.to_string())
print(p_values.to_string())

fig = moabb_plt.meta_analysis_plot(stats, "Windowed + LDA", "Windowed + Random Forest")
