"""
Example: Compare custom model to LDA baseline.

This example shows how to:
1. Run a custom model
2. Compare it to saved LDA baseline results with the corrected t-test

Prerequisites:
- Run the LDA baseline first, from the repository root:
    uv run python scripts/classify_dataset.py -m lda --suite ma --seed 42

Results Interpretation:
- "Diff" column shows the accuracy difference between the two models
- "P-Orig" shows uncorrected p-values from corrected paired t-tests
- "P-FDR" shows False Discovery Rate corrected p-values (Benjamini-Hochberg)
- "Sig" shows significance levels: *** (p < 0.001), ** (p < 0.01), * (p < 0.05)
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fnirsatlas.classification import run_benchmark
from fnirsatlas.compare import compare_models, format_comparison_table
from fnirsatlas.feature_extraction import WindowedFeatureExtractor
from fnirsatlas.results import Results
from fnirsatlas.utils import get_datasets_from_args


def get_custom_svm(**kwargs):
    """Custom SVM with RBF kernel model."""
    random_state = kwargs["random_state"]
    return Pipeline(
        [
            (
                "feature_extractor",
                WindowedFeatureExtractor(n_windows=1, features=["std"]),
            ),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(kernel="rbf", random_state=random_state, C=1.0, gamma="scale"),
            ),
        ]
    )


suite = "ma"
seed = 42
metric = "cross_acc"

# Load saved LDA baseline results
lda_results = Results.gather_results(
    models=["lda"],
    datasets=get_datasets_from_args(None, suite),
    metric=metric,
    seeds=[seed],
    classification_setting="cross",
)
if not lda_results:
    raise FileNotFoundError(
        "LDA baseline not found. Run first:\n"
        f"    uv run python scripts/classify_dataset.py -m lda --suite {suite} --seed {seed}"
    )

print("Running custom SVM with RBF kernel model...")
custom_results = run_benchmark(
    get_clf=get_custom_svm,
    suite=suite,
    classification_setting="cross",
    seed=seed,
    n_jobs=1,
    param_grid={
        "clf__C": [1e-3, 1, 1e3],
    },
)
custom_results["model_name"] = "Custom SVM RBF"

# Compare to LDA baseline
comparisons = compare_models([custom_results, *lda_results], metric=metric, alpha=0.05)
for comparison_key, task_results in comparisons.items():
    format_comparison_table(comparison_key, task_results, alpha=0.05, metric=metric)
