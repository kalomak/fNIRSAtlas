"""
Example: Compare custom model to LDA baseline.

This example shows how to:
1. Run a custom model and save results
2. Compare to LDA baseline with corrected t-test

Prerequisites:
- Run LDA baseline first: python scripts/classify_dataset.py -m lda --seed 42
  (creates `all_ccross_mlda_s42.json` in `paths.results_cache`)

Results Interpretation:
- The comparison table shows performance differences between Custom SVM RBF and LDA Baseline
- "Diff" column shows the accuracy difference (Custom - LDA)
- "P-Orig" shows uncorrected p-values from paired t-tests
- "P-FDR" shows False Discovery Rate corrected p-values (Benjamini-Hochberg)
- "Sig" shows significance levels: *** (p < 0.001), ** (p < 0.01), * (p < 0.05)
- Negative differences indicate LDA performs better than Custom SVM
- The results show that the LDA baseline significantly outperforms the Custom SVM across all datasets
"""

import json
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from fnirsatlas.feature_extraction import WindowedFeatureExtractor
from fnirsatlas.classification import run_benchmark
from fnirsatlas.compare import compare_models, format_comparison_table


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


seed = 42
classification_setting = "cross"

# Check if LDA baseline exists
from fnirsatlas.config import get_results_cache_dir
lda_file = get_results_cache_dir() / f"all_c{classification_setting}_mlda_s{seed}.json"
if not lda_file.exists():
    raise FileNotFoundError(
        "LDA baseline not found!\nPlease run first:\n"
        "    python scripts/classify_dataset.py -m lda --seed 42"
    )

print("Running custom SVM with RBF kernel model...")
custom_results = run_benchmark(
    get_clf=get_custom_svm,
    suite="ma",
    classification_setting="cross",
    seed=seed,
    n_jobs=1,
    param_grid={
        "clf__C": [1e-3, 1, 1e3],
    },
)
custom_results["model_name"] = "Custom SVM RBF"

# Load LDA baseline results
with open(lda_file, "r") as f:
    lda_results = json.load(f)
lda_results["model_name"] = "LDA Baseline"

# Compare to LDA baseline
comparisons = compare_models(
    [custom_results, lda_results], metric="cross_acc", alpha=0.05
)
format_comparison_table(comparisons, alpha=0.05, metric="cross_acc")

# Save custom model results
custom_file = get_results_cache_dir() / f"ma_c{classification_setting}_custom_svm_s{seed}.json"
with open(custom_file, "w") as f:
    json.dump(custom_results, f, indent=2)

print(f"\nResults files saved:")
print(f"  Custom SVM: {custom_file}")
print(f"  LDA Baseline: {lda_file}")
print(f"\nTo re-run comparison later:")
print(f"  python scripts/compare_models.py {custom_file} {lda_file}")
