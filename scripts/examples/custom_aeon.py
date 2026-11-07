"""Minimal aeon example.

Run with:
    uv run --with aeon --extra all python scripts/examples/custom_aeon.py
"""

from sklearn.pipeline import Pipeline
from fnirsatlas.classification import run_benchmark
from fnirsatlas.feature_extraction import StandardScalerND
from aeon.classification.convolution_based import MultiRocketHydraClassifier


def get_clf(**kwargs):
    return Pipeline(
        [
            ("scaler", StandardScalerND()),
            (
                "clf",
                MultiRocketHydraClassifier(
                    random_state=kwargs.get("random_state", 42),
                ),
            ),
        ]
    )


results = run_benchmark(
    get_clf=get_clf,
    suite="ma",
    classification_setting="cross",
    seed=42,
    n_jobs=1,
)
