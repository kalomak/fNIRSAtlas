from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fnirsatlas.classification import run_benchmark
from fnirsatlas.feature_extraction import WindowedFeatureExtractor


def get_clf(**kwargs):
    random_state = kwargs.pop("random_state", 42)
    return Pipeline(
        [
            ("extractor", WindowedFeatureExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_state, max_iter=100000)),
        ]
    )


results = run_benchmark(
    get_clf=get_clf,
    classification_setting="cross",
    suite="ma",
    seed=42,
    n_jobs=1,
    param_grid={
        "extractor__n_windows": [1, 6],  # Grid search supports feature extraction
        "clf__C": [1e-3, 1, 1e3],  # Regularization strength
    },
)
