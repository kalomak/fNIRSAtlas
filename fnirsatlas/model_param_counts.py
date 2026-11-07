"""Parameter counts of searched models on a Bak2019-shaped reference input."""

import itertools

import numpy as np

from fnirsatlas.classification import get_baseline_model
from fnirsatlas.config import get_param_grid

REFERENCE_META = {
    "dataset": "bak2019",
    # 2250 epochs / 3 classes (docs/datasets/bak2019.md) -> 1500 LEFT/RIGHT epochs.
    "n_samples": 1500,
    "total_channels": 40,
    "n_chs": 20,
    "n_ch_types": 2,
    "n_timepoints": 76,
    "sfreq": 5.0,
    "n_classes": 2,
}


# Per-node arrays of a fitted sklearn tree.
TREE_NODE_ARRAYS = [
    "children_left",
    "children_right",
    "feature",
    "threshold",
    "value",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
]


def _size_and_nbytes(arrays):
    return sum(a.size for a in arrays), sum(a.nbytes for a in arrays)


def _count_linear(clf):
    return _size_and_nbytes([clf.coef_, clf.intercept_])


def _count_svm(clf):
    return _size_and_nbytes([clf.support_vectors_, clf.dual_coef_, clf.intercept_])


def _count_rf(clf):
    trees = [estimator.tree_ for estimator in clf.estimators_]
    n_bytes = sum(getattr(t, name).nbytes for t in trees for name in TREE_NODE_ARRAYS)
    return sum(t.node_count for t in trees), n_bytes


def _count_knn(clf):
    return _size_and_nbytes([clf._fit_X])


def _count_mlp(clf):
    return _size_and_nbytes(clf.coefs_ + clf.intercepts_)


def _count_aeon(clf):
    return _count_linear(clf.classifier)


def _count_torch(clf):
    params = [p for p in clf.parameters() if p.requires_grad]
    return sum(p.numel() for p in params), sum(p.numel() * p.element_size() for p in params)


# Models whose size is only known after fitting. All others are torch modules.
FITTED_COUNTERS = {
    "lda": _count_linear,
    "logreg": _count_linear,
    "svm": _count_svm,
    "rf": _count_rf,
    "knn": _count_knn,
    "ann": _count_mlp,
    "aeon_multirockethydra": _count_aeon,
}


def param_key(params):
    """Return a hashable key for a parameter dict that matches 1 and 1.0."""

    def hashable(value):
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (list, tuple)):
            return tuple(hashable(v) for v in value)
        return value

    return tuple(sorted((k, hashable(v)) for k, v in params.items()))


def reference_inputs():
    """Return random `(X, y, info)` shaped like the reference dataset."""
    meta = REFERENCE_META
    rng = np.random.default_rng(0)
    shape = (meta["n_samples"], meta["total_channels"], meta["n_timepoints"])
    X = rng.standard_normal(shape).astype("float32")
    y = np.arange(meta["n_samples"]) % meta["n_classes"]
    ch_names = [f"S{i:02d} {t}" for t in ("hbo", "hbr") for i in range(meta["n_chs"])]
    return X, y, {"ch_names": ch_names, "sfreq": meta["sfreq"]}


def count_model_params(model, params, X, y, info):
    """
    Count trainable parameters and their storage size for one model setting.

    Torch models count tensors with `requires_grad`. Fitted models count
    learned arrays: linear coefficients, SVM support vectors and dual
    coefficients, RF tree nodes, kNN stored training features. Bytes are
    measured from the arrays as fitted (RF: all per-node tree arrays).

    Parameters
    ----------
    model : str
        Model name accepted by `get_baseline_model`.
    params : dict
        Pipeline parameters, e.g. `{"clf__lr": 0.01}`.
    X, y, info
        Reference inputs from `reference_inputs`.

    Returns
    -------
    tuple[int, int]
        `(n_params, n_bytes)`.
    """
    pipe = get_baseline_model(model, X=X, y=y, info=info)
    pipe.set_params(**params)
    clf = pipe.named_steps["clf"]
    if model not in FITTED_COUNTERS:
        n_params, n_bytes = _count_torch(clf)
    else:
        pipe.fit(X, y)
        n_params, n_bytes = FITTED_COUNTERS[model](clf)
    return int(n_params), int(n_bytes)


def generate_param_counts(models):
    """
    Count parameters for every grid setting of each model.

    Returns
    -------
    dict
        `{"meta": REFERENCE_META, "models": {model: [{"params", "n_params", "n_bytes"}]}}`.
        Param keys carry the `param_` prefix of search results.
    """
    X, y, info = reference_inputs()
    counts = {}
    for model in models:
        grid = get_param_grid(model)
        settings = []
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            n_params, n_bytes = count_model_params(model, params, X, y, info)
            settings.append(
                {
                    "params": {f"param_{k}": v for k, v in params.items()},
                    "n_params": n_params,
                    "n_bytes": n_bytes,
                }
            )
        counts[model] = settings
    return {"meta": REFERENCE_META, "models": counts}
