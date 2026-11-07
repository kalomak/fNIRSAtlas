"""Aeon model integration for fNIRS classification."""

import inspect

from sklearn.pipeline import Pipeline

from fnirsatlas.feature_extraction import StandardScalerND


AEON_MODELS = (
    "aeon_minirocket",
    "aeon_multirocket",
    "aeon_multirockethydra",
)


def _get_aeon_models():
    from aeon.classification.convolution_based import (
        MiniRocketClassifier,
        MultiRocketClassifier,
        MultiRocketHydraClassifier,
    )

    return {
        "aeon_minirocket": MiniRocketClassifier,
        "aeon_multirocket": MultiRocketClassifier,
        "aeon_multirockethydra": MultiRocketHydraClassifier,
    }


def _filter_model_kwargs(model_cls, kwargs):
    params = inspect.signature(model_cls).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in params}


def create_aeon_model(model_name, **kwargs):
    """Create an aeon estimator for use in fnirsatlas benchmarks."""
    models = _get_aeon_models()
    model_cls = models[model_name]
    return model_cls(**_filter_model_kwargs(model_cls, kwargs))


def create_aeon_pipeline(model_name, **kwargs):
    """Create a minimal preprocessing pipeline for an aeon estimator."""
    return Pipeline(
        [
            ("scaler", StandardScalerND()),
            ("clf", create_aeon_model(model_name, **kwargs)),
        ]
    )
