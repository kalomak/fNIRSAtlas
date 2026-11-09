import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

pytest.importorskip("aeon")

from aeon.classification.convolution_based import RocketClassifier

from fnirsatlas.pipelines import AeonClassifier, StandardScalerND


def test_aeon_pipeline_fits_with_metadata_routing_enabled():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4, 32))
    y = np.array(["left", "right"] * 5)
    pipeline = make_pipeline(
        StandardScalerND(), AeonClassifier(RocketClassifier(n_kernels=20))
    )

    with config_context(enable_metadata_routing=True):
        fitted = clone(pipeline).fit(X, y)

    assert set(fitted[-1].classes_) == {"left", "right"}
    assert fitted.predict(X).shape == (10,)
    assert fitted.predict_proba(X).shape == (10, 2)
