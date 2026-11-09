"""Pipeline definition files load by path and classify array input."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

from fnirsatlas.benchmark import load_pipeline_file

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipelines"
ARRAY_PIPELINES = {
    "Windowed_LogReg.yml": "Windowed + LogReg",
    "Windowed_PCA_LDA.yml": "Windowed + PCA + LDA",
    "Windowed_RF.yml": "Windowed + Random Forest",
    "Windowed_RBF_SVM.yml": "Windowed + SVM-RBF",
    "Windowed_Linear_SVM.yml": "Windowed + SVM-Linear",
    "Windowed_ANN.yml": "Windowed + ANN",
    "multirockethydra.py": "MultiRocketHydra",
}
OPTIONAL_IMPORTS = {"multirockethydra.py": "aeon"}


def _toy_data():
    """Return what a paradigm emits, and labels."""
    X = np.random.default_rng(0).normal(size=(12, 2, 4, 40))
    y = np.array(["left", "right"] * 6)
    return X, y


@pytest.mark.parametrize(("filename", "name"), ARRAY_PIPELINES.items())
def test_pipeline_file_classifies_array_input(filename, name):
    if filename in OPTIONAL_IMPORTS:
        pytest.importorskip(OPTIONAL_IMPORTS[filename])
    X, y = _toy_data()

    (config,) = load_pipeline_file(PIPELINE_DIR / filename)
    pipeline = clone(config["pipeline"]).fit(X, y)

    assert config["name"] == name
    assert config["paradigms"] == ["FNIRSClassification"]
    assert set(pipeline.predict(X)) <= {"left", "right"}
    assert pipeline.predict(X).shape == (12,)


@pytest.mark.parametrize("filename", ["does_not_exist.yml", "does_not_exist.py"])
def test_missing_pipeline_file_raises(filename):
    with pytest.raises((ValueError, FileNotFoundError)):
        load_pipeline_file(PIPELINE_DIR / filename)
