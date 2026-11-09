"""Test dl pipelines"""

from pathlib import Path

import mne
import numpy as np
import pytest
from sklearn.base import clone

from fnirsatlas.benchmark import load_pipeline_file
from fnirsatlas.paradigms.fnirs import _split_chromophores

pytest.importorskip("torch")


@pytest.mark.parametrize(
    ("filename", "name"),
    [
        ("eegnet.py", "EEGNet"),
        ("lstm.py", "LSTM"),
        ("cnn2a.py", "CNN2a"),
        ("dcn.py", "DCN"),
        ("fnirs_transformer.py", "fNIRS-T"),
        ("fnirsnet23.py", "fNIRSNet23"),
        ("fnirsnet24.py", "fNIRSNet24"),
    ],
)
@pytest.mark.parametrize("n_times", [64, 76])
def test_model_pipeline_fits(filename, name, n_times):
    path = Path(__file__).parents[1] / "pipelines" / "dl" / filename
    (config,) = load_pipeline_file(path)
    assert config["name"] == name
    pipeline = clone(config["pipeline"]).set_params(
        fnirsclassifier__max_epochs=1,
        fnirsclassifier__batch_size=2,
        fnirsclassifier__device="cpu",
    )
    names = [f"S{ch}_D{ch} {kind}" for ch in range(1, 5) for kind in ("hbo", "hbr")]
    info = mne.create_info(names, 5.0, ["hbo", "hbr"] * 4)
    epochs = mne.EpochsArray(
        np.random.default_rng(0).normal(size=(5, 8, n_times)), info, verbose=False
    )
    X = _split_chromophores(epochs)
    labels = np.array(["left", "right", "rest", "left", "right"])

    pipeline.fit(X, labels)
    probabilities = pipeline.predict_proba(X)

    assert X.shape == (5, 2, 4, n_times)
    assert probabilities.shape == (5, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert pipeline.predict(X[:1]).shape == (1,)
