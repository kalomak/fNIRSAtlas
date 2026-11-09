"""Tests for the sklearn-compatible fNIRS neural classifier."""

import mne
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

torch = pytest.importorskip("torch")
from torch import nn

from fnirsatlas.pipelines.dl import FNIRSClassifier


class TinyNet(nn.Module):
    def __init__(self, n_channels, n_times, n_outputs, n_chromophores=1, hidden=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.n_chromophores = n_chromophores
        self.hidden = hidden
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_chromophores * n_channels * n_times, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, X):
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input, got {X.ndim}D")
        return self.layers(X)


@pytest.mark.parametrize(
    ("shape", "n_chromophores"), [((8, 4, 16), 1), ((8, 2, 4, 16), 2)]
)
def test_classifier_clones_and_infers_module_dimensions(shape, n_chromophores):
    X = np.random.default_rng(0).normal(size=shape)
    y = np.array(["left", "right"] * 4)
    classifier = FNIRSClassifier(
        module=TinyNet,
        module__hidden=3,
        max_epochs=1,
        batch_size=4,
        device="cpu",
        random_state=0,
    )
    fitted = clone(classifier).fit(X, y)
    probabilities = fitted.predict_proba(X)
    assert fitted.module_.n_channels == 4
    assert fitted.module_.n_times == 16
    assert fitted.module_.n_outputs == 2
    assert fitted.module_.n_chromophores == n_chromophores
    assert fitted.n_features_in_ == np.prod(shape[1:])
    assert fitted.module_.hidden == 3
    assert next(fitted.module_.parameters()).dtype == torch.float32
    assert probabilities.shape == (8, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    assert set(fitted.predict(X)) <= {"left", "right"}


def test_classifier_reports_multidimensional_input_and_unfitted_prediction():
    classifier = FNIRSClassifier(module=TinyNet)
    tags = classifier.__sklearn_tags__()
    assert tags.input_tags.three_d_array
    assert not tags.input_tags.two_d_array
    with pytest.raises(NotFittedError):
        classifier.predict(np.ones((2, 4, 16)))


def test_classifier_steps_learning_rate_scheduler_after_each_epoch():
    X = np.random.default_rng(0).normal(size=(8, 4, 16))
    y = np.array([0, 1] * 4)
    classifier = FNIRSClassifier(
        module=TinyNet,
        scheduler=torch.optim.lr_scheduler.StepLR,
        scheduler__step_size=1,
        scheduler__gamma=0.5,
        lr=0.01,
        max_epochs=2,
        batch_size=4,
        device="cpu",
        random_state=0,
    )
    fitted = classifier.fit(X, y)
    assert fitted.optimizer_.param_groups[0]["lr"] == pytest.approx(0.0025)


def test_classifier_fits_and_predicts_mne_epochs():
    X = np.random.default_rng(0).normal(size=(8, 4, 16))
    info = mne.create_info(
        [f"S{channel}_D{channel} hbo" for channel in range(4)],
        5.0,
        ["hbo"] * 4,
    )
    metadata = pd.DataFrame({"subject": [1] * 8})
    epochs = mne.EpochsArray(
        X,
        info,
        events=np.column_stack([np.arange(8) * 16, np.zeros(8, dtype=int), [1, 2] * 4]),
        event_id={"left": 1, "right": 2},
        metadata=metadata,
        verbose=False,
    )
    classifier = FNIRSClassifier(
        module=TinyNet,
        unit_factor=2.0,
        max_epochs=1,
        batch_size=4,
        device="cpu",
        random_state=0,
    )
    fitted = classifier.fit(epochs, y=None)
    np.testing.assert_array_equal(fitted.classes_, ["left", "right"])
    assert fitted.metadata_.equals(metadata)
    assert fitted.info_["ch_names"] == epochs.ch_names
    assert fitted.predict(epochs).shape == (8,)
