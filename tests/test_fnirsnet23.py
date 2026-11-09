import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from fnirsatlas.pipelines.dl import FNIRSClassifier, fNIRSNet23


@pytest.mark.parametrize("batch_size", [1, 3])
def test_fnirsnet23_returns_class_logits(batch_size):
    model = fNIRSNet23(n_chs=4, T=16, n_ch_types=2, num_classes=3)

    logits = model(torch.ones(batch_size, 2, 4, 16))

    assert logits.shape == (batch_size, 3)


def test_fnirsnet23_classifier_infers_dimensions_and_handles_singleton_batch():
    X = np.random.default_rng(0).normal(size=(5, 2, 4, 16))
    y = np.array(["left", "right", "left", "right", "left"])
    classifier = FNIRSClassifier(
        module=fNIRSNet23,
        max_epochs=1,
        batch_size=2,
        device="cpu",
        random_state=0,
    )

    fitted = clone(classifier).fit(X, y)

    assert fitted.module_.n_chs == 4
    assert fitted.module_.T == 16
    assert fitted.module_.n_ch_types == 2
    assert fitted.module_.num_classes == 2
    assert fitted.predict(X).shape == (5,)
