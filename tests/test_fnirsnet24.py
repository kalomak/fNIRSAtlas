import numpy as np
import pytest
from sklearn.base import clone

torch = pytest.importorskip("torch")

from fnirsatlas.pipelines.dl import FNIRSClassifier, fNIRSNet24


@pytest.mark.parametrize("batch_size", [1, 3])
def test_fnirsnet24_returns_class_logits(batch_size):
    model = fNIRSNet24(n_chs=4, T=64, n_ch_types=2, num_classes=3)

    logits = model(torch.ones(batch_size, 2, 4, 64))

    assert logits.shape == (batch_size, 3)


def test_fnirsnet24_classifier_infers_dimensions():
    X = np.random.default_rng(0).normal(size=(5, 2, 4, 64))
    y = np.array(["left", "right", "left", "right", "left"])
    classifier = FNIRSClassifier(
        module=fNIRSNet24,
        max_epochs=1,
        batch_size=2,
        device="cpu",
        random_state=0,
    )

    fitted = clone(classifier).fit(X, y)

    assert fitted.module_.n_chs == 4
    assert fitted.module_.T == 64
    assert fitted.module_.n_ch_types == 2
    assert fitted.module_.num_classes == 2
    assert fitted.predict(X).shape == (5,)
