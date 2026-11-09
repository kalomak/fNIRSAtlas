"""Integration test for a Braindecode classifier on MOABB arrays."""

import numpy as np
import pytest
from sklearn.base import clone

pytestmark = pytest.mark.data

pytest.importorskip("braindecode")
pytest.importorskip("torch")

from braindecode import EEGClassifier
from braindecode.models import EEGNet

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import ChannelMerger


@pytest.mark.manual
def test_braindecode_eegnet_infers_moabb_data_shape_on_fit():
    dataset = Bnci2014FnirsMA(subjects=[1])
    X, y, _ = FNIRSClassification().get_data(dataset=dataset, subjects=[1])
    X = ChannelMerger().transform(X)
    _, y_encoded = np.unique(y[:8], return_inverse=True)
    classifier = EEGClassifier(
        module=EEGNet,
        module__kernel_length=16,
        module__depthwise_kernel_length=8,
        module__pool1_kernel_size=2,
        module__pool2_kernel_size=2,
        module__drop_prob=0.1,
        max_epochs=1,
        batch_size=4,
        train_split=None,
        verbose=0,
        device="cpu",
    )

    fitted = clone(classifier).fit(X[:8], y_encoded)

    assert fitted.module_.n_chans == X.shape[1]
    assert fitted.module_.n_times == X.shape[2]
    assert fitted.module_.n_outputs == len(np.unique(y_encoded))
    assert fitted.predict(X[:8]).shape == (8,)
