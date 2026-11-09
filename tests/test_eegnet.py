import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

torch = pytest.importorskip("torch")

from fnirsatlas.pipelines import StandardScalerND
from fnirsatlas.pipelines.dl import EEGNet, FNIRSClassifier


def test_eegnet_accepts_single_chromophore_shape():
    model = EEGNet(n_channels=8, n_times=64, n_outputs=2)
    assert model(torch.ones(3, 1, 8, 64)).shape == (3, 2)


def test_eegnet_accepts_split_chromophore_shape():
    model = EEGNet(
        n_channels=4,
        n_times=64,
        n_outputs=3,
        n_chromophores=2,
        F2=6,
    )
    assert model(torch.ones(3, 2, 4, 64)).shape == (3, 3)


@pytest.mark.data
def test_eegnet_classifier_fits_moabb_arrays():
    from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification

    dataset = Bnci2014FnirsMA(subjects=[1])
    X, y, _ = FNIRSClassification().get_data(dataset=dataset, subjects=[1])
    classifier = FNIRSClassifier(
        module=EEGNet,
        max_epochs=1,
        batch_size=4,
        device="cpu",
        random_state=0,
    )
    fitted = clone(classifier).fit(X[:8], y[:8])
    assert fitted.module_.n_chromophores == X.shape[1]
    assert fitted.module_.n_channels == X.shape[2]
    assert fitted.module_.n_times == X.shape[3]
    assert fitted.module_.n_outputs == len(np.unique(y[:8]))
    assert fitted.predict(X[:8]).shape == (8,)


@pytest.mark.data
@pytest.mark.manual
def test_eegnet_runs_in_native_moabb_evaluation(tmp_path):
    from moabb.evaluations import WithinSessionEvaluation

    from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification

    classifier = make_pipeline(
        StandardScalerND(dim=(0, 2, 3)),
        FNIRSClassifier(
            module=EEGNet,
            max_epochs=1,
            batch_size=32,
            device="cpu",
            random_state=0,
        ),
    )
    evaluation = WithinSessionEvaluation(
        paradigm=FNIRSClassification(),
        datasets=[Bnci2014FnirsMA(subjects=[1])],
        hdf5_path=str(tmp_path / "eegnet_smoke"),
        overwrite=True,
        n_jobs=1,
        n_splits=2,
    )
    results = evaluation.process({"EEGNet": classifier})
    assert len(results) == 3
    assert results["score"].between(0, 1).all()
