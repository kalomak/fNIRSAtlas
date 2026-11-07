"""
Test for the new DL pipeline functionality.
"""

import numpy as np
import torch
from sklearn.pipeline import Pipeline
from fnirsatlas.feature_extraction import ChannelSplitter, UnsqueezeLayer
from fnirsatlas.dl.EEGNet import EEGNet
from fnirsatlas.dl.fNIRSNet24 import fNIRSNet24, create_fnirsnet24_pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score


def test_channel_splitter():
    """Test ChannelSplitter transformer."""
    # Create dummy data
    X = np.random.randn(10, 4, 20)  # 10 samples, 4 channels, 20 timepoints
    ch_names = ["S1-D1 hbo", "S1-D1 hbr", "S2-D2 hbo", "S2-D2 hbr"]

    # Test ChannelSplitter
    splitter = ChannelSplitter(ch_names=ch_names)
    splitter.fit(X)
    X_split = splitter.transform(X)

    assert X_split.shape == (10, 2, 2, 20)  # 2 channel types, 2 channels per type
    assert np.allclose(X_split[:, 0, :, :], X[:, [0, 2], :])  # hbo channels
    assert np.allclose(X_split[:, 1, :, :], X[:, [1, 3], :])  # hbr channels
    print("ChannelSplitter test passed")


def test_eegnet_sklearn_compatibility():
    """Test EEGNet sklearn compatibility."""
    # Create dummy data with appropriate dimensions for EEGNet
    X = np.random.randn(10, 8, 64)  # 10 samples, 8 channels, 64 timepoints
    ch_names = [
        "S1-D1 hbo",
        "S1-D1 hbr",
        "S2-D2 hbo",
        "S2-D2 hbr",
        "S3-D3 hbo",
        "S3-D3 hbr",
        "S4-D4 hbo",
        "S4-D4 hbr",
    ]
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    # Create pipeline
    pipeline = Pipeline(
        [
            ("channel_splitter", ChannelSplitter(ch_names=ch_names)),
            (
                "eegnet",
                EEGNet(
                    n_ch_types=2,
                    n_chs=4,  # 4 channels per type
                    T=64,
                    sfreq=5.0,
                    num_classes=2,
                ),
            ),
        ]
    )

    # Test fit and predict
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)
    print("EEGNet sklearn compatibility test passed")


def test_dl_model_base_compatibility():
    """Test DLModelBase basic functionality."""
    from fnirsatlas.dl.DLModelBase import DLModelBase

    # Create a simple test model
    class TestModel(DLModelBase):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.linear = torch.nn.Linear(10, 2)

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            return self.linear(x)

    # Test basic functionality
    model = TestModel()
    assert model.batch_size == 32
    assert model.device == "auto"

    # Test factory function approach
    def create_model():
        return TestModel()

    model2 = create_model()
    assert model2.batch_size == 32

    print("DLModelBase factory function approach works")


def test_dl_model_base_custom_dataloader_hook():
    """Test that subclasses can override train/val dataloader creation."""
    from fnirsatlas.dl.DLModelBase import DLModelBase
    from fnirsatlas.dl.utils import to_dataloader

    class TestModel(DLModelBase):
        def __init__(self, **kwargs):
            super().__init__(n_epochs=1, batch_size=4, device="cpu", **kwargs)
            self.linear = torch.nn.Linear(10, 2)
            self.train_dataloader_calls = []
            self.val_dataloader_calls = []

        def create_train_dataloader(self, X, y, *, device, **kwargs):
            self.train_dataloader_calls.append({"device": device.type})
            return to_dataloader(
                X,
                y,
                batch_size=self.batch_size,
                shuffle=True,
                device=device.type,
                **kwargs,
            )

        def create_val_dataloader(self, X, y, *, device, **kwargs):
            self.val_dataloader_calls.append({"device": device.type})
            return to_dataloader(
                X,
                y,
                batch_size=self.batch_size,
                shuffle=False,
                device=device.type,
                **kwargs,
            )

        def forward(self, x):
            return self.linear(x)

    X = np.random.randn(8, 10).astype(np.float32)
    y = np.array([0, 1] * 4)
    model = TestModel()

    model.fit(X, y, X_val=X, y_val=y)

    assert model.train_dataloader_calls == [{"device": "cpu"}]
    assert model.val_dataloader_calls == [{"device": "cpu"}]


def test_unsqueeze_layer():
    X = np.random.randn(5, 8, 64)
    layer = UnsqueezeLayer(dim=1)
    X_out = layer.fit_transform(X)

    assert X_out.shape == (5, 1, 8, 64)
    assert np.allclose(X_out[:, 0], X)


def test_fnirsnet24_sklearn_compatibility():
    X = np.random.randn(10, 8, 64)
    y = np.array([0, 1] * 5)
    pipeline = Pipeline(
        [
            ("unsqueeze", UnsqueezeLayer(dim=1)),
            (
                "fnirsnet24",
                fNIRSNet24(
                    n_chs=8,
                    T=64,
                    num_classes=2,
                    n_epochs=1,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_fnirsnet24_factory_pipeline():
    X = np.random.randn(10, 8, 64)
    y = np.array([0, 1] * 5)
    pipeline = create_fnirsnet24_pipeline(
        X=X,
        y=y,
        random_state=42,
        classification_setting="cross",
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)


def test_custom_cross_validate():
    """Test custom cross_validate function."""
    from fnirsatlas.classification import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    import numpy as np

    # Create dummy data
    X = np.random.randn(20, 10)
    y = np.array([0, 1] * 10)

    # Create factory function
    def create_model():
        return LogisticRegression(random_state=42)

    # Test custom cross_validate
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = cross_validate(
        X,
        y,
        list(cv.split(X, y)),
        create_model,
        scoring={"accuracy": accuracy_score, "kappa": cohen_kappa_score},
        n_jobs=1,
    )

    assert "test_accuracy" in results
    assert "test_kappa" in results
    assert len(results["test_accuracy"]) == 3
    assert len(results["test_kappa"]) == 3

    print("Custom cross_validate test passed")


CH_NAMES = [
    "S1-D1 hbo",
    "S1-D1 hbr",
    "S2-D2 hbo",
    "S2-D2 hbr",
    "S3-D3 hbo",
    "S3-D3 hbr",
    "S4-D4 hbo",
    "S4-D4 hbr",
]
N_CHS = 4
N_CH_TYPES = 2
T = 64


def test_cross_validate_default_metrics():
    """Test custom cross_validate default metrics."""
    from fnirsatlas.classification import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    X = np.random.randn(20, 10)
    y = np.array([0, 1] * 10)

    def create_model():
        return LogisticRegression(random_state=42)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = cross_validate(X, y, list(cv.split(X, y)), create_model, n_jobs=1)

    assert "test_accuracy" in results
    assert "test_kappa" in results
    assert "test_f1_macro" in results
    assert "test_precision" in results
    assert "test_recall" in results
    assert len(results["test_f1_macro"]) == 3
    assert len(results["test_precision"]) == 3
    assert len(results["test_recall"]) == 3


def test_cross_validate_default_metrics_with_search():
    """Test default metrics also work through GridSearchCV scorers."""
    from fnirsatlas.classification import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression

    X = np.random.randn(20, 10)
    y = np.array([0, 1] * 10)

    def create_model():
        return LogisticRegression(random_state=42)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    results = cross_validate(
        X,
        y,
        list(cv.split(X, y)),
        create_model,
        n_jobs=1,
        param_grid={"C": [0.1, 1.0]},
        classification_setting="within",
        search_strategy="single",
    )

    assert "test_accuracy" in results
    assert "test_kappa" in results
    assert "test_f1_macro" in results
    assert "test_precision" in results
    assert "test_recall" in results
    assert len(results["test_f1_macro"]) == 3
    assert len(results["test_precision"]) == 3
    assert len(results["test_recall"]) == 3
    assert "search_results" in results

    print("Custom cross_validate default metrics with search test passed")


if __name__ == "__main__":
    test_channel_splitter()
    test_eegnet_sklearn_compatibility()
    test_dl_model_base_compatibility()
    test_dl_model_base_custom_dataloader_hook()
    test_unsqueeze_layer()
    test_fnirsnet24_sklearn_compatibility()
    test_fnirsnet24_factory_pipeline()
    test_custom_cross_validate()
    test_tscnn_sklearn_compatibility()
    test_stcnn_sklearn_compatibility()
    test_tscnn_stcnn_param_grid()
    test_cross_validate_default_metrics()
    test_cross_validate_default_metrics_with_search()
    print("All tests passed!")
