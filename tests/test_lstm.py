"""Tests for the LSTM baseline model."""

import numpy as np
from sklearn.pipeline import Pipeline
from fnirsatlas.feature_extraction import ChannelSplitter
from fnirsatlas.dl.LSTM import LSTM, create_lstm_pipeline


def test_lstm_sklearn_compatibility():
    X = np.random.randn(10, 8, 64)
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
    pipeline = Pipeline(
        [
            ("channel_splitter", ChannelSplitter(ch_names=ch_names)),
            (
                "lstm",
                LSTM(
                    n_chs=4,
                    T=64,
                    n_ch_types=2,
                    sfreq=5.0,
                    num_classes=2,
                    num_layers=2,
                    hidden_size=16,
                    n_epochs=2,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_lstm_pipeline_factory():
    X = np.random.randn(10, 8, 64)
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
    info = {"ch_names": ch_names, "sfreq": 5.0}
    pipe = create_lstm_pipeline(X=X, y=y, info=info, n_epochs=2)
    pipe.fit(X, y)
    assert len(pipe.predict(X)) == len(y)
