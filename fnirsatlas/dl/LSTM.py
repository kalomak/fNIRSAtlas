import torch.nn as nn
import logging
import numpy as np
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter

logger = logging.getLogger(__name__)


class LSTM(DLModelBase):
    """Stacked LSTM baseline from Asgher et al. 2020 (doi:10.3389/fnins.2020.00584).

    Treats the fNIRS epoch as a sequence over time, with per-timestep features
    formed by concatenating all channels across channel types (HbO + HbR).
    This implementation is not guaranteed to be correct, has not been checked
    by the original authors, and is reimplemented based on the paper.
    """

    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=2,
        num_classes=2,
        sfreq=5.0,
        hidden_size=64,
        num_layers=4,
        p_dropout=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.sfreq = sfreq
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_dropout = p_dropout

        # dropout between stacked layers is only applied when num_layers > 1
        lstm_dropout = p_dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_ch_types * n_chs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, n_ch_types, n_chs, T) -> (B, T, n_ch_types*n_chs)
        x = x.flatten(1, 2).transpose(1, 2)
        out, _ = self.lstm(x)
        # take last timestep
        out = out[:, -1, :]
        return self.classifier(out)


def create_lstm_pipeline(**kwargs):
    num_classes = np.unique(kwargs["y"]).shape[0]
    ch_names = kwargs["info"]["ch_names"]
    n_ch_types = len(np.unique([c.split()[1] for c in ch_names]))
    n_chs = len(ch_names) // n_ch_types
    sfreq = kwargs["info"]["sfreq"]
    T = kwargs["X"].shape[-1]
    classification_setting = kwargs.get("classification_setting", "")
    if classification_setting == "within":
        kwargs["batch_size"] = 8
    return Pipeline(
        [
            ("ch_splitter", ChannelSplitter(ch_names)),
            ("scaler", StandardScalerND(dim=(0, 2, 3))),
            (
                "clf",
                LSTM(
                    n_chs,
                    T,
                    n_ch_types,
                    num_classes=num_classes,
                    sfreq=sfreq,
                ),
            ),
        ]
    )
