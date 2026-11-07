"""
CNN2-a model from Trakoolwilaiwan et al. (2017): "Convolutional neural network
for high-accuracy fNIRS in a BCI" doi: 10.1117/1.NPh.5.1.011008

Based on Table 4 architecture: 3 Conv2d blocks (temporal) + 2 FC layers.

The paper uses input of shape (n, c, t), we adapt it to use input shape
(n, ct, c, t) like the other models.
"""

import torch.nn as nn
import numpy as np
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter


class CNN2a(DLModelBase):
    """
    Conv2d temporal CNN matching Table 4 of Trakoolwilaiwan et al. (2017).

    Input: (B, n_ch_types, n_chs, T) from ChannelSplitter.
    First conv collapses the channel dimension with kernel (n_chs, 3).
    This implementation is not guaranteed to be correct, has not been checked
    by the original authors, and is reimplemented based on the paper.
    """

    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=2,
        num_classes=2,
        n_filters=32,
        kernel_size=3,
        pool_size=2,
        p_dropout=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.p_dropout = p_dropout

        pad = (kernel_size - 1) // 2

        # Trace the exact temporal size through each conv+pool block
        def _t_out(t):
            return (t + 2 * pad - kernel_size + 1) // pool_size

        fc_in = n_filters * _t_out(_t_out(_t_out(T)))

        # First block: kernel spans all channels → collapses spatial dim to 1
        self.block1 = nn.Sequential(
            nn.Conv2d(
                n_ch_types,
                n_filters,
                kernel_size=(n_chs, kernel_size),
                padding=(0, pad),
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pool_size)),
            nn.Dropout(p_dropout),
        )

        # Subsequent blocks: spatial dim is 1, kernel (1, kernel_size)
        def later_block():
            return nn.Sequential(
                nn.Conv2d(
                    n_filters, n_filters, kernel_size=(1, kernel_size), padding=(0, pad)
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pool_size)),
                nn.Dropout(p_dropout),
            )

        self.block2 = later_block()
        self.block3 = later_block()

        self.classifier = nn.Sequential(
            nn.Linear(fc_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def create_cnn2a_pipeline(**kwargs):
    num_classes = np.unique(kwargs["y"]).shape[0]
    ch_names = kwargs["info"]["ch_names"]
    n_ch_types = len(np.unique([c.split()[1] for c in ch_names]))
    n_chs = len(ch_names) // n_ch_types
    T = kwargs["X"].shape[-1]
    if kwargs.get("classification_setting") == "within":
        kwargs["batch_size"] = 8
    return Pipeline(
        [
            ("ch_splitter", ChannelSplitter(ch_names)),
            ("scaler", StandardScalerND(dim=(0, 2, 3))),
            ("clf", CNN2a(n_chs, T, n_ch_types, num_classes=num_classes)),
        ]
    )
