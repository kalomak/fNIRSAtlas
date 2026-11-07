# Adapted from https://github.com/tufts-ml/fNIRS-mental-workload-classifiers (MIT, Copyright (c) 2021 Tufts Machine Learning).

import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter

logger = logging.getLogger(__name__)


class DCN(DLModelBase):
    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=2,
        num_classes=2,
        sfreq=5.0,
        kernel_size=2,
        pool_size=2,
        pool_stride=2,
        stride=1,
        p_dropout=0.5,
        D1=25,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.sfreq = sfreq
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.stride = stride
        self.p_dropout = p_dropout
        self.D1 = D1

        padding = (kernel_size - 1) // 2

        b1_size = (T + 2 * padding - kernel_size) // stride + 1
        b1_size = (b1_size - pool_size) // pool_stride + 1

        b2_size = (b1_size + 2 * padding - kernel_size) // stride + 1
        b2_size = (b2_size - pool_size) // pool_stride + 1

        b3_size = (b2_size + 2 * padding - kernel_size) // stride + 1
        b3_size = (b3_size - pool_size) // pool_stride + 1

        b4_size = (b3_size + 2 * padding - kernel_size) // stride + 1
        b4_size = (b4_size - pool_size) // pool_stride + 1

        # print(f"b1_size: {b1_size}")
        # print(f"b2_size: {b2_size}")
        # print(f"b3_size: {b3_size}")
        # print(f"b4_size: {b4_size}")

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=n_ch_types,
                out_channels=D1,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=True,
            ),
            nn.Conv2d(
                in_channels=D1,
                out_channels=D1,
                kernel_size=(n_chs, 1),
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=D1,
                out_channels=D1 * 2,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1 * 2,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),  # use by author
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=D1 * 2,
                out_channels=D1 * 4,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1 * 4,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),  # use by author
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Conv2d(
                in_channels=D1 * 4,
                out_channels=D1 * 8,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(
                num_features=D1 * 8,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ELU(),  # use by author
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_stride)),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels=D1 * 8,
                out_channels=num_classes,
                kernel_size=(1, b4_size),
                bias=True,
            ),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = x.flatten(1)
        return x


def create_dcn_pipeline(**kwargs):
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
                DCN(
                    n_chs,
                    T,
                    n_ch_types,
                    num_classes=num_classes,
                    sfreq=sfreq,
                ),
            ),
        ]
    )
