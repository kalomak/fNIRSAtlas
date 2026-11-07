# Adapted from https://github.com/tufts-ml/fNIRS-mental-workload-classifiers (MIT, Copyright (c) 2021 Tufts Machine Learning).

"""
EEGNet model implementation adapted from:

https://github.com/tufts-ml/fNIRS-mental-workload-classifiers/blob/b5199d6184e659152d1fe650db48eba53a221186/helpers/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.dl.layers import PrintShapeLayer
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter

logger = logging.getLogger(__name__)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )

        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(DLModelBase):
    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=2,
        sfreq=5.0,
        kernel_size_1=3,
        num_classes=2,
        F1=4,
        D=2,
        F2=None,
        p_dropout=0.5,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        # Store all parameters as attributes for sklearn compatibility
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.sfreq = sfreq
        self.kernel_size_1 = kernel_size_1
        self.num_classes = num_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.p_dropout = p_dropout

        F2 = F2 if F2 is not None else F1 * D
        if kernel_size_1 != sfreq // 2 and kernel_size_1 != sfreq // 2 + 1:
            logger.warning(f"Original implementation uses kernel_size_1=sfreq / 2")

        W1 = T - kernel_size_1 + 3
        W2 = (W1 - 4) // 4 + 1
        W3 = (W2 - 8) // 8 + 1
        clf_in = F2 * W3

        # Temporal convolution
        self.firstConv = nn.Sequential(
            PrintShapeLayer("c0"),
            nn.Conv2d(
                in_channels=n_ch_types,
                out_channels=F1,
                kernel_size=(1, kernel_size_1),
                stride=1,
                padding=(0, 1),
                bias=False,
            ),
            PrintShapeLayer("c1"),
            #'same' padding: used by the author;
            # kernel_size=(1,3):  author recommend kernel length be half of the sampling rate
            nn.BatchNorm2d(
                num_features=F1,
                eps=1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(
                F1,
                F1 * D,
                kernel_size=(n_chs, 1),
                stride=(1, 1),
                groups=F1,
                bias=False,
            ),
            PrintShapeLayer("c2"),
            #'valid' padding: used by the author;
            # kernel_size = (feature_size, 1): used by the author
            nn.BatchNorm2d(
                F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ELU(),  # used by author
            nn.AvgPool2d(kernel_size=(1, 4)),  # kernel_size=(1,4) used by author
            PrintShapeLayer("p2"),
            nn.Dropout(p=p_dropout),
        )

        # depthwise convolution follow by pointwise convolution (pointwise convolution is just Conv2d with 1x1 kernel)
        self.separableConv = nn.Sequential(
            nn.Conv2d(
                F1 * D,
                F2,
                kernel_size=(1, 3),
                stride=1,
                padding=(0, 1),
                groups=F1 * D,
                bias=False,
            ),
            PrintShapeLayer("c3"),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            PrintShapeLayer("c4"),
            nn.BatchNorm2d(
                F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
            ),
            nn.ELU(),  # use by author
            nn.AvgPool2d(kernel_size=(1, 8)),  # kernel_size=(1,8): used by author
            PrintShapeLayer("p3"),
            nn.Dropout(p=p_dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=clf_in, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.reshape(batch_size, -1)
        PrintShapeLayer("flatten")(x)
        x = self.classifier(x)
        PrintShapeLayer("output")(x)
        return x


def create_eegnet_pipeline(**kwargs):
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
                EEGNet(
                    n_chs,
                    T,
                    n_ch_types,
                    num_classes=num_classes,
                    sfreq=sfreq,
                ),
            ),
        ]
    )
