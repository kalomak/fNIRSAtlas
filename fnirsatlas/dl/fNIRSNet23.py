# Adapted from https://github.com/wzhlearning/fNIRSNet (Apache-2.0).

"""
Model adapted from:
    
https://github.com/wzhlearning/fNIRSNet/blob/2a2010d581f66f1b950ccfed390582d24dcedac2/fNIRSNet.py
"""
import numpy as np
import torch.nn as nn
from sklearn.pipeline import Pipeline

from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import ChannelSplitter, StandardScalerND


class fNIRSNet23(DLModelBase):
    def __init__(self, n_chs, T, n_ch_types=2, num_classes=2, F1=4, F2=8, **kwargs):
        super().__init__(**kwargs)
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.F1 = F1
        self.F2 = F2

        self.dhr_conv = nn.Conv2d(n_ch_types, F1, kernel_size=(1, T))
        self.bn1 = nn.BatchNorm2d(F1)
        self.dw_conv = nn.Conv2d(F1, F1, kernel_size=(n_chs, 1), groups=F1)
        self.pw_conv = nn.Conv2d(F1, F2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(F2)
        self.activation = nn.Sigmoid()
        self.classifier = nn.Linear(F2, num_classes)

    def forward(self, x):
        x = self.activation(self.bn1(self.dhr_conv(x)))
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        # BatchNorm2d cannot compute training stats from a singleton [1, C, 1, 1].
        if not (self.training and x.shape == (1, self.F2, 1, 1)):
            x = self.bn2(x)
        x = self.activation(x)
        x = x.flatten(1)
        return self.classifier(x)


def create_fnirsnet23_pipeline(**kwargs):
    num_classes = np.unique(kwargs["y"]).shape[0]
    ch_names = kwargs["info"]["ch_names"]
    n_ch_types = len(np.unique([c.split()[1] for c in ch_names]))
    n_chs = len(ch_names) // n_ch_types
    T = kwargs["X"].shape[-1]
    if kwargs.get("classification_setting", "") == "within":
        kwargs["batch_size"] = 8
    clf_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"X", "y", "info", "classification_setting"}
    }
    return Pipeline(
        [
            ("ch_splitter", ChannelSplitter(ch_names)),
            ("scaler", StandardScalerND(dim=(0, 2, 3))),
            (
                "clf",
                fNIRSNet23(
                    n_chs,
                    T,
                    n_ch_types=n_ch_types,
                    num_classes=num_classes,
                    **clf_kwargs,
                ),
            ),
        ]
    )
