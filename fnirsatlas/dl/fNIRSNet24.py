"""
fNIRSNET Model from:
https://doi.org/10.1016/j.engappai.2024.109256

This is the variant that achieved the highest accuracies (parallel input,
fusion type).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline

from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.dl.EEGNet import Conv2dWithConstraint
from fnirsatlas.feature_extraction import StandardScalerND, UnsqueezeLayer


class Square(nn.Module):
    def forward(self, x):
        return x.square()


class SafeLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super().forward(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_norm=2):
        super().__init__()
        self.depthwise = Conv2dWithConstraint(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding="same",
            groups=in_channels,
            bias=False,
            max_norm=max_norm,
        )
        self.pointwise = Conv2dWithConstraint(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            max_norm=max_norm,
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def _pooled_width(width, kernel_size=25, stride=8):
    return (width - kernel_size) // stride + 1


class fNIRSNet24(DLModelBase):
    """Reimplementation of the fNIRSNet24 fusion variant from the paper.

    This implementation is not guaranteed to be correct, has not been checked
    by the original authors, and is reimplemented based on the paper.
    """

    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=1,
        num_classes=2,
        temporal_filters=20,
        spatial_filters=60,
        fusion_filters=10,
        temporal_kernel=5,
        fusion_kernel=10,
        pool_kernel=25,
        pool_stride=8,
        p_dropout=0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.temporal_filters = temporal_filters
        self.spatial_filters = spatial_filters
        self.fusion_filters = fusion_filters
        self.temporal_kernel = temporal_kernel
        self.fusion_kernel = fusion_kernel
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride
        self.p_dropout = p_dropout
        time_after_temporal = T - temporal_kernel + 1
        pooled_width = _pooled_width(time_after_temporal, pool_kernel, pool_stride)
        self.view1 = self._build_view1(time_after_temporal)
        self.view2 = self._build_view2(time_after_temporal)
        self.fusion = self._build_fusion(time_after_temporal)
        self.pool = self._build_pool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                fusion_filters * pooled_width,
                num_classes,
                max_norm=0.5,
            ),
        )

    def _build_view1(self, time_after_temporal):
        return nn.Sequential(
            Conv2dWithConstraint(
                self.n_ch_types,
                self.temporal_filters,
                kernel_size=(1, self.temporal_kernel),
                bias=False,
                max_norm=2,
            ),
            nn.LayerNorm(
                [self.temporal_filters, self.n_chs, time_after_temporal], eps=1e-5
            ),
            Square(),
            Conv2dWithConstraint(
                self.temporal_filters,
                self.spatial_filters,
                kernel_size=(self.n_chs, 1),
                bias=False,
                max_norm=1,
            ),
            nn.LayerNorm([self.spatial_filters, 1, time_after_temporal], eps=1e-2),
            Square(),
        )

    def _build_view2(self, time_after_temporal):
        return nn.Sequential(
            Conv2dWithConstraint(
                self.n_ch_types,
                self.spatial_filters,
                kernel_size=(self.n_chs, 1),
                bias=False,
                max_norm=1,
            ),
            nn.LayerNorm([self.spatial_filters, 1, self.T], eps=1e-2),
            Square(),
            Conv2dWithConstraint(
                self.spatial_filters,
                self.temporal_filters,
                kernel_size=(1, self.temporal_kernel),
                bias=False,
                max_norm=2,
            ),
            nn.LayerNorm([self.temporal_filters, 1, time_after_temporal], eps=1e-5),
            Square(),
        )

    def _build_fusion(self, time_after_temporal):
        in_channels = self.spatial_filters + self.temporal_filters
        return nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5),
            SeparableConv2d(
                in_channels,
                self.fusion_filters,
                kernel_size=(1, self.fusion_kernel),
            ),
            nn.LayerNorm([self.fusion_filters, 1, time_after_temporal], eps=1.93e-3),
            Square(),
        )

    def _build_pool(self):
        return nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, self.pool_kernel), stride=(1, self.pool_stride)
            ),
            SafeLog(),
            nn.Dropout(self.p_dropout),
        )

    def forward(self, x):
        view1 = self.view1(x)
        view2 = self.view2(x)
        fused = torch.cat([view1, view2], dim=1)
        fused = self.fusion(fused)
        pooled = self.pool(fused)
        flattened = self.classifier[0](pooled)
        logits = self.classifier[1](flattened)
        return logits


def create_fnirsnet24_pipeline(**kwargs):
    num_classes = np.unique(kwargs["y"]).shape[0]
    n_chs = kwargs["X"].shape[1]
    T = kwargs["X"].shape[-1]
    classification_setting = kwargs.get("classification_setting", "")
    if classification_setting == "within":
        kwargs["batch_size"] = 8
    clf_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"X", "y", "info", "classification_setting"}
    }
    return Pipeline(
        [
            # The paper explicitly tests different input shapes and shows that
            # (n, 1, chs, t) is a more effective input than (n, 2, ch_pairs, t)
            # se we use it here, ChannelSplitter should also work
            ("unsqueeze", UnsqueezeLayer(dim=1)),
            ("scaler", StandardScalerND(dim=(0, 2, 3))),
            (
                "clf",
                fNIRSNet24(
                    n_chs=n_chs,
                    T=T,
                    num_classes=num_classes,
                    **clf_kwargs,
                ),
            ),
        ]
    )
