"""fNIRSNet24 fusion architecture for chromophore-split fNIRS arrays.

Reimplementation of the parallel-input fusion variant from
https://doi.org/10.1016/j.engappai.2024.109256. Input: ``(b, chromo, ch, t)``.
"""

import torch
from torch import nn

from .eegnet import Conv2dWithConstraint


class Square(nn.Module):
    """Element-wise square activation."""

    def forward(self, X):
        """Return the squared input."""
        return X.square()


class SafeLog(nn.Module):
    """Logarithm with a lower clamp on the input."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        """Return the clamped logarithm."""
        return torch.log(torch.clamp(X, min=self.eps))


class LinearWithConstraint(nn.Linear):
    """Linear layer with a maximum weight norm."""

    def __init__(self, *args, max_norm=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, input):
        """Apply the constrained linear projection."""
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(input)


class SeparableConv2d(nn.Module):
    """Depthwise convolution followed by a pointwise projection."""

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

    def forward(self, X):
        """Return the separable convolution of ``X``."""
        return self.pointwise(self.depthwise(X))


class fNIRSNet24(nn.Module):
    """Two parallel temporal/spatial views fused before log-power pooling."""

    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=2,
        num_classes=2,
        temporal_filters=20,
        spatial_filters=60,
        fusion_filters=10,
        temporal_kernel=5,
        fusion_kernel=10,
        pool_kernel=25,
        pool_stride=8,
        p_dropout=0.3,
    ):
        super().__init__()
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
        width = T - temporal_kernel + 1
        pooled_width = (width - pool_kernel) // pool_stride + 1
        if pooled_width < 1:
            raise ValueError("T is too short for fNIRSNet24 pooling")
        self.view1 = self._build_view1(width)
        self.view2 = self._build_view2(width)
        self.fusion = self._build_fusion(width)
        self.pool = self._build_pool()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                fusion_filters * pooled_width, num_classes, max_norm=0.5
            ),
        )

    def _build_view1(self, width):
        return nn.Sequential(
            Conv2dWithConstraint(
                self.n_ch_types,
                self.temporal_filters,
                kernel_size=(1, self.temporal_kernel),
                bias=False,
                max_norm=2,
            ),
            nn.LayerNorm([self.temporal_filters, self.n_chs, width], eps=1e-5),
            Square(),
            Conv2dWithConstraint(
                self.temporal_filters,
                self.spatial_filters,
                kernel_size=(self.n_chs, 1),
                bias=False,
                max_norm=1,
            ),
            nn.LayerNorm([self.spatial_filters, 1, width], eps=1e-2),
            Square(),
        )

    def _build_view2(self, width):
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
            nn.LayerNorm([self.temporal_filters, 1, width], eps=1e-5),
            Square(),
        )

    def _build_fusion(self, width):
        in_channels = self.spatial_filters + self.temporal_filters
        return nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-5),
            SeparableConv2d(
                in_channels, self.fusion_filters, kernel_size=(1, self.fusion_kernel)
            ),
            nn.LayerNorm([self.fusion_filters, 1, width], eps=1.93e-3),
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

    def forward(self, X):
        """Return class logits."""
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(X.shape)}")
        fused = self.fusion(torch.cat([self.view1(X), self.view2(X)], dim=1))
        return self.classifier(self.pool(fused))
