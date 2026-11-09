# Adapted from https://github.com/tufts-ml/fNIRS-mental-workload-classifiers (MIT, Copyright (c) 2021 Tufts Machine Learning).

"""EEGNet adapted to chromophore-split fNIRS arrays."""

import torch
from torch import nn


class Conv2dWithConstraint(nn.Conv2d):
    """Two-dimensional convolution with a maximum weight norm."""

    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, X):
        """Apply the constrained convolution."""
        with torch.no_grad():
            self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(X)


class EEGNet(nn.Module):
    """EEGNet module accepting four-dimensional fNIRS arrays."""

    def __init__(
        self,
        n_channels,
        n_times,
        n_outputs,
        n_chromophores=1,
        kernel_size=3,
        F1=4,
        D=2,
        F2=None,
        dropout=0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_outputs = n_outputs
        self.n_chromophores = n_chromophores
        self.kernel_size = kernel_size
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout = dropout
        self._build_layers()

    def _build_layers(self):
        output_filters = self.F1 * self.D if self.F2 is None else self.F2
        pooled_times = ((self.n_times - self.kernel_size + 3) // 4) // 8
        if pooled_times < 1:
            raise ValueError("n_times is too short for EEGNet pooling")
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                self.n_chromophores,
                self.F1,
                (1, self.kernel_size),
                padding=(0, 1),
                bias=False,
            ),
            nn.BatchNorm2d(self.F1),
        )
        self.depthwise_conv = nn.Sequential(
            Conv2dWithConstraint(
                self.F1,
                self.F1 * self.D,
                (self.n_channels, 1),
                groups=self.F1,
                bias=False,
            ),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                self.F1 * self.D,
                self.F1 * self.D,
                (1, 3),
                padding=(0, 1),
                groups=self.F1 * self.D,
                bias=False,
            ),
            nn.Conv2d(self.F1 * self.D, output_filters, 1, bias=False),
            nn.BatchNorm2d(output_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout),
        )
        self.classifier = nn.Linear(output_filters * pooled_times, self.n_outputs)

    def forward(self, X):
        """Return class logits."""
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(X.shape)}")
        X = self.first_conv(X)
        X = self.depthwise_conv(X)
        X = self.separable_conv(X)
        return self.classifier(X.flatten(start_dim=1))
