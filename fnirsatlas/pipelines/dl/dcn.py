# Adapted from https://github.com/tufts-ml/fNIRS-mental-workload-classifiers (MIT, Copyright (c) 2021 Tufts Machine Learning).

"""Deep convolutional network for chromophore-split fNIRS epochs."""

from torch import nn


def _conv_block(in_channels, out_channels, n_chs, kernel_size, stride, pool, dropout):
    layers = []
    if n_chs is None:
        layers.append(nn.Dropout(dropout))
    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, (kernel_size - 1) // 2),
            bias=n_chs is not None,
        )
    )
    if n_chs is not None:
        layers.append(nn.Conv2d(out_channels, out_channels, (n_chs, 1), bias=False))
    layers.extend(
        [
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, pool[0]), stride=(1, pool[1])),
        ]
    )
    return nn.Sequential(*layers)


class DCN(nn.Module):
    """Four convolution/pooling stages followed by a full-width classifier."""

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
    ):
        super().__init__()
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.sfreq = sfreq
        width = T
        for _ in range(4):
            width = (width + 2 * ((kernel_size - 1) // 2) - kernel_size) // stride + 1
            width = (width - pool_size) // pool_stride + 1
        if width < 1:
            raise ValueError("n_times is too short for DCN pooling")
        self.blocks = nn.Sequential(
            *[
                _conv_block(
                    n_ch_types if index == 0 else D1 * 2 ** (index - 1),
                    D1 * 2**index,
                    n_chs if index == 0 else None,
                    kernel_size,
                    stride,
                    (pool_size, pool_stride),
                    p_dropout,
                )
                for index in range(4)
            ]
        )
        self.classifier = nn.Conv2d(D1 * 8, num_classes, kernel_size=(1, width))

    def forward(self, X):
        """Return class logits for chromophore-split epochs."""
        return self.classifier(self.blocks(X)).flatten(1)
