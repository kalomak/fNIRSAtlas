"""CNN2-a from Trakoolwilaiwan et al. (2017), doi:10.1117/1.NPh.5.1.011008."""

from torch import nn


def _conv_block(in_channels, out_channels, height, kernel_size, pool_size, dropout):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(height, kernel_size),
            padding=(0, (kernel_size - 1) // 2),
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(1, pool_size)),
        nn.Dropout(dropout),
    )


class CNN2a(nn.Module):
    """Three temporal convolution blocks and a dense classifier from Table 4.

    Input shape is (trials, chromophores, channels, timepoints).
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
    ):
        super().__init__()
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        width = T
        for _ in range(3):
            width = (
                width + 2 * ((kernel_size - 1) // 2) - kernel_size + 1
            ) // pool_size
        if width < 1:
            raise ValueError("n_times is too short for CNN2a pooling")
        self.blocks = nn.Sequential(
            *[
                _conv_block(
                    n_ch_types if index == 0 else n_filters,
                    n_filters,
                    n_chs if index == 0 else 1,
                    kernel_size,
                    pool_size,
                    p_dropout,
                )
                for index in range(3)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(n_filters * width, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, X):
        """Return class logits for chromophore-split epochs."""
        return self.classifier(self.blocks(X).flatten(1))
