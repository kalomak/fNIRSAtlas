"""Stacked LSTM based on Asgher et al. (2020)."""

from torch import nn


class LSTM(nn.Module):
    """Classify chromophore-split epochs using the final recurrent timestep.

    Based on doi:10.3389/fnins.2020.00584.
    Input shape is (trials, chromophores, channels, timepoints).
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
    ):
        super().__init__()
        self.n_chs = n_chs
        self.T = T
        self.n_ch_types = n_ch_types
        self.num_classes = num_classes
        self.sfreq = sfreq
        self.lstm = nn.LSTM(
            input_size=n_ch_types * n_chs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=p_dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        """Return class logits for chromophore-split epochs."""
        X = X.flatten(1, 2).transpose(1, 2)
        outputs, _ = self.lstm(X)
        return self.classifier(outputs[:, -1, :])
