"""fNIRSNet23 architecture for chromophore-split fNIRS arrays."""

from torch import nn


class fNIRSNet23(nn.Module):
    """fNIRSNet23 module accepting four-dimensional fNIRS arrays."""

    def __init__(
        self,
        n_chs,
        T,
        n_ch_types=2,
        num_classes=2,
        F1=4,
        F2=8,
    ):
        super().__init__()
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

    def forward(self, X):
        """Return class logits."""
        if X.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(X.shape)}")
        X = self.activation(self.bn1(self.dhr_conv(X)))
        X = self.pw_conv(self.dw_conv(X))
        if self.training and X.numel() // X.shape[1] == 1:
            X = nn.functional.batch_norm(
                X,
                self.bn2.running_mean,
                self.bn2.running_var,
                self.bn2.weight,
                self.bn2.bias,
                training=False,
                eps=self.bn2.eps,
            )
        else:
            X = self.bn2(X)
        return self.classifier(self.activation(X).flatten(start_dim=1))
