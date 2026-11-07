import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from fnirsatlas.dl.DLModelBase import DLModelBase
from fnirsatlas.feature_extraction import StandardScalerND, ChannelSplitter
from fnirsatlas.classification import run_benchmark


class SimpleNet(DLModelBase):
    def __init__(self, T, n_chs, num_classes, p_dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.T = T
        self.n_chs = n_chs
        self.num_classes = num_classes
        self.p_dropout = p_dropout
        self.layers = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=(1, T)),
            nn.ELU(),
            nn.Dropout(self.p_dropout),
            nn.Conv2d(4, 8, kernel_size=(n_chs, 1)),
            nn.ELU(),
            nn.Dropout(self.p_dropout),
            nn.Flatten(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x):
        return self.layers(x)

    def _on_epoch_end(self, optimizer):
        if not self.epoch % 10:
            print(f"Epoch: {self.epoch}, LR: {optimizer.param_groups[0]['lr']}")


def get_clf(**kwargs):
    random_state = kwargs["random_state"]
    num_classes = np.unique(kwargs["y"]).shape[0]
    ch_names = kwargs["info"]["ch_names"]
    T = kwargs["X"].shape[-1]
    n_chs = len(ch_names) // 2
    torch.manual_seed(random_state)
    return Pipeline(
        [
            ("splitter", ChannelSplitter(ch_names)),
            ("scaler", StandardScalerND(dim=(0, 2, 3))),
            ("clf", SimpleNet(T, n_chs, num_classes)),
        ]
    )


res = run_benchmark(
    get_clf=get_clf,
    suite="ma",
    classification_setting="cross",
    seed=42,
    param_grid={
        "clf__p_dropout": [0.1, 0.5, 0.9],
    },
)
