# Running a Custom PyTorch Model

Install PyTorch support:

```bash
uv sync --extra dl
```

Define a module for input shaped `(trials, chromophores, channels, timepoints)`.
{class}`fnirsatlas.pipelines.dl.FNIRSClassifier` infers the signal dimensions when
fitting each evaluation fold:

```python
from torch import nn


class LinearModule(nn.Module):
    def __init__(
        self,
        n_chromophores,
        n_channels,
        n_times,
        n_outputs,
        hidden_features=32,
    ):
        super().__init__()
        n_features = n_chromophores * n_channels * n_times
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, n_outputs),
        )

    def forward(self, X):
        return self.layers(X)
```

An fNIRS paradigm returns arrays shaped ``(trials, chromophores, channels,
timepoints)``.

```python
from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import StepLR

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import StandardScalerND
from fnirsatlas.pipelines.dl import FNIRSClassifier


pipeline = make_pipeline(
    StandardScalerND(dim=(0, 2, 3)),
    FNIRSClassifier(
        module=LinearModule,
        module__hidden_features=64,
        criterion__label_smoothing=0.0,
        scheduler=StepLR,
        scheduler__step_size=20,
        scheduler__gamma=0.5,
        lr=0.01,
        max_epochs=20,
        device="auto",
        random_state=42,
    ),
)
evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA()],
)
results = evaluation.process({"LinearModule": pipeline})
print(results)
```
