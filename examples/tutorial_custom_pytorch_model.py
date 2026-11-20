"""
===============================
Evaluate a custom PyTorch model
===============================

Define a PyTorch module, wrap it as a scikit-learn estimator, and evaluate it
on the BNCI mental arithmetic dataset with a MOABB within-session evaluation.

Requires the ``dl`` extra.
"""

# Authors: Kalle Mäkelä
#
# License: MIT

from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.optim.lr_scheduler import StepLR

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import StandardScalerND
from fnirsatlas.pipelines.dl import FNIRSClassifier

# %%
# Define the module
# -----------------
#
# The paradigm returns epochs shaped ``(trials, chromophores, channels,
# timepoints)``. ``FNIRSClassifier`` infers ``n_chromophores``, ``n_channels``,
# ``n_times``, and ``n_outputs`` when fitting each evaluation fold and passes
# them to the module.


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


# %%
# Define the pipeline
# -------------------
#
# ``StandardScalerND`` standardizes each chromophore over trials, channels,
# and time. ``module__``, ``criterion__``, and ``scheduler__`` parameters are
# passed to the module, the loss, and the learning-rate scheduler.

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

# %%
# Evaluate
# --------

evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA(subjects=[1])],
    n_splits=5,
    random_state=42,
)
results = evaluation.process({"LinearModule": pipeline})
print(results[["subject", "session", "score"]].to_string(index=False))
