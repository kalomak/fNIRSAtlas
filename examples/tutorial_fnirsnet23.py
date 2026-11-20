"""
===========================
Evaluate fNIRSNet23 on BNCI
===========================

Evaluate a native PyTorch model, fNIRSNet23, on the BNCI mental arithmetic
dataset with a MOABB within-session evaluation.

Requires the ``dl`` extra.
"""

# Authors: Kalle Mäkelä
#
# License: MIT

from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import make_pipeline

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import StandardScalerND
from fnirsatlas.pipelines.dl import FNIRSClassifier, fNIRSNet23

# %%
# Define the pipeline
# -------------------
#
# ``StandardScalerND`` standardizes each chromophore over trials, channels,
# and time. ``FNIRSClassifier`` wraps a PyTorch module as a scikit-learn
# estimator. It infers the input shape and number of classes on fit, and
# ``module__`` parameters are passed to the module.

pipeline = make_pipeline(
    StandardScalerND(dim=(0, 2, 3)),
    FNIRSClassifier(
        module=fNIRSNet23,
        module__F1=4,
        lr=0.01,
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
results = evaluation.process({"fNIRSNet23": pipeline})
print(results[["subject", "session", "score"]].to_string(index=False))
