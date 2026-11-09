"""
Evaluate EEGNet on BNCI
=======================

Evaluate the native fNIRS EEGNet on the BNCI dataset through MOABB.
"""

from moabb.evaluations import WithinSessionEvaluation
from sklearn.pipeline import make_pipeline
from torch.optim.lr_scheduler import StepLR

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import StandardScalerND
from fnirsatlas.pipelines.dl import EEGNet, FNIRSClassifier

pipeline = make_pipeline(
    StandardScalerND(dim=(0, 2, 3)),
    FNIRSClassifier(
        module=EEGNet,
        module__kernel_size=3,
        criterion__label_smoothing=0.0,
        scheduler=StepLR,
        scheduler__step_size=20,
        scheduler__gamma=0.5,
        lr=0.01,
        device="auto",
        random_state=42,
    ),
)
evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA(subjects=[1])],
    n_splits=5,
    random_state=42,
)
results = evaluation.process({"EEGNet": pipeline})
print(results[["subject", "session", "score"]].to_string(index=False))
