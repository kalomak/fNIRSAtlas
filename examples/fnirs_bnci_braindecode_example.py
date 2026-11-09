"""
Evaluate Braindecode EEGNet on BNCI
====================================

Evaluate Braindecode EEGNet on the BNCI fNIRS dataset through MOABB.
"""

from braindecode import EEGClassifier
from braindecode.models import EEGNet
from moabb.evaluations import WithinSessionEvaluation

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification

classifier = EEGClassifier(
    module=EEGNet,
    module__kernel_length=16,
    module__depthwise_kernel_length=8,
    module__pool1_kernel_size=2,
    module__pool2_kernel_size=2,
    module__drop_prob=0.1,
    max_epochs=4,
    batch_size=8,
    train_split=None,
    verbose=0,
    device="cpu",
)
evaluation = WithinSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA(subjects=[1])],
)
results = evaluation.process({"Braindecode EEGNet": classifier})
print(results[["subject", "session", "score"]].to_string(index=False))
