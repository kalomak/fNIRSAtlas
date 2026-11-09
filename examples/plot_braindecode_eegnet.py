"""
===================================
Evaluate Braindecode EEGNet on BNCI
===================================

Evaluate the Braindecode EEGNet on the BNCI mental arithmetic dataset with a
MOABB cross-session evaluation.

Requires the ``braindecode`` extra.
"""

# Authors: Kalle Mäkelä
#
# License: MIT

from braindecode import EEGClassifier
from braindecode.models import EEGNet
from moabb.evaluations import CrossSessionEvaluation
from sklearn.pipeline import make_pipeline

from fnirsatlas import Bnci2014FnirsMA, FNIRSClassification
from fnirsatlas.pipelines import ChannelMerger

# %%
# Define the classifier
# ---------------------
#
# Braindecode models expect ``(trials, channels, timepoints)``, so
# ``ChannelMerger`` merges the chromophore dimension into the channels. The
# kernel and pooling sizes are smaller than the Braindecode defaults.

pipeline = make_pipeline(
    ChannelMerger(),
    EEGClassifier(
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
    ),
)

# %%
# Evaluate
# --------

evaluation = CrossSessionEvaluation(
    paradigm=FNIRSClassification(),
    datasets=[Bnci2014FnirsMA(subjects=[1])],
)
results = evaluation.process({"Braindecode EEGNet": pipeline})
print(results[["subject", "session", "score"]].to_string(index=False))
