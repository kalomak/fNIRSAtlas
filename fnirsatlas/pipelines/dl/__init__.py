"""Deep-learning estimators and models for fNIRS."""

from .classifier import FNIRSClassifier
from .cnn2a import CNN2a
from .dcn import DCN
from .eegnet import EEGNet
from .fnirs_transformer import fNIRS_T
from .fnirsnet23 import fNIRSNet23
from .fnirsnet24 import fNIRSNet24
from .lstm import LSTM
from .pipeline import make_dl_pipeline

__all__ = [
    "DCN",
    "LSTM",
    "CNN2a",
    "EEGNet",
    "FNIRSClassifier",
    "fNIRSNet23",
    "fNIRSNet24",
    "fNIRS_T",
    "make_dl_pipeline",
]
