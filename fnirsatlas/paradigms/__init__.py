"""fnirsatlas paradigms."""

from .conversion import epochs_to_array
from .fnirs import (
    BaseFNIRS,
    Emotion,
    FingerTapping,
    FNIRSClassification,
    MentalArithmetic,
    MotorExecution,
    MotorImagery,
    NBack,
)

__all__ = [
    "BaseFNIRS",
    "Emotion",
    "FingerTapping",
    "FNIRSClassification",
    "MentalArithmetic",
    "MotorExecution",
    "MotorImagery",
    "NBack",
    "epochs_to_array",
]
