"""fnirsatlas - fNIRS extensions for MOABB."""

__version__ = "0.1.1"
from .benchmark import aggregate_seed_results, fnirs_benchmark
from .datasets import get_dataset, list_datasets
from .datasets.bak2019_fnirs_tapping import Bak2019FnirsTapping
from .datasets.bnci2014_fnirs_ma import Bnci2014FnirsMA
from .datasets.bonnal2025_fnirs_me import Bonnal2025FnirsME
from .datasets.buccino2016_fnirs_me import Buccino2016FnirsME
from .datasets.carlton2026_fnirs_squeezing import Carlton2026FnirsSqueezing
from .datasets.cockx2023_fnirs_me import Cockx2023FnirsME
from .datasets.gao2023_fnirs_squeezing import Gao2023FnirsSqueezing
from .datasets.guevara2024_fnirs_parkinson import Guevara2024FnirsParkinson
from .datasets.herff2014_fnirs_workload import Herff2014FnirsWorkload
from .datasets.huang2021_fnirs_workload import Huang2021FnirsWorkload
from .datasets.klein2026_fnirs_motion import Klein2026FnirsMotion
from .datasets.liu2020_fnirs import Liu2020FnirsControl, Liu2020FnirsStroke
from .datasets.luke2021_fnirs_speech import Luke2021FnirsSpeech
from .datasets.luke2021_fnirs_tapping import Luke2021FnirsTapping
from .datasets.nemo2024_fnirs import Nemo2024FnirsImagery, Nemo2024FnirsPerception
from .datasets.ortega2020_fnirs_grip import Ortega2020FnirsGrip
from .datasets.shader2021_fnirs_speech import Shader2021FnirsSpeech
from .datasets.shin2016_fnirs import Shin2016FnirsMA, Shin2016FnirsMI
from .datasets.shin2018_fnirs import Shin2018FnirsNBack, Shin2018FnirsWG
from .datasets.shin2018_fnirs_ternary import Shin2018FnirsTernary
from .datasets.shin2018b_fnirs_ma import Shin2018bFnirsMA
from .datasets.yi2025_fnirs_mi import Yi2025FnirsMI
from .datasets.yucel2025_fnirs_tapping import Yucel2025FnirsTapping
from .paradigms.fnirs import (
    BaseFNIRS,
    Emotion,
    FingerTapping,
    FNIRSClassification,
    MentalArithmetic,
    MotorExecution,
    MotorImagery,
    NBack,
)
from .pipelines.transformations import (
    ChannelMerger,
    ChannelSplitter,
    StandardScalerND,
    UnsqueezeLayer,
    WindowedFeatureExtractor,
)

__all__ = [
    "Bak2019FnirsTapping",
    "BaseFNIRS",
    "Bnci2014FnirsMA",
    "Bonnal2025FnirsME",
    "Buccino2016FnirsME",
    "Carlton2026FnirsSqueezing",
    "ChannelMerger",
    "ChannelSplitter",
    "Cockx2023FnirsME",
    "Emotion",
    "FNIRSClassification",
    "FingerTapping",
    "Gao2023FnirsSqueezing",
    "Guevara2024FnirsParkinson",
    "Herff2014FnirsWorkload",
    "Huang2021FnirsWorkload",
    "Klein2026FnirsMotion",
    "Liu2020FnirsControl",
    "Liu2020FnirsStroke",
    "Luke2021FnirsSpeech",
    "Luke2021FnirsTapping",
    "MentalArithmetic",
    "MotorExecution",
    "MotorImagery",
    "NBack",
    "Nemo2024FnirsImagery",
    "Nemo2024FnirsPerception",
    "Ortega2020FnirsGrip",
    "Shader2021FnirsSpeech",
    "Shin2016FnirsMA",
    "Shin2016FnirsMI",
    "Shin2018FnirsNBack",
    "Shin2018FnirsTernary",
    "Shin2018FnirsWG",
    "Shin2018bFnirsMA",
    "StandardScalerND",
    "UnsqueezeLayer",
    "WindowedFeatureExtractor",
    "Yi2025FnirsMI",
    "Yucel2025FnirsTapping",
    "aggregate_seed_results",
    "fnirs_benchmark",
    "get_dataset",
    "list_datasets",
]
