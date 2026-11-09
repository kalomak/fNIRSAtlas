"""fnirsatlas datasets."""

from .bak2019_fnirs_tapping import Bak2019FnirsTapping
from .base import BaseFNIRSDataset
from .bids_base import BaseBIDSFNIRSDataset
from .bnci2014_fnirs_ma import Bnci2014FnirsMA
from .buccino2016_fnirs_me import Buccino2016FnirsME
from .cockx2023_fnirs_me import Cockx2023FnirsME
from .gao2023_fnirs_squeezing import Gao2023FnirsSqueezing
from .guevara2024_fnirs_parkinson import Guevara2024FnirsParkinson
from .herff2014_fnirs_workload import Herff2014FnirsWorkload
from .huang2021_fnirs_workload import Huang2021FnirsWorkload
from .luke2021_fnirs_speech import Luke2021FnirsSpeech
from .luke2021_fnirs_tapping import Luke2021FnirsTapping
from .nemo2024_fnirs import Nemo2024FnirsImagery, Nemo2024FnirsPerception
from .ortega2020_fnirs_grip import Ortega2020FnirsGrip
from .shader2021_fnirs_speech import Shader2021FnirsSpeech
from .shin2016_fnirs import Shin2016FnirsMA, Shin2016FnirsMI
from .shin2018_fnirs import Shin2018FnirsNBack, Shin2018FnirsWG
from .shin2018_fnirs_ternary import Shin2018FnirsTernary
from .shin2018b_fnirs_ma import Shin2018bFnirsMA
from .yi2025_fnirs_mi import Yi2025FnirsMI
from .yucel2025_fnirs_tapping import Yucel2025FnirsTapping

# Datasets paradigms use by default. Other exported datasets load only when named.
BENCHMARK_DATASETS = (
    Bak2019FnirsTapping,
    Bnci2014FnirsMA,
    Buccino2016FnirsME,
    Cockx2023FnirsME,
    Gao2023FnirsSqueezing,
    Guevara2024FnirsParkinson,
    Herff2014FnirsWorkload,
    Huang2021FnirsWorkload,
    Luke2021FnirsTapping,
    Luke2021FnirsSpeech,
    Yucel2025FnirsTapping,
    Nemo2024FnirsImagery,
    Nemo2024FnirsPerception,
    Ortega2020FnirsGrip,
    Shader2021FnirsSpeech,
    Shin2016FnirsMA,
    Shin2016FnirsMI,
    Shin2018FnirsNBack,
    Shin2018FnirsWG,
    Shin2018bFnirsMA,
    Shin2018FnirsTernary,
    Yi2025FnirsMI,
)


def list_datasets() -> list[str]:
    """Names of all exported fNIRS datasets.

    Returns
    -------
    list of str
        Class names accepted by :func:`get_dataset`.
    """
    bases = (BaseFNIRSDataset, BaseBIDSFNIRSDataset)
    return [
        name
        for name in __all__
        if isinstance(cls := globals()[name], type)
        and issubclass(cls, BaseFNIRSDataset)
        and cls not in bases
    ]


def get_dataset(name: str, **kwargs) -> BaseFNIRSDataset:
    """Instantiate an exported dataset by class name.

    Parameters
    ----------
    name : str
        Dataset class name, e.g. ``"Bnci2014FnirsMA"``. See :func:`list_datasets`.
    **kwargs
        Passed to the dataset constructor.

    Returns
    -------
    BaseFNIRSDataset
        The dataset instance.
    """
    names = list_datasets()
    if name not in names:
        raise ValueError(f"Unknown dataset {name!r}. Available: {', '.join(names)}")
    return globals()[name](**kwargs)


__all__ = [
    "BENCHMARK_DATASETS",
    "Bak2019FnirsTapping",
    "BaseBIDSFNIRSDataset",
    "BaseFNIRSDataset",
    "Bnci2014FnirsMA",
    "Buccino2016FnirsME",
    "Cockx2023FnirsME",
    "Gao2023FnirsSqueezing",
    "Guevara2024FnirsParkinson",
    "Herff2014FnirsWorkload",
    "Huang2021FnirsWorkload",
    "Luke2021FnirsSpeech",
    "Luke2021FnirsTapping",
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
    "Yi2025FnirsMI",
    "Yucel2025FnirsTapping",
    "get_dataset",
    "list_datasets",
]
