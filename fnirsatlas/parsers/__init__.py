"""
Parser module for fnirsatlas.

Contains all dataset parsers and a factory function to get parser classes by name.
"""

from .base import BaseFNIRSParser
from .bids import BIDSParser
from .bak2019 import Bak2019Parser
from .shin2016 import Shin2016Parser
from .shin2019 import Shin2019Parser
from .shin2018 import Shin2018Parser
from .shin2018b import Shin2018bParser
from .luke2021 import Luke2021Parser
from .luke2024a import Luke2024aParser
from .luke2024b import Luke2024bParser
from .nemo import NEMOParser
from .bnci import BNCIParser
from .cockx2023 import Cockx2023Parser
from .shader2021 import Shader2021Parser
from .herff2014 import Herff2014Parser
from .ortega2020 import Ortega2020Parser
from .buccino2016 import Buccino2016Parser
from .yi2025 import Yi2025Parser
from .guevara2024 import Guevara2024Parser
from .huang2021 import Huang2021Parser
from .gao2024 import Gao2024Parser
from fnirsatlas.dataset_catalog import (
    SUITE_MAP,
    get_suite_datasets,
    get_available_datasets,
    get_suite_for_dataset,
)

# Dictionary mapping dataset names to parser classes
PARSER_MAP: dict[str, type[BaseFNIRSParser]] = {
    "bids": BIDSParser,
    "bak2019": Bak2019Parser,
    "shin2016": Shin2016Parser,
    "shin2019": Shin2019Parser,
    "shin2018": Shin2018Parser,
    "shin2018b": Shin2018bParser,
    "luke2021": Luke2021Parser,
    "luke2024a": Luke2024aParser,
    "luke2024b": Luke2024bParser,
    "nemo": NEMOParser,
    "bnci": BNCIParser,
    "cockx2023": Cockx2023Parser,
    "shader2021": Shader2021Parser,
    "herff2014": Herff2014Parser,
    "ortega2020": Ortega2020Parser,
    "buccino2016": Buccino2016Parser,
    "yi2025": Yi2025Parser,
    "guevara2024": Guevara2024Parser,
    "huang2021": Huang2021Parser,
    "gao2024": Gao2024Parser,
}


def get_parser_class(dataset_name: str) -> type[BaseFNIRSParser]:
    """
    Get parser class for a given dataset name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'bak2019', 'ortega2020')

    Returns
    -------
    class
        Parser class for the dataset

    Raises
    ------
    ValueError
        If dataset name is not supported
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in PARSER_MAP:
        available = list(PARSER_MAP.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Available datasets: {available}"
        )
    return PARSER_MAP[dataset_name]


__all__ = [
    "BaseFNIRSParser",
    "BIDSParser",
    "Bak2019Parser",
    "Shin2016Parser",
    "Shin2019Parser",
    "Shin2018Parser",
    "Shin2018bParser",
    "Luke2021Parser",
    "Luke2024aParser",
    "Luke2024bParser",
    "NEMOParser",
    "BNCIParser",
    "Cockx2023Parser",
    "Shader2021Parser",
    "Herff2014Parser",
    "Ortega2020Parser",
    "Buccino2016Parser",
    "Yi2025Parser",
    "Guevara2024Parser",
    "Huang2021Parser",
    "Gao2024Parser",
    "get_parser_class",
    "PARSER_MAP",
    "get_suite_datasets",
    "get_available_datasets",
    "get_suite_for_dataset",
]
