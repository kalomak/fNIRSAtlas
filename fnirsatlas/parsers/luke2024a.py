from ast import Dict
from pathlib import Path
from typing import List
import warnings
import logging
import mne
from fnirsatlas.parsers.bids import BIDSParser

logger = logging.getLogger(__name__)


class Luke2024aParser(BIDSParser):
    dataset_id = "luke2024a"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
