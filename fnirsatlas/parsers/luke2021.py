"""
Luke2021 dataset parser for BIDS format data.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from .bids import BIDSParser

logger = logging.getLogger(__name__)


class Luke2021Parser(BIDSParser):
    """Luke2021 dataset parser that extends BIDSParser with dataset-specific logic."""

    dataset_id = "luke2021"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.cond2label = {
            "Control": 1,
            "Tapping/Left": 2,
            "Tapping/Right": 3,
        }

    def apply_metadata(self, metadata: pd.DataFrame, data: dict = None) -> pd.DataFrame:
        """Filter out events with trial_type '15.0'."""
        return metadata[metadata["trial_type"] != "15.0"]
