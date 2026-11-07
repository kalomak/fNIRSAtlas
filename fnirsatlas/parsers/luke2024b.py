from ast import Dict
from pathlib import Path
import logging
import pandas as pd
from fnirsatlas.parsers.bids import BIDSParser

logger = logging.getLogger(__name__)
CANONICAL_SFREQ = 8.928571428571429


class Luke2024bParser(BIDSParser):
    """Motor dataset from the FRESH project. L/R Finger Tapping."""
    dataset_id = "luke2024b"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.cond2label = {
            "left2s": 1,
            "right2s": 2,
            "left3s": 3,
            "right3s": 4,
        }

    def get_events_from_tsv(self, data: Dict) -> pd.DataFrame:
        """Each trial has its own session."""
        metadata = super().get_events_from_tsv(data)
        session = data["session"]
        metadata["trial_type"] = session
        return metadata

    def _raw_from_session(self, data: dict):
        raw = super()._raw_from_session(data)
        # Luke2024b sessions carry tiny sfreq float differences that make MNE
        # derive slightly different epoch baselines and fail epoch concatenation.
        with raw.info._unlock():
            raw.info["sfreq"] = CANONICAL_SFREQ
        return raw
