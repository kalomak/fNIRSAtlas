"""
NEMO dataset parser for BIDS format data.
"""

from pathlib import Path
import logging
from typing import List, Dict, Tuple
import numpy as np
from .bids import BIDSParser

logger = logging.getLogger(__name__)


class NEMOParser(BIDSParser):
    dataset_id = "nemo"
    EXPERIMENTS = ["empe", "afim"]

    def __init__(self, dataset_path: Path = None, experiment="empe"):
        super().__init__(dataset_path)
        self.experiment = experiment
        self.cond2label = {"HANV": 1, "HAPV": 2, "LANV": 3, "LAPV": 4}
        if self.experiment == "afim":
            self.cond2label = {f"afim_{k}": v for k, v in self.cond2label.items()}

    def get_experiments(self, subject_id: str, session: str = None) -> List[str]:
        return self.EXPERIMENTS  # just to reorder

    def _raw_from_session(self, data: dict):
        raw = super()._raw_from_session(data)
        if self.montage is not None:
            try:
                raw.set_montage(self.montage)
            except Exception as e:
                logger.warning(f"Could not set montage for NEMO dataset: {e}")
        return raw
