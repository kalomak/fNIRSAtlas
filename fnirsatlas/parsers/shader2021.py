"""
Shader2021 dataset parser for BIDS format data.
Audio/Video/Control experiment dataset.
"""

from pathlib import Path
import logging
from .bids import BIDSParser

logger = logging.getLogger(__name__)


class Shader2021Parser(BIDSParser):
    dataset_id = "shader2021"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.cond2label = {
            "Audio": 1,
            "Control": 2,
            "Video": 3,
        }

    def apply_metadata(self, metadata, data=None, **kwargs):
        # Remove events with trial_type == 'Misc'
        return metadata[metadata["trial_type"] != "Misc"]
