from pathlib import Path

import pandas as pd

from fnirsatlas.parsers.bids import BIDSParser


class Gao2024Parser(BIDSParser):
    """Parser for the historical Gao BallSqueezing benchmark in ds007420."""

    dataset_id = "gao2024"
    subjects = {
        "170", "171", "173", "174", "176", "177", "179", "181", "182",
        "183", "184", "185",
    }

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.experiment = "BallSqueezing"
        self.cond2label = {
            "Left": 1,
            "Right": 2,
        }

    def get_subject_ids(self) -> list[str]:
        """Return the 12 subjects included in the original Gao dataset."""
        subjects = self.subjects.intersection(self._get_bids_sub_ids())
        return [self._from_bids_sub_id(subject) for subject in sorted(subjects)]

    def apply_metadata(self, metadata: pd.DataFrame, data: dict = None) -> pd.DataFrame:
        """Restore hand labels from the replacement dataset's numeric codes."""
        metadata = metadata.copy()
        metadata["trial_type"] = metadata["trial_type"].astype(str).replace(
            {"1": "Left", "2": "Right"}
        )
        return metadata
