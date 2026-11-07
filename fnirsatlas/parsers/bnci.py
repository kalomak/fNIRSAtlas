import numpy as np
import pandas as pd
from pathlib import Path
from .base import BaseFNIRSParser
from typing import List, Dict, Any
import mne
from scipy.io import loadmat


ch_names_ordered = [
    # row 1, from down to up
    "S1_D1",
    "S2_D1",
    "S2_D2",
    "S3_D2",
    "S3_D3",
    "S4_D3",
    "S4_D4",
    "S5_D4",
    "S5_D5",
    "S6_D5",
    # row 2
    "S1_D6",
    "S7_D1",
    "S2_D7",
    "S8_D2",
    "S3_D8",
    "S9_D3",
    "S4_D9",
    "S10_D4",
    "S5_D10",
    "S11_D5",
    "S6_D11",
    # row 3
    "S7_D6",
    "S7_D7",
    "S8_D7",
    "S8_D8",
    "S9_D8",
    "S9_D9",
    "S10_D9",
    "S10_D10",
    "S11_D10",
    "S11_D11",
    # row 4
    "S12_D6",
    "S7_D12",
    "S13_D7",
    "S8_D13",
    "S14_D8",
    "S9_D14",
    "S15_D9",
    "S10_D15",
    "S16_D10",
    "S11_D16",
    "S17_D11",
    # row 5
    "S12_D12",
    "S13_D12",
    "S13_D13",
    "S14_D13",
    "S14_D14",
    "S15_D14",
    "S15_D15",
    "S16_D15",
    "S16_D16",
    "S17_D16",
]


class BNCIParser(BaseFNIRSParser):
    dataset_id = "bnci"
    """Parser for BNCI Horizon 2020 Dataset Mental arithmetic (003-2014)."""

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.sfreq = 10  # Sampling frequency is 10 Hz
        self.cond2label = {
            # change the names to match e.g. Shin2016
            "MA": 1,
            "BL": 2,
        }
        self.sessions = {
            "S01": [1, 2, 3],
            "S02": [1, 2, 3],
            "S03": [1, 2, 3],
            "S04": [1, 2, 3, 4],
            "S05": [1, 2, 3, 4],
            "S06": [1, 2, 3, 4],
            "S07": [1, 2, 3, 4],
            "S08": [1, 2, 3, 4],
        }

    def get_sessions(self, subject_id: str) -> list[str]:
        """Get available sessions for a subject."""
        return self.sessions[subject_id]

    def load_data(self, subject_id: str, session: str = None) -> Dict:
        mat_file = self.dataset_path / f"{subject_id}.mat"
        mat_data = loadmat(mat_file, squeeze_me=True, struct_as_record=False)
        data_array = mat_data["data"]
        data = {
            "subject_id": subject_id,
            "sfreq": self.sfreq,
        }

        # Process each session
        # hb_data = {}
        # events_data = {}
        # for idx, session_label in enumerate(sessions):
        # Extract session data from the structured array
        session_idx = self.sessions[subject_id].index(session)
        session_data = data_array[session_idx]
        X = session_data.X  # Shape: (channels, timepoints)
        fs = session_data.fs
        trial_samples = session_data.trial.flatten()
        trial_labels = session_data.y.flatten()

        # MODIFICATION: the original starts the rest epoch immediately after the end of the previous MA epoch, meaning that the effect of the MA epoch will be carried over to the rest epoch. We add 8 second separation between the end of the MA epoch and the start of the rest epoch. Original: (12s MA -> 28s rest), new: (12s MA + 8s space (total 20s) > 20s rest).
        rest_trigger_sample_ix = trial_labels == self.cond2label["BL"]
        trial_samples[rest_trigger_sample_ix] += int(8 * self.sfreq - 1)

        hb_data = X
        events_data = {
            "trigger_type": trial_labels,
            "trigger_sample": trial_samples,
        }
        data["hb_data"] = hb_data
        data["events_data"] = events_data
        return data

    def extract_channels(self, data: Dict) -> tuple:
        """Extract channel data for a specific session."""
        hb_data = data["hb_data"]

        # hb_data shape: (n_times, 156) [52 hbo, 52 hbr, 52 hbt]
        # Only use hbo and hbr (first 104 columns)
        ch_data = hb_data[:, :104].T  # shape: (104, n_times)

        # Create channel names for hbo and hbr
        ch_names = [f"{ch} hbo" for ch in ch_names_ordered] + [
            f"{ch} hbr" for ch in ch_names_ordered
        ]
        ch_types = ["hbo"] * len(ch_names_ordered) + ["hbr"] * len(ch_names_ordered)

        if ch_data.shape[0] != 104:
            raise ValueError(
                f"Expected 104 channels (52 hbo + 52 hbr), got {ch_data.shape[0]}"
            )

        return ch_data, ch_names, ch_types

    def extract_events(self, data: Dict) -> tuple:
        """Extract events for a specific session."""
        events_data = data["events_data"]
        trigger_sample = events_data["trigger_sample"]
        trigger_type = events_data["trigger_type"]
        events = np.column_stack(
            [trigger_sample, np.zeros(len(trigger_sample), dtype=int), trigger_type]
        )
        event_desc = {v: k for k, v in self.cond2label.items()}
        return events, event_desc

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        return ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"]
