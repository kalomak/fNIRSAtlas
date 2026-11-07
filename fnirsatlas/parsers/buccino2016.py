import numpy as np
from pathlib import Path
from scipy.io import loadmat
from .base import BaseFNIRSParser
from typing import Dict, List, Tuple, Any

orig2mne = {
    # left
    "Fc3A": "S1_D1",
    "Fc1A": "S1_D2",
    "Fc3L": "S2_D1",
    "Fc3M": "S1_D3",
    "Fc1M": "S3_D2",
    "C3A": "S2_D3",
    "C1A": "S3_D3",
    "C3L": "S2_D4",
    "C3M": "S4_D3",
    "C1M": "S3_D5",
    "Cp3A": "S4_D4",
    "Cp1A": "S4_D5",
    "Cp3L": "S5_D4",
    "Cp3M": "S4_D6",
    "Cp1M": "S6_D5",
    "Cp3P": "S5_D6",
    "Cp1P": "S6_D6",
    # right
    "Fc2A": "S7_D7",
    "Fc4A": "S7_D8",
    "Fc2M": "S8_D7",
    "Fc4M": "S7_D9",
    "Fc4L": "S9_D8",
    "C2A": "S8_D9",
    "C4A": "S9_D9",
    "C2M": "S8_D10",
    "C4M": "S10_D9",
    "C4L": "S9_D11",
    "Cp2A": "S10_D10",
    "Cp4A": "S10_D11",
    "Cp2M": "S11_D10",
    "Cp4M": "S10_D12",
    "Cp4L": "S12_D11",
    "Cp2P": "S11_D12",
    "Cp4P": "S12_D12",
}


class Buccino2016Parser(BaseFNIRSParser):
    dataset_id = "buccino2016"
    """Parser for Buccino2016 EEG-fNIRS hybrid SMR BCI dataset."""

    def __init__(self, dataset_path: Path = None, keep_rest_events: bool = False):
        super().__init__(dataset_path)
        self.orig2mne = orig2mne

        # NIRS channel mapping (34 channels)
        self.nirs_channels = [
            "Fc3A",
            "Fc1A",
            "Fc3L",
            "Fc3M",
            "Fc1M",
            "C3A",
            "C1A",
            "C3L",
            "C3M",
            "C1M",
            "Cp3A",
            "Cp1A",
            "Cp3L",
            "Cp3M",
            "Cp1M",
            "Cp3P",
            "Cp1P",
            "Fc2A",
            "Fc4A",
            "Fc2M",
            "Fc4M",
            "Fc4L",
            "C2A",
            "C4A",
            "C2M",
            "C4M",
            "C4L",
            "Cp2A",
            "Cp4A",
            "Cp2M",
            "Cp4M",
            "Cp4L",
            "Cp2P",
            "Cp4P",
        ]
        self.keep_rest_events = keep_rest_events

    def load_data(self, subject_id: str, session: Any = None) -> Dict:
        """Load data for a specific subject."""
        if int(subject_id[1:]) <= 8:
            subject_dir = self.dataset_path / "EEG-fNIRS hybrid SMR BCI data S1-S8"
        else:
            subject_dir = self.dataset_path / "EEG-fNIRS hybrid SMR BCI data S9-S15"
        subject_file = subject_dir / f"{subject_id}.mat"
        if not subject_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_file}")
        mat_data = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
        data = mat_data["data"]
        nirs_data = data[1]  # NIRS data is at index 1
        return {
            "sfreq": nirs_data[0],  # Sampling frequency (already squeezed)
            "nirs_n_channels": nirs_data[1],  # Number of channels (already squeezed)
            "nirs_labels": nirs_data[2],  # Channel labels
            "nirs_time_points": nirs_data[3],  # Time points (already squeezed)
            "nirs_wl1": nirs_data[4],  # Wavelength 1 data
            "nirs_wl2": nirs_data[5],  # Wavelength 2 data
            "events": data[2],  # Event data
        }

    def extract_channels(
        self, data: Dict, session: Any = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract NIRS channel data."""
        wl1_data = data["nirs_wl1"]  # Shape: (n_timepoints, n_channels)
        wl2_data = data["nirs_wl2"]  # Shape: (n_timepoints, n_channels)
        ch_data = np.concatenate(
            [wl1_data, wl2_data], axis=1
        ).T  # (n_channels, n_timepoints)
        ch_names, ch_types = [], []
        for i, ch_name in enumerate(self.nirs_channels):
            mne_ch_name = self.orig2mne[ch_name]
            ch_names.extend([f"{mne_ch_name} 760", f"{mne_ch_name} 850"])
            ch_types.extend(["fnirs_cw_amplitude", "fnirs_cw_amplitude"])
        return ch_data, ch_names, ch_types

    def extract_events(
        self, data: Dict, session: Any = None
    ) -> Tuple[np.ndarray, Dict]:
        """Extract events from the event matrix."""
        events_raw = data["events"]  # Shape: (n_events, 2)
        sfreq = data["sfreq"]  # Changed from "nirs_sfreq" to "sfreq"
        events = []
        cond2label = {
            "rest": 1,
            "right_arm": 3,
            "left_arm": 4,
            "right_hand": 7,
            "left_hand": 8,
        }
        label2cond = {v: k for k, v in cond2label.items()}
        # Event codes: 1=rest, 3=right_arm, 4=left_arm, 7=right_hand, 8=left_hand, 10=block_start
        for event_time, event_code in events_raw:
            sample_idx = int(event_time * sfreq)
            if event_code in cond2label.values():
                event_id = int(event_code)
                if not self.keep_rest_events and event_id == 1:  # rest event
                    continue
                events.append([sample_idx, 0, event_id])
        event_desc = label2cond
        return np.array(events), event_desc

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        return [
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
            "S8",
            "S9",
            "S10",
            "S11",
            "S12",
            "S13",
            "S14",
            "S15",
        ]
