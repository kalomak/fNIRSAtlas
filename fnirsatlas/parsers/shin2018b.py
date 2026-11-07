import numpy as np
from pathlib import Path
from scipy.io import loadmat
from .base import BaseFNIRSParser
from typing import List, Dict, Tuple
import mne
from fnirsatlas.montage import load_montage_from_file


CORRECTED_CH_NAMES = [
    "S1_D1",
    "S2_D1",
    "S2_D2",
    "S3_D1",
    "S3_D2",
    "S3_D3",
    "S4_D2",
    "S4_D3",
    "S5_D3",
]


class Shin2018bParser(BaseFNIRSParser):
    dataset_id = "shin2018b"
    USE_LEGACY_SENSORS = True
    # The saved montage uses inferred optode locations and corrected S/D pairs.
    # It is better than the legacy colocated sensors, but the old layout remains
    # the default to avoid breaking downstream assumptions right now.

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.nirs_path = self.dataset_path / "NIRS"
        self.cond2label = {"BL": 1, "MA": 2}
        if self.USE_LEGACY_SENSORS:
            self.montage = None

    def _get_legacy_montage(self, data: Dict) -> mne.channels.DigMontage:
        """Create the legacy montage that reuses channel centers as optodes."""
        mnt = data["mnt"]
        scale = 0.08
        coord_frame = "head"
        nasion = np.array([0.00146763, 0.08506715, -0.03483611])
        lpa = np.array([-0.08061612, -0.02908875, -0.04131077])
        rpa = np.array([0.08436285, -0.02850276, -0.04127743])
        ch_pos = {}
        n_ch = len(mnt.clab)
        for i in range(n_ch):
            ch_pos[f"S{i + 1}"] = mnt.pos_3d[:, i] * scale
            ch_pos[f"D{i + 1}"] = mnt.pos_3d[:, i] * scale
        return mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame=coord_frame,
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
        )

    def load_data(self, subject_id: str, session: int = None) -> Dict:
        """Load data for a specific subject."""
        sub_dir = self.nirs_path / subject_id
        cnt = loadmat(sub_dir / "cnt.mat", squeeze_me=True, struct_as_record=False)[
            "cnt"
        ]
        mrk = loadmat(sub_dir / "mrk.mat", squeeze_me=True, struct_as_record=False)[
            "mrk"
        ]
        mnt = loadmat(sub_dir / "mnt.mat", squeeze_me=True, struct_as_record=False)[
            "mnt"
        ]
        sfreq = float(cnt.oxy.fs)
        return {"cnt": cnt, "mrk": mrk, "mnt": mnt, "sfreq": sfreq}

    def get_montage(self, data: Dict = None) -> mne.channels.DigMontage:
        """Get the Shin2018b montage using either legacy or inferred sensors."""
        if self.USE_LEGACY_SENSORS:
            return self._get_legacy_montage(data)
        self.montage = load_montage_from_file(self.dataset_id)
        return self.montage

    def extract_channels(self, data: dict) -> Tuple[np.ndarray, List[str], List[str]]:
        """Extract channel data, names, and types."""
        cnt = data["cnt"]

        # Extract oxy and deoxy data
        oxy_data = cnt.oxy.x.T  # (n_channels, n_timepoints)
        deoxy_data = cnt.deoxy.x.T

        # Create channel names following MNE NIRS convention
        ch_names = []
        ch_types = []

        if self.USE_LEGACY_SENSORS:
            mne_ch_names = [f"S{i + 1}_D{i + 1}" for i in range(len(cnt.oxy.clab))]
        else:
            mne_ch_names = CORRECTED_CH_NAMES

        # Add oxy channels
        for i, mne_ch_name in enumerate(mne_ch_names):
            ch_names.append(f"{mne_ch_name} hbo")
            ch_types.append("hbo")

        # Add deoxy channels
        for i, mne_ch_name in enumerate(mne_ch_names):
            ch_names.append(f"{mne_ch_name} hbr")
            ch_types.append("hbr")

        # Concatenate data
        ch_data = np.vstack([oxy_data, deoxy_data])

        return ch_data, ch_names, ch_types

    def extract_events(self, data: dict) -> Tuple[np.ndarray, Dict]:
        """Extract events and event descriptions."""
        mrk = data["mrk"]
        sfreq = self.get_sampling_frequency(data)

        # Use oxy events (should be same as deoxy)
        event_times = mrk.oxy.time
        event_descs = mrk.oxy.event.desc

        # Convert time to sample indices
        idx = (event_times * sfreq / 1000).round().astype(int)

        # Create events array: [sample, 0, event_id]
        events = np.column_stack([idx, np.zeros_like(idx), event_descs])

        # Create event description mapping
        event_desc = {}
        for i, class_name in enumerate(mrk.oxy.className):
            event_desc[i + 1] = class_name

        return events, event_desc

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        subject_dirs = [
            d
            for d in self.nirs_path.iterdir()
            if d.is_dir() and d.name.startswith("VP")
        ]
        subject_ids = [d.name for d in subject_dirs]
        subject_ids.sort()
        return subject_ids

    def _raw_from_session(self, data: Dict) -> mne.io.RawArray:
        raw = super()._raw_from_session(data)
        montage = self.get_montage(data)
        raw.set_montage(montage)
        return raw
