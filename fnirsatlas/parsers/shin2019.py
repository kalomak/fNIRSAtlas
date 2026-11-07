import numpy as np
from pathlib import Path
from scipy.io import loadmat
from .base import BaseFNIRSParser
from typing import List
import mne
from fnirsatlas.preprocessing import fill_missing_chs
from fnirsatlas.channels import to_mne_ch_order


class Shin2019Parser(BaseFNIRSParser):
    dataset_id = "shin2019"

    def __init__(self, dataset_path: Path = None, experiment: str = "MA"):
        super().__init__(dataset_path)
        self.experiment = experiment
        self.cond2label = {
            "Idle State": 1,
            "Mental Arithmetic": 2,
            "Motor Imagery": 3,
        }
        self.orig2mne = {
            "Ch12": "S1_D1",
            "Ch5": "S2_D1",
            "Ch18": "S1_D2",
            "Ch11": "S2_D2",
            "Ch4": "S2_D3",
            "Ch17": "S3_D2",
            "Ch10": "S3_D3",
            "Ch3": "S4_D3",
            "Ch16": "S3_D4",
            "Ch9": "S4_D4",
            "Ch2": "S4_D5",
            "Ch15": "S5_D4",
            "Ch8": "S5_D5",
            "Ch1": "S6_D5",
            "Ch14": "S5_D6",
            "Ch7": "S6_D6",
        }

    def load_data(self, subject_id: str, session: int = None) -> dict:
        subject_file = self.dataset_path / f"subject_{subject_id}.mat"
        if not subject_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_file}")
        data = loadmat(subject_file)
        data["sfreq"] = float(data["nfo"][0, 0][0][0])
        return data

    def extract_channels(self, data: dict):
        nfo = data["nfo"][0, 0]
        ch_names = np.array([c[0] for c in nfo[1][0]])
        ch_types = ["hbr" if "deoxy" in str(c) else "hbo" for c in ch_names]
        idx = [
            i
            for i, x in enumerate(nfo[1][0])
            if ("oxy" in str(x[0]) or "deoxy" in str(x[0]))
        ]
        ch_data = np.stack([data[f"ch{i+1}"].squeeze() for i in idx])
        ch_names_mne = []
        for i, ch in enumerate(ch_names):
            ch_type = ch_types[i]
            ch_loc = "".join(filter(str.isdigit, ch))
            ch_loc = self.orig2mne[f"Ch{ch_loc}"]
            ch_names_mne.append(ch_loc + " " + ch_type)
        return ch_data, ch_names_mne, ch_types

    def extract_events(self, data: dict):
        nfo = data["nfo"][0, 0]
        sfreq = float(nfo[0][0])
        mrk = data["mrk"][0, 0]
        event_onsets = (mrk[0].squeeze() * sfreq / 1000).astype(int)
        event_labels = mrk[3][0][0]["desc"].squeeze()
        event_descs = [desc[0] for desc in mrk[2][0]]
        events = np.column_stack(
            [event_onsets, np.zeros_like(event_onsets), event_labels.squeeze()]
        )
        unique_labels = np.unique(event_labels)
        event_desc = {
            int(label): event_descs[i] for i, label in enumerate(unique_labels)
        }
        return events, event_desc

    def _raw_from_session(self, data: dict) -> mne.io.RawArray:
        raw = super()._raw_from_session(data)
        all_chs = [
            f"{ch_loc} {ch_type}"
            for ch_type in ["hbo", "hbr"]
            for ch_loc in self.orig2mne.values()
        ]
        raw = fill_missing_chs(raw, all_chs)
        raw = raw.reorder_channels(to_mne_ch_order(raw.info["chs"]))
        if self.montage is not None:
            raw.set_montage(self.montage)
        with mne.use_log_level("ERROR"):
            raw.interpolate_bads()
        return raw

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        subject_files = list(self.dataset_path.glob("subject_*.mat"))
        subject_ids = [f.stem.split("_")[-1] for f in subject_files]
        subject_ids.sort()
        return subject_ids
