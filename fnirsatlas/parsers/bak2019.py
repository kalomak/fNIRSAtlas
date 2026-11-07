import numpy as np
from pathlib import Path
from scipy.io import loadmat
from typing import List, Any
from .base import BaseFNIRSParser
from typing import List


class Bak2019Parser(BaseFNIRSParser):
    dataset_id = "bak2019"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.raw_path = self.dataset_path
        self.orig2mne = {
            "Ch01": "S1_D1",
            "Ch02": "S1_D2",
            "Ch03": "S2_D1",
            "Ch04": "S2_D2",
            "Ch05": "S3_D2",
            "Ch06": "S2_D3",
            "Ch07": "S3_D3",
            "Ch08": "S3_D4",
            "Ch09": "S4_D3",
            "Ch10": "S4_D4",
            "Ch11": "S5_D5",
            "Ch12": "S5_D6",
            "Ch13": "S6_D5",
            "Ch14": "S6_D6",
            "Ch15": "S7_D6",
            "Ch16": "S6_D7",
            "Ch17": "S7_D7",
            "Ch18": "S7_D8",
            "Ch19": "S8_D7",
            "Ch20": "S8_D8",
        }
        self.cond2label = {"FOOT": 1, "LEFT": 2, "RIGHT": 3}

    def load_data(self, subject_id: str, session: int = None) -> dict:
        subject_file = self.raw_path / f"fNIRS {subject_id}.mat"
        if not subject_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_file}")
        data = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
        data["sfreq"] = data["nfo"].fs
        return data

    def extract_channels(self, data: dict):
        mnt = data["mnt"]
        ch_names = mnt.clab
        ch_data = np.array([data[f"ch{i+1}"] for i, chn in enumerate(ch_names)])
        ch_type_mask = np.array([not "deoxy" in ch for ch in ch_names])
        ch_types = ["hbo" if is_hbo else "hbr" for is_hbo in ch_type_mask]
        ch_names_mne = [
            f"{self.orig2mne[chn[:4]]} {ch_type}"
            for chn, ch_type in zip(ch_names, ch_types)
        ]
        return ch_data, ch_names_mne, ch_types

    def extract_events(self, data: dict):
        mrk = data["mrk"]
        mat_info = data["nfo"]
        sfreq = mat_info.fs
        idx = ((mrk.time) / (1000 / sfreq)).round().astype(int)
        events = np.column_stack([idx, np.zeros_like(idx), mrk.event.desc])
        event_desc = {i + 1: c for i, c in enumerate(mrk.className)}
        return events, event_desc

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        subject_files = list(self.raw_path.glob("fNIRS *.mat"))
        subject_ids = [f.stem.split()[-1] for f in subject_files]
        subject_ids.sort()
        return subject_ids
