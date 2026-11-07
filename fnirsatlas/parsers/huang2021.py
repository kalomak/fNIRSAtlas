"""
Most of raw data is missing so we'll use the preprocessed data for now.
See scripts/checks/inspect_huang2021.py for reading the raw data.
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from typing import List, Dict, Tuple, Any

from fnirsatlas.epoching import create_epochs_from_raw
from .base import BaseFNIRSParser
import logging

logger = logging.getLogger(__name__)


class Huang2021Parser(BaseFNIRSParser):
    dataset_id = "huang2021"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        if self.dataset_path.stem == "raw_data":
            raise ValueError("Raw data not supported for Huang2021")
        self.sfreq = 5.2
        self.ch_map = {
            "AB_I_O": "S1_D1 hbo",
            "AB_I_DO": "S1_D1 hbr",
            "CD_I_O": "S2_D2 hbo",
            "CD_I_DO": "S2_D2 hbr",
        }
        self.label2cond = {
            1: "0-back",
            2: "1-back",
            3: "2-back",
            4: "3-back",
            5: "rest",
        }
        self.cond2label = {v: k for k, v in self.label2cond.items()}
        self.default_wls = [690, 830]
        self.montage = self.get_montage()

    def get_subject_ids(self) -> List[str]:
        return sorted(
            [
                int(d.name[4:-4])
                for d in self.dataset_path.iterdir()
                if d.name.startswith("sub_")
            ]
        )

    def load_data(self, subject_id: str, session: Any = None) -> Dict:
        df = pd.read_csv(self.dataset_path / f"sub_{subject_id}.csv")
        return {"df": df, "subject_id": subject_id, "sfreq": self.sfreq}

    def extract_channels(self, data: Dict) -> Tuple[np.ndarray, List[str], List[str]]:
        df = data["df"]
        ch_names = [
            c for c in df.columns if c.endswith("_I_O") or c.endswith("_I_DO")
        ]  # only use intensity data for now to avoid mne issues
        ch_data = df[ch_names].values.T
        ch_names = [self.ch_map[c] for c in ch_names]
        ch_types = [c.split(" ")[-1] for c in ch_names]
        return ch_data, ch_names, ch_types

    def extract_events(self, data: Dict) -> Tuple[np.ndarray, Dict]:
        label_map = {
            0: 1,  # 0-back
            1: 2,  # 1-back
            2: 3,  # 2-back
            3: 4,  # 3-back
            -1: 5,  # rest
        }
        labels = data["df"]["label"].map(label_map).values
        onset_ix = np.where(np.diff(labels) != 0)[0] + 1
        onset_ix = np.insert(onset_ix, 0, 0)
        events = np.c_[onset_ix, np.zeros(len(onset_ix)), labels[onset_ix]].astype(int)
        event_desc = self.label2cond
        return np.array(events), event_desc

    def _create_epochs(self, raw):
        return create_epochs_from_raw(
            raw=raw,
            dataset_name=self.dataset_id,
            # The raw data is already epoched
            tmin=0,
            baseline=(0, 0),
        )

    def get_montage(self):
        x_start = -0.075
        y = 0.075

        # # Original sensors:
        # dists = [0, 0.025, 0.01, 0.025, 0.04, 0.025, 0.01, 0.025]
        # sensors = ["S1", "D1", "D2", "S2", "S3", "D3", "D4", "S4"]

        # # Average chs (after dual slope)
        dists = [0.015, 0.03, 0.06, 0.03]
        sensors = ["S1", "D1", "S2", "D2"]
        x = x_start + np.cumsum(dists)
        sensor_positions = {s: np.array([x[i], y, 0]) for i, s in enumerate(sensors)}

        coord_frame = "head"
        nasion = np.array([0.00146763, 0.08506715, -0.03483611])
        lpa = np.array([-0.08061612, -0.02908875, -0.04131077])
        rpa = np.array([0.08436285, -0.02850276, -0.04127743])
        return mne.channels.make_dig_montage(
            ch_pos=sensor_positions,
            coord_frame=coord_frame,
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
        )
