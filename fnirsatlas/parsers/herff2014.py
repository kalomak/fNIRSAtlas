"""
label: 1 x nrsamples contains information about trials for each sample
    0 = pause
    1 = 1-back
    2 = 2-back
    3 = 3-back
    4 = relax
    5 = next task
    -1 = long break

events: 1 x nrsamples contains keypress events of the subject
    0 no keypress
    1 correctly identified target
    -1 wrong keypress

chars: 1 x nrsamples contains the displayed letter for every sample
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
import pickle
from .base import BaseFNIRSParser
from typing import List, Dict, Tuple


class Herff2014Parser(BaseFNIRSParser):
    dataset_id = "herff2014"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        # Event mapping: match main.py (zero-based for classification)
        self.cond2label = {
            "1-back": 1,
            "2-back": 2,
            "3-back": 3,
            # "relax": 4,
            # "next task": 5,
            # "long break": 6,
        }
        self.label2cond = {v: k for k, v in self.cond2label.items()}
        self.sfreq = 25.0
        self.orig2mne = {
            "CH1": "S1_D1",
            "CH2": "S2_D1",
            "CH3": "S1_D2",
            "CH4": "S2_D2",
            "CH5": "S3_D3",
            "CH6": "S4_D3",
            "CH7": "S3_D4",
            "CH8": "S4_D4",
        }
        self.ch_names = [f"{self.orig2mne[f'CH{i+1}']} hbo" for i in range(8)] + [
            f"{self.orig2mne[f'CH{i+1}']} hbr" for i in range(8)
        ]
        self.ch_types = [
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr",
            "hbo",
            "hbr",
        ]

    def load_data(self, subject_id: str, session: str = None) -> Dict:
        subject_file = self.dataset_path / f"subject{subject_id}.mat"
        if not subject_file.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_file}")
        mat = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
        labels = mat["label"]
        events = mat["events"]
        chars = mat["chars"]
        return {
            "delta_c": mat["data"] * 1e-6,  # convert uM to M
            "labels": labels,
            "events": events,
            "chars": chars,
            "sfreq": self.sfreq,
        }

    def extract_channels(self, data: dict) -> Tuple[np.ndarray, List[str], List[str]]:
        ch_data = data["delta_c"]
        ch_names = self.ch_names.copy()
        ch_types = self.ch_types.copy()
        return ch_data, ch_names, ch_types

    def extract_events(self, data: dict) -> Tuple[np.ndarray, Dict]:
        labels = data["labels"].flatten()
        # Find transitions (onset of each event)
        event_onsets = np.where(np.diff(labels, prepend=labels[0]) != 0)[0]
        events = []
        event_desc = {}
        orig_label2cond = {
            -1: "long break",
            0: "pause",
            1: "1-back",
            2: "2-back",
            3: "3-back",
            4: "relax",
            5: "next task",
        }
        for idx in event_onsets:
            orig_event_id = int(labels[idx])
            cond = orig_label2cond[orig_event_id]
            # filter out non-back events already since they have varying length
            if not "-back" in cond:
                continue
            event_id = self.cond2label[cond]
            events.append([idx, 0, event_id])
            event_desc[event_id] = self.label2cond[event_id]
        return np.array(events), event_desc

    def get_subject_ids(self) -> List[str]:
        subject_files = list(self.dataset_path.glob("subject*.mat"))
        subject_ids = [f.stem.replace("subject", "") for f in subject_files]
        subject_ids.sort(key=lambda x: int(x))
        return subject_ids
