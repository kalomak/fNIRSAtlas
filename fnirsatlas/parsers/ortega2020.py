import numpy as np
from pathlib import Path
import h5py
from .base import BaseFNIRSParser
from typing import List, Dict, Tuple, Any


class Ortega2020Parser(BaseFNIRSParser):
    dataset_id = "ortega2020"

    def __init__(self, dataset_path: Path = None):
        super().__init__(dataset_path)
        self.h5_file = self.dataset_path / "hygrip.h5"
        self.cond2label = {"left": 1, "rest": 2, "right": 3}
        self.label2cond = {v: k for k, v in self.cond2label.items()}
        self.ch_pair2ch = {
            0: "S1_D1",
            1: "S2_D1",
            2: "S1_D2",
            3: "S3_D1",
            4: "S2_D3",
            5: "S3_D2",
            6: "S3_D3",
            7: "S4_D2",
            8: "S3_D4",
            9: "S5_D3",
            10: "S4_D4",
            11: "S5_D4",
            12: "S6_D6",
            13: "S7_D6",
            14: "S6_D7",
            15: "S8_D6",
            16: "S7_D8",
            17: "S8_D7",
            18: "S8_D8",
            19: "S9_D7",
            20: "S8_D9",
            21: "S10_D8",
            22: "S9_D9",
            23: "S10_D9",
        }

    def load_data(self, subject_id: str, session: Any = None) -> Dict:
        with h5py.File(self.h5_file, "r") as ds:
            x = np.concatenate(
                [ds[f"{subject_id}/oi1"][:], ds[f"{subject_id}/oi2"][:]], axis=0
            )
            events = ds[f"{subject_id}/oi1"].attrs["events"]
            sfreq = ds.attrs["oxy_sfreq"]
        return {"data": x, "events": events, "sfreq": sfreq, "subject_id": subject_id}

    def extract_channels(self, data: dict) -> Tuple[np.ndarray, List[str], List[str]]:
        ch_data = data["data"]
        subject_id = data["subject_id"]
        ch_names = []
        ch_types = []
        with h5py.File(self.h5_file, "r") as ds:
            for k in range(ds[subject_id]["oi1"].shape[0]):
                ch_pair_name = self.ch_pair2ch[k]
                for j in range(2):
                    wl = ["760", "850"][j]
                    ch_names.append(f"{ch_pair_name} {wl}")
                    ch_types.append("fnirs_cw_amplitude")

        return ch_data, ch_names, ch_types

    def extract_events(self, data: dict) -> Tuple[np.ndarray, Dict]:
        events = data["events"]
        sfreq = data["sfreq"]
        times = np.arange(0, data["data"].shape[-1]) / sfreq
        event_times = events[:, 0]
        sample_times = np.searchsorted(times, event_times)
        event_mapping = {-1: "rest", 0: "left", 1: "right"}  # original event mapping
        event_conds = [event_mapping[e] for e in events[:, 1]]
        event_ids = [self.cond2label[c] for c in event_conds]
        events_arr = np.c_[sample_times, np.zeros(events.shape[0]), event_ids].astype(
            int
        )
        label2cond = {v: k for k, v in self.cond2label.items()}
        return events_arr, label2cond

    def get_subject_ids(self) -> List[str]:
        with h5py.File(self.h5_file, "r") as ds:
            subject_ids = [
                k
                for k in ds.keys()
                if hasattr(ds[k], "keys") and "oi1" in ds[k] and "oi2" in ds[k]
            ]
        subject_ids.sort()
        return subject_ids
