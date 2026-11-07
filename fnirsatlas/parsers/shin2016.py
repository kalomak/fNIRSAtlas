import numpy as np
from pathlib import Path
from scipy.io import loadmat
from .base import BaseFNIRSParser
from typing import List, Dict, Any
import mne

from fnirsatlas.channels import to_mne_ch_order


class Shin2016Parser(BaseFNIRSParser):
    dataset_id = "shin2016"
    EXPERIMENTS = ["MA", "MI"]

    def __init__(self, dataset_path: Path = None, experiment: str = "MA"):
        super().__init__(dataset_path)
        self.experiment = experiment.lower()
        self.cond2label = (
            {"BL": 1, "MA": 2}
            if self.experiment.lower() == "ma"
            else {"LMI": 3, "RMI": 4}
        )

    def load_data(self, subject_id: str, session: int = None) -> dict:
        sub_dir = self.dataset_path / f"subject {subject_id}"
        cnt = loadmat(sub_dir / "cnt.mat", squeeze_me=True, struct_as_record=False)[
            "cnt"
        ][session]
        mrk = loadmat(sub_dir / "mrk.mat", squeeze_me=True, struct_as_record=False)[
            "mrk"
        ][session]
        mnt = loadmat(sub_dir / "mnt.mat", squeeze_me=True, struct_as_record=False)[
            "mnt"
        ]
        return {"cnt": cnt, "mrk": mrk, "mnt": mnt}

    def get_montage(
        self, ch_names: List[str], data: dict = None
    ) -> mne.channels.DigMontage:
        """Create montage using actual 3D coordinates from Shin2016 data."""
        mnt = data["mnt"]
        source_name_map = {orig: f"S{i+1}" for i, orig in enumerate(mnt.source.clab)}
        detector_name_map = {
            orig: f"D{i+1}" for i, orig in enumerate(mnt.detector.clab)
        }

        coord_frame = "head"
        nasion = np.array([0.00146763, 0.08506715, -0.03483611])
        lpa = np.array([-0.08061612, -0.02908875, -0.04131077])
        rpa = np.array([0.08436285, -0.02850276, -0.04127743])
        scale = 0.08

        source_pos_map = {
            source_name_map[orig]: mnt.source.pos_3d[:, i] * scale
            for i, orig in enumerate(mnt.source.clab)
        }
        detector_pos_map = {
            detector_name_map[orig]: mnt.detector.pos_3d[:, i] * scale
            for i, orig in enumerate(mnt.detector.clab)
        }

        sensor_dict = {}
        for ch_name in ch_names:
            if ch_name == "Stim":
                continue
            s, d = ch_name.split(" ")[0].split("_")
            sensor_dict[s] = source_pos_map[s]
            sensor_dict[d] = detector_pos_map[d]

        montage = mne.channels.make_dig_montage(
            ch_pos=sensor_dict,
            coord_frame=coord_frame,
            nasion=nasion,
            lpa=lpa,
            rpa=rpa,
        )
        return montage

    def get_sessions(self, subject_id: str) -> List[str]:
        # sessions 0, 2, 4 are motor imagery, 1, 3, 5 mental arithmetic
        if self.experiment.lower() == "ma":
            return range(1, 6, 2)
        elif self.experiment.lower() == "mi":
            return range(0, 6, 2)

    def extract_channels(self, data: dict):
        session_data = data["cnt"]
        fnirs_data = session_data.x.T * 1e-6
        mnt = data["mnt"]
        source_name_map = {orig: f"S{i+1}" for i, orig in enumerate(mnt.source.clab)}
        detector_name_map = {
            orig: f"D{i+1}" for i, orig in enumerate(mnt.detector.clab)
        }
        sd_pairs = [
            (mnt.source.clab[si - 1], mnt.detector.clab[di - 1]) for si, di in mnt.sd
        ]
        wl_lo, wl_hi = session_data.wavelengths
        ch_constr = [(s, d, "lowWL") for s, d in sd_pairs] + [
            (s, d, "highWL") for s, d in sd_pairs
        ]
        ch_names_mne = [
            f'{source_name_map[s]}_{detector_name_map[d]} {wl_lo if wl=="lowWL" else wl_hi}'
            for s, d, wl in ch_constr
        ]
        ch_types = ["fnirs_cw_amplitude"] * len(ch_names_mne)
        return fnirs_data, ch_names_mne, ch_types

    def extract_events(self, data: dict):
        mrk = data["mrk"]
        sfreq = data["cnt"].fs
        idx = ((mrk.time) / (1000 / sfreq)).round().astype(int)

        # event_desc = {i + 1: c for i, c in enumerate(mrk.className)}
        conds = [mrk.className[e - 1] for e in mrk.event.desc]
        conds = [self.cond2label[c] for c in conds]
        events = np.column_stack([idx, np.zeros_like(idx), conds])
        event_desc = {v: k for k, v in self.cond2label.items() if v in conds}

        return events, event_desc

    def get_sampling_frequency(self, data: dict, session_idx=0) -> float:
        return data["cnt"].fs

    def get_subject_ids(self) -> List[str]:
        """Get list of available subject IDs."""
        subject_dirs = [
            d
            for d in self.dataset_path.iterdir()
            if d.is_dir() and d.name.startswith("subject")
        ]
        subject_ids = [d.name.split()[-1] for d in subject_dirs]
        subject_ids.sort()
        return subject_ids

    def _raw_from_session(self, data: Dict) -> mne.io.RawArray:
        raw = super()._raw_from_session(data)
        montage = self.get_montage(raw.ch_names, data)
        raw.set_montage(montage)
        raw.reorder_channels(to_mne_ch_order(raw.info["chs"]))
        return raw
