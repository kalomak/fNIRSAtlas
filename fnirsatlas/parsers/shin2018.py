"""
Montage extraction is complicated so we just load a precomputed montage.
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from .base import BaseFNIRSParser
from typing import List, Dict, Tuple
import mne
import collections

orig2mne = {
    "AFp7": "S1_D1",
    "AFp3": "S2_D1",
    "AFpz": "S2_D2",
    "AFp4": "S3_D2",
    "AFp8": "S3_D3",
    "AF7": "S1_D4",
    "AF5h": "S4_D1",
    "AF1": "S2_D5",
    "AF2": "S5_D2",
    "AF6h": "S3_D6",
    "AF8": "S6_D3",
    "AFF5": "S4_D4",
    "AFF3h": "S4_D5",
    "AFFz": "S5_D5",
    "AFF4h": "S5_D6",
    "AFF6": "S6_D6",
    "FCC3": "S7_D7",
    "C5h": "S8_D7",
    "C3h": "S7_D8",
    "CCP3": "S8_D8",
    "FCC4": "S9_D9",
    "C4h": "S10_D9",
    "C6h": "S9_D10",
    "CCP4": "S10_D10",
    "CPP3": "S11_D11",
    "P5h": "S12_D11",
    "P3h": "S11_D12",
    "PPO3": "S12_D12",
    "CPP4": "S13_D13",
    "P4h": "S14_D13",
    "P6h": "S13_D14",
    "PPO4": "S14_D14",
    "PPOz": "S15_D15",
    "PO1": "S16_D15",
    "PO2": "S15_D16",
    "POOz": "S16_D16",
}


class Shin2018Parser(BaseFNIRSParser):
    dataset_id = "shin2018"
    EXPERIMENTS = ["wg", "dsr", "nback"]

    def __init__(self, dataset_path: Path = None, experiment: str = "wg"):
        super().__init__(dataset_path)
        self.experiment = experiment.lower()
        self.cond2label = self.get_cond2label()

    def get_cond2label(self) -> Dict:
        if self.experiment == "wg":
            return {"BL": 1, "WG": 2}
        elif self.experiment == "dsr":
            return {"DSR": 1}
        elif self.experiment == "nback":
            return {"0-back": 1, "2-back": 2, "3-back": 3}
        else:
            raise ValueError(f"Experiment {self.experiment} not supported")

    def load_data(self, subject_id: str, session: int = None) -> Dict:
        sub_dir = self.dataset_path / f"{subject_id}"
        cnt = loadmat(
            sub_dir / f"cnt_{self.experiment}.mat",
            squeeze_me=True,
            struct_as_record=False,
        )[f"cnt_{self.experiment}"]
        mrk = loadmat(
            sub_dir / f"mrk_{self.experiment}.mat",
            squeeze_me=True,
            struct_as_record=False,
        )[f"mrk_{self.experiment}"]
        mnt = loadmat(
            sub_dir / f"mnt_{self.experiment}.mat",
            squeeze_me=True,
            struct_as_record=False,
        )[f"mnt_{self.experiment}"]
        sfreq = float(cnt.oxy.fs)
        return {"cnt": cnt, "mrk": mrk, "mnt": mnt, "sfreq": sfreq}

    def extract_channels(self, data: dict) -> Tuple[np.ndarray, List[str], List[str]]:
        cnt = data["cnt"]
        oxy = cnt.oxy.x
        deoxy = cnt.deoxy.x
        ch_names_hbo = [f"{orig2mne[chn]} hbo" for chn in cnt.oxy.clab]
        ch_names_hbr = [f"{orig2mne[chn]} hbr" for chn in cnt.deoxy.clab]
        ch_types = ["hbo"] * len(ch_names_hbo) + ["hbr"] * len(ch_names_hbr)
        ch_data = np.concatenate([oxy.T, deoxy.T], axis=0)
        return ch_data, ch_names_hbo + ch_names_hbr, ch_types

    def extract_events(self, data: dict) -> Tuple[np.ndarray, Dict]:
        mrk = data["mrk"]
        sfreq = self.get_sampling_frequency(data)
        idx = ((mrk.time) * sfreq / 1000).round().astype(int)
        events = np.column_stack([idx, np.zeros_like(idx), mrk.event.desc])

        # Map event IDs to class names properly
        unique_event_ids = sorted(set(mrk.event.desc))
        event_desc = {}
        for i, event_id in enumerate(unique_event_ids):
            event_desc[event_id] = mrk.className[i][:6]

        return events, event_desc

    def get_subject_ids(self) -> List[str]:
        subject_dirs = [
            d
            for d in self.dataset_path.iterdir()
            if d.is_dir() and d.name.startswith("VP")
        ]
        subject_ids = [d.name for d in subject_dirs]
        subject_ids.sort()
        return subject_ids


### MONTAGE CODE BELOW ###

# def get_channels_for_sensor(ch_dict, sensor_name):
#     connected_channels = []
#     for ch_name in ch_dict.keys():
#         source, detector = ch_name.split("_")
#         if source == sensor_name or detector == sensor_name:
#             connected_channels.append(ch_name)
#     return connected_channels


# def get_sensor_positions(ch_dict, set_locations):
#     sensor_positions = {**set_locations}
#     processed_sensors = set(sensor_positions.keys())
#     to_process = list(sensor_positions.keys())
#     iters = 0
#     while to_process and iters < 1000:
#         current_sensor = to_process.pop(0)
#         channels = get_channels_for_sensor(ch_dict, current_sensor)
#         ch_directions = np.array(
#             [ch_dict[ch] - sensor_positions[current_sensor] for ch in channels]
#         )
#         ch_distances = np.linalg.norm(ch_directions, axis=1)
#         ch_directions = ch_directions / ch_distances[:, None]
#         channel_sensors = [ch.split("_") for ch in channels]

#         # place sensors to opposite sides of channels
#         for ch, ch_dir, ch_dist, ch_sensor in zip(
#             channels, ch_directions, ch_distances, channel_sensors
#         ):
#             neighbor_sensor = (
#                 ch_sensor[0] if ch_sensor[1] == current_sensor else ch_sensor[1]
#             )
#             if neighbor_sensor in processed_sensors:
#                 continue
#             sensor_positions[neighbor_sensor] = (
#                 sensor_positions[current_sensor] + ch_dir * ch_dist * 2
#             )
#             to_process.append(neighbor_sensor)
#         processed_sensors.add(current_sensor)
#         iters += 1

#     return sensor_positions


# def _get_mne_ch_info_and_positions(mnt):
#     ch_sort_ix = np.array(
#         [
#             2,
#             4,
#             8,
#             10,
#             22,
#             0,
#             3,
#             6,
#             9,
#             20,
#             23,
#             1,
#             5,
#             7,
#             19,
#             21,
#             11,
#             13,
#             12,
#             14,
#             24,
#             26,
#             25,
#             27,
#             15,
#             17,
#             16,
#             18,
#             28,
#             30,
#             29,
#             31,
#             32,
#             33,
#             34,
#             35,
#         ]
#     )
#     ch_names_sorted = mnt.clab[ch_sort_ix]
#     sources = {
#         1: {"channels": [0, 5]},
#         2: {"channels": [1, 2, 7]},
#         3: {"channels": [3, 4, 9]},
#         4: {"channels": [6, 11, 12]},
#         5: {"channels": [8, 13, 14]},
#         6: {"channels": [10, 15]},
#         7: {"channels": [16, 18]},
#         8: {"channels": [17, 19]},
#         9: {"channels": [20, 22]},
#         10: {"channels": [21, 23]},
#         11: {"channels": [24, 26]},
#         12: {"channels": [25, 27]},
#         13: {"channels": [28, 30]},
#         14: {"channels": [29, 31]},
#         15: {"channels": [32, 34]},
#         16: {"channels": [33, 35]},
#     }
#     detectors = {
#         1: {"channels": [0, 1, 6]},
#         2: {"channels": [2, 3, 8]},
#         3: {"channels": [4, 10]},
#         4: {"channels": [5, 11]},
#         5: {"channels": [7, 12, 13]},
#         6: {"channels": [9, 14, 15]},
#         7: {"channels": [16, 17]},
#         8: {"channels": [18, 19]},
#         9: {"channels": [20, 21]},
#         10: {"channels": [22, 23]},
#         11: {"channels": [24, 25]},
#         12: {"channels": [26, 27]},
#         13: {"channels": [28, 29]},
#         14: {"channels": [30, 31]},
#         15: {"channels": [32, 33]},
#         16: {"channels": [34, 35]},
#     }
#     mne_ch_names = {}
#     for i in range(36):
#         source = [k for k, v in sources.items() if i in v["channels"]][0]
#         detector = [k for k, v in detectors.items() if i in v["channels"]][0]
#         mne_ch_names[i] = f"S{source}_D{detector}"
#     orig2mne = {
#         ch_names_sorted[i]: mne_ch_names[i] for i in range(len(ch_names_sorted))
#     }

#     # Channel positions
#     scale = 0.08
#     ch_dict = {
#         mnt.clab[i]: np.array([mnt.x[i], mnt.y[i], 0]) * scale
#         for i in range(len(mnt.clab))
#     }
#     mne_ch_dict = {orig2mne[orig]: ch_dict[orig] for orig in ch_dict}
#     ch_dict = mne_ch_dict

#     # Set initial sensor locations (from working implementation)
#     set_locations = {
#         "D1": np.mean([ch_dict["S1_D1"], ch_dict["S2_D1"]], axis=0),
#         "S2": np.mean([ch_dict["S2_D1"], ch_dict["S2_D2"]], axis=0),
#         "D2": np.mean([ch_dict["S2_D2"], ch_dict["S3_D2"]], axis=0),
#         "S3": np.mean([ch_dict["S3_D2"], ch_dict["S3_D3"]], axis=0),
#         "S4": np.mean([ch_dict["S4_D4"], ch_dict["S4_D5"]], axis=0),
#         "D5": np.mean([ch_dict["S4_D5"], ch_dict["S5_D5"]], axis=0),
#         "S5": np.mean([ch_dict["S5_D5"], ch_dict["S5_D6"]], axis=0),
#         "D6": np.mean([ch_dict["S5_D6"], ch_dict["S6_D6"]], axis=0),
#         "D7": np.array([ch_dict["S8_D7"][0], ch_dict["S7_D7"][1], 0]),
#         "D9": np.array([ch_dict["S10_D9"][0], ch_dict["S9_D9"][1], 0]),
#         "D11": np.array([ch_dict["S12_D11"][0] - 0.002, ch_dict["S11_D11"][1], 0]),
#         "D13": np.array([ch_dict["S14_D13"][0] + 0.002, ch_dict["S13_D13"][1], 0]),
#         "D15": np.array([ch_dict["S16_D15"][0], ch_dict["S15_D15"][1], 0]),
#     }

#     # Use the working sensor position calculation
#     sensor_positions = get_sensor_positions(ch_dict, set_locations)

#     return orig2mne, sensor_positions
