from ast import Dict
from pathlib import Path
from typing import List
import warnings
import logging
import mne
from fnirsatlas.parsers.base import BaseFNIRSParser
from fnirsatlas.channels import validate_ch_names, to_mne_ch_order
from fnirsatlas.utils import override_raw_sfreq

logger = logging.getLogger(__name__)


class Guevara2024Parser(BaseFNIRSParser):
    dataset_id = "guevara2024"
    EXPERIMENTS = ["finger_tapping", "resting_state", "walking"]

    def __init__(self, dataset_path: Path = None, experiment: str = "finger_tapping"):
        super().__init__(dataset_path)
        self.experiment = experiment.lower()
        self.groups = ["Control"]  # , "PD"] # We use control subjects only

    def get_subject_ids(self) -> List[str]:
        sub_nums = range(1, 21)
        return [
            f"{group}{sub_num:02d}" for group in self.groups for sub_num in sub_nums
        ]

    def load_data(self, subject_id: str, session: str = None) -> Dict:
        subject_group = subject_id[:-2]
        sub_dir = self.dataset_path / self.experiment / subject_group / subject_id
        snirf_path = list(sub_dir.glob("*.snirf"))
        if len(snirf_path) != 1:
            raise ValueError(
                f"Expected 1 snirf file for subject {subject_id}, found {len(snirf_path)}"
            )
        return {
            "snirf_path": snirf_path[0],
            "subject_id": subject_id,
            "experiment": self.experiment,
        }

    def _raw_from_session(self, data: Dict) -> mne.io.RawArray:
        snirf_path = data["snirf_path"]

        with warnings.catch_warnings():
            # We set the montage manually below
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=".*If it is not possible to include 3D positions in your data, please consider using the*",
            )
            raw = mne.io.read_raw_snirf(snirf_path, verbose=False, preload=True)

        # Replace 2d montage with a 3d montage
        raw = self.set_montage(raw)

        # Correct channel ordering to mne norm
        if not validate_ch_names(raw.ch_names):
            raw.reorder_channels(to_mne_ch_order(raw.info["chs"]))

        # We suspect that subject "Control18" has the wrong sfreq.
        # See `scripts/checks/inspect_guevara2024_c18.py` for details.
        if self.experiment == "finger_tapping" and data["subject_id"] == "Control18":
            # 12.5Hz produces 76 post-crop samples here, 12.49999 produces 75
            # like the rest of the dataset 
            raw = override_raw_sfreq(raw, factor=12.49999 / raw.info["sfreq"])
            logger.info(
                f"Adjusted sfreq for {data['subject_id']} to {raw.info['sfreq']} Hz"
            )
        return raw
