"""Bak et al. 2019 fNIRS finger/foot-tapping dataset."""

from pathlib import Path

import numpy as np
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)
from scipy.io import loadmat

from fnirsatlas.datasets.base import BaseFNIRSDataset
from fnirsatlas.datasets.metadata import FnirsAcquisitionMetadata
from fnirsatlas.datasets.utils import concentration_length_to_molar, extract_archive

# Recorded with a Shimadzu LIGHTNIRS, which uses 780, 805 and 830 nm. 780 and
# 830 nm best separate HbO and HbR.
WAVELENGTHS = (780, 830)
MONTAGE_PATH = Path(__file__).parent / "montages" / "bak2019_montage_3d.json"
URL = "https://ndownloader.figshare.com/files/18069143"
CODE = "Bak2019FnirsTapping"
SOURCE_DETECTOR_DISTANCE_CM = 3.0
SFREQ = 40 / 3  # 13.333... Hz
CH_NAMES_ORDERED = [
    "S1_D1",
    "S1_D2",
    "S2_D1",
    "S2_D2",
    "S3_D2",
    "S2_D3",
    "S3_D3",
    "S3_D4",
    "S4_D3",
    "S4_D4",
    "S5_D5",
    "S5_D6",
    "S6_D5",
    "S6_D6",
    "S7_D6",
    "S6_D7",
    "S7_D7",
    "S7_D8",
    "S8_D7",
    "S8_D8",
]
CLASS_NAME_MAP = {"RIGHT": 1, "LEFT": 2, "FOOT": 3}
EVENTS = {"Tapping/Right": 1, "Tapping/Left": 2, "Tapping/Foot": 3}
N_CHANNELS = 20
# -2 would drop first epoch for each subject. Onsets are at least 28.95 s
# apart.
INTERVAL = [-1.98, 15]


class Bak2019FnirsTapping(BaseFNIRSDataset):
    """Bak et al. 2019 fNIRS finger/foot-tapping dataset.

    Right-hand finger, left-hand finger and foot tapping for 10 s, with the task
    shown on a monitor and started by a beep [1]_.

    References
    ----------
    .. [1] Bak, S., Park, J., Shin, J., & Jeong, J. (2019).
           Open-access fNIRS dataset for classification of unilateral
           finger- and foot-tapping. Electronics, 8(12), 1486.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"hbo": N_CHANNELS, "hbr": N_CHANNELS},
            hardware="Shimadzu LIGHTNIRS",
            # Recorded at Korea University, Seoul.
            line_freq=60.0,
            montage="custom",
            n_sources=8,
            n_detectors=8,
            wavelengths_nm=[780.0, 805.0, 830.0],
            n_short_channels=0,
            # montage-toolkit positions, constructed from the paper's optode image
            # and C3/C4/Cz reference positions.
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            gender={"male": 17, "female": 13},
            # The paper reports only the right-handed count.
            handedness={"right": 29},
            age_mean=23.4,
            age_std=2.5,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="finger and foot tapping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=10.0,
            stimulus_type="visual task cue and beep",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3390/electronics8121486",
            investigators=["SuJin Bak", "Jinwoo Park", "Jaeyoung Shin", "Jichai Jeong"],
            institution="Korea University",
            country="KR",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.9783755",
            license="CC BY 4.0",
            publication_year=2019,
        ),
        data_structure=DataStructureMetadata(
            n_trials=75,
            n_trials_per_class={label: 25 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="MATLAB",
        # Ternary SVM, leave-one-out cross-validation.
        performance={"accuracy_percent": 70.4, "accuracy_std": 18.4},
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 31)),
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
            doi="10.3390/electronics8121486",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        del update_path
        data_dir = self._download_data(subject, path, force_update, verbose)
        return str(data_dir / f"fNIRS {subject:02d}.mat")

    def _get_single_subject_data(self, subject):
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True, struct_as_record=False)
        raw = self._convert_subject(mat)
        return self._single_run_session(raw)

    def _convert_subject(self, mat):
        # Shimadzu exports mM cm concentration-length values.
        ch_data = concentration_length_to_molar(
            [mat[f"ch{i + 1}"] for i in range(N_CHANNELS * 2)],
            SOURCE_DETECTOR_DISTANCE_CM,
        )
        mrk = mat["mrk"]
        sfreq = mat["nfo"].fs
        samples = (mrk.time / (1000 / sfreq)).round().astype(int)
        labels = np.array(
            [CLASS_NAME_MAP[mrk.className[d - 1]] for d in mrk.event.desc]
        )
        stim_raw = self._make_stim_raw(ch_data.shape[1], sfreq, samples, labels)
        data = np.vstack([ch_data, stim_raw.get_data()])
        ch_names = (
            [f"{ch} hbo" for ch in CH_NAMES_ORDERED]
            + [f"{ch} hbr" for ch in CH_NAMES_ORDERED]
            + ["Stim"]
        )
        ch_types = ["hbo"] * N_CHANNELS + ["hbr"] * N_CHANNELS + ["stim"]
        raw = self._make_raw(
            data, ch_names, ch_types, sfreq, montage=self._load_montage()
        )
        raw = self._populate_fnirs_metadata(raw, default_wavelengths=WAVELENGTHS)
        return self._standardize_raw(raw)

    def _download_data(self, subject, path, force_update, verbose):
        data_dir = Path(dl.get_dataset_path(CODE, path))
        base_dir = data_dir / f"MNE-{CODE.lower()}-data"
        if (base_dir / f"fNIRS {subject:02d}.mat").is_file() and not force_update:
            return base_dir
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        base_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, base_dir)
        return base_dir

    @staticmethod
    def _load_montage():
        return BaseFNIRSDataset._load_montage_from_json(MONTAGE_PATH)[0]
