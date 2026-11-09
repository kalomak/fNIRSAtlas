"""Shin 2016 EEG+NIRS mental arithmetic and motor imagery fNIRS datasets."""

import shutil
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
from fnirsatlas.datasets.utils import download_lock, extract_archive

URL = "https://doc.ml.tu-berlin.de/hBCI/NIRS/NIRS_01-29.zip"
DOWNLOAD_CODE = "BBCIFNIRS"
SUBJECTS = list(range(1, 30))
# Onsets are at least 27.44 s apart.
INTERVAL = [-2, 15]
SFREQ = 10.0
N_CHANNEL_PAIRS = 36
MA_SESSION_INDICES = (1, 3, 5)
MA_EVENTS = {"mental_arithmetic": 1, "baseline": 2}
MA_CLASS_NAME_MAP = {
    "MA": MA_EVENTS["mental_arithmetic"],
    "BL": MA_EVENTS["baseline"],
}
MI_SESSION_INDICES = (0, 2, 4)
MI_EVENTS = {"left_motor_imagery": 3, "right_motor_imagery": 4}
MI_CLASS_NAME_MAP = {
    "LMI": MI_EVENTS["left_motor_imagery"],
    "RMI": MI_EVENTS["right_motor_imagery"],
}
HEAD_SCALE = 0.08
FIDUCIALS = {
    "nasion": np.array([0.00146763, 0.08506715, -0.03483611]),
    "lpa": np.array([-0.08061612, -0.02908875, -0.04131077]),
    "rpa": np.array([0.08436285, -0.02850276, -0.04127743]),
}
DOI = "10.1109/TNSRE.2016.2628057"
# Each session has 10 trials per class.
N_TRIALS_PER_CLASS = 10 * len(MA_SESSION_INDICES)
ACQUISITION = FnirsAcquisitionMetadata(
    # Recorded at 12.5 Hz, released at 10 Hz.
    sampling_rate=SFREQ,
    channel_types={"fnirs_cw_amplitude": 2 * N_CHANNEL_PAIRS},
    hardware="NIRx NIRScout",
    # Recorded at TU Berlin, DE.
    line_freq=50.0,
    montage="custom",
    # cnt.mat wavelengths field.
    wavelengths_nm=[760.0, 850.0],
    n_sources=14,
    n_detectors=16,
    n_short_channels=0,
    # pos_3d of the released mnt.mat files, scaled by HEAD_SCALE.
    sensor_positions="3d",
    other_modalities=["EEG", "EOG", "ECG", "respiration"],
)
# The paper reports 29 right-handed and 1 left-handed subjects but 29 subjects in
# total, so handedness is omitted.
PARTICIPANTS = ParticipantMetadata(
    n_subjects=len(SUBJECTS),
    gender={"male": 14, "female": 15},
    age_mean=28.5,
    age_std=3.7,
)
DOCUMENTATION = DocumentationMetadata(
    doi=DOI,
    investigators=[
        "Jaeyoung Shin",
        "Alexander von Lühmann",
        "Benjamin Blankertz",
        "Do-Won Kim",
        "Jichai Jeong",
        "Han-Jeong Hwang",
        "Klaus-Robert Müller",
    ],
    institution="Technische Universität Berlin",
    country="DE",
    repository="TU Berlin Machine Learning Group",
    data_url="https://doc.ml.tu-berlin.de/hBCI/",
    # The download page links gnu.org/licenses/gpl.txt without naming a version.
    license="GPL-3.0",
    publication_year=2017,
)
LICENSE_ERROR = (
    "You must accept licence term to download this dataset,"
    "set accept=True when instantiating the dataset."
)


class BaseShin2016Fnirs(BaseFNIRSDataset):
    """Base class for Shin 2016 fNIRS datasets (MA and MI tasks)."""

    # Set by subclasses.
    _session_indices: tuple[int, ...]
    _class_name_map: dict[str, int]

    def __init__(self, accept=False, **kwargs):
        self.accept = accept
        super().__init__(**kwargs)

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        accept=False,
    ):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        if accept:
            self.accept = True
        subject_dir = self._subject_dir(path, subject)
        # MA and MI share one archive.
        with download_lock(subject_dir.parent):
            if self._has_subject_files(subject_dir) and not force_update:
                return subject_dir
            self._download_data(path, force_update, verbose)
        if not self._has_subject_files(subject_dir):
            raise FileNotFoundError(
                f"Missing Shin2016 data for subject {subject}: {subject_dir}"
            )
        return subject_dir

    def _get_single_subject_data(self, subject):
        cnt, mrk, mnt = self._load_subject_mats(self.data_path(subject))
        montage = self._make_montage(mnt)
        sessions = {}
        for session_num, session_idx in enumerate(self._session_indices, start=1):
            raw = self._convert_session(
                cnt[session_idx], mrk[session_idx], mnt, montage
            )
            self._add_run(sessions, f"{session_num}fnirs", raw)
        return sessions

    @staticmethod
    def _subject_dir(path, subject):
        return (
            Path(dl.get_dataset_path(DOWNLOAD_CODE, path))
            / "MNE-eegfnirs-data"
            / "NIRS"
            / f"subject {subject:02d}"
        )

    @staticmethod
    def _has_subject_files(subject_dir):
        return all(
            (subject_dir / name).is_file() for name in ("cnt.mat", "mrk.mat", "mnt.mat")
        )

    @staticmethod
    def _load_subject_mats(subject_dir):
        cnt = loadmat(subject_dir / "cnt.mat", squeeze_me=True, struct_as_record=False)[
            "cnt"
        ]
        mrk = loadmat(subject_dir / "mrk.mat", squeeze_me=True, struct_as_record=False)[
            "mrk"
        ]
        mnt = loadmat(subject_dir / "mnt.mat", squeeze_me=True, struct_as_record=False)[
            "mnt"
        ]
        return cnt, mrk, mnt

    def _download_data(self, path, force_update, verbose):
        nirs_dir = self._subject_dir(path, 1).parent
        if not self.accept:
            raise AttributeError(LICENSE_ERROR)
        archive_path = Path(dl.data_dl(URL, DOWNLOAD_CODE, path, force_update, verbose))
        if force_update and nirs_dir.is_dir():
            shutil.rmtree(nirs_dir)
        nirs_dir.parent.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, nirs_dir.parent)
        return nirs_dir

    def _convert_session(self, session_data, marker_data, montage_data, montage):
        raw = self._make_cw_raw(session_data, montage_data, montage)
        raw.add_channels(
            [self._marker_stim_raw(raw.n_times, session_data.fs, marker_data)]
        )
        return self._standardize_raw(raw)

    def _make_cw_raw(self, session_data, montage_data, montage):
        data, ch_names = self._cw_channel_data(session_data, montage_data)
        raw = self._make_raw(
            data,
            ch_names,
            ["fnirs_cw_amplitude"] * len(ch_names),
            session_data.fs,
        )
        self._populate_fnirs_metadata(raw)
        raw.set_montage(montage)
        return raw

    @staticmethod
    def _cw_channel_data(session_data, montage_data):
        source_map = {
            orig: f"S{i + 1}" for i, orig in enumerate(montage_data.source.clab)
        }
        detector_map = {
            orig: f"D{i + 1}" for i, orig in enumerate(montage_data.detector.clab)
        }
        sd_pairs = [
            (montage_data.source.clab[src - 1], montage_data.detector.clab[det - 1])
            for src, det in montage_data.sd
        ]
        wavelengths = [int(wl) for wl in np.atleast_1d(session_data.wavelengths)]
        split = len(sd_pairs)
        low = session_data.x[:, :split].T * 1e-6
        high = session_data.x[:, split:].T * 1e-6
        ch_names, rows = [], []
        for idx, (source, detector) in enumerate(sd_pairs):
            pair = f"{source_map[source]}_{detector_map[detector]}"
            ch_names.extend([f"{pair} {wavelengths[0]}", f"{pair} {wavelengths[1]}"])
            rows.extend([low[idx], high[idx]])
        return np.asarray(rows), ch_names

    @staticmethod
    def _make_montage(montage_data):
        source_map = {
            orig: f"S{i + 1}" for i, orig in enumerate(montage_data.source.clab)
        }
        detector_map = {
            orig: f"D{i + 1}" for i, orig in enumerate(montage_data.detector.clab)
        }
        source_pos = {
            source_map[orig]: montage_data.source.pos_3d[:, idx] * HEAD_SCALE
            for idx, orig in enumerate(montage_data.source.clab)
        }
        detector_pos = {
            detector_map[orig]: montage_data.detector.pos_3d[:, idx] * HEAD_SCALE
            for idx, orig in enumerate(montage_data.detector.clab)
        }
        sensor_pos = {}
        for src, det in montage_data.sd:
            source_name = source_map[montage_data.source.clab[src - 1]]
            detector_name = detector_map[montage_data.detector.clab[det - 1]]
            sensor_pos[source_name] = source_pos[source_name]
            sensor_pos[detector_name] = detector_pos[detector_name]
        return BaseFNIRSDataset._make_head_montage(sensor_pos, FIDUCIALS)

    def _marker_stim_raw(self, n_times, sfreq, marker_data):
        samples = np.rint(
            np.atleast_1d(marker_data.time) * float(sfreq) / 1000.0
        ).astype(int)
        desc = np.atleast_1d(marker_data.event.desc).astype(int)
        class_names = np.atleast_1d(marker_data.className).tolist()
        codes = np.array(
            [self._class_name_map[class_names[idx - 1]] for idx in desc],
            dtype=float,
        )
        return self._make_stim_raw(n_times, sfreq, samples, codes)


class Shin2016FnirsMA(BaseShin2016Fnirs):
    """Shin 2016 mental arithmetic fNIRS dataset.

    Serial subtraction of a one-digit number, cued by a 2 s subtraction problem,
    versus rest, each for 10 s [1]_.

    References
    ----------
    .. [1] Shin, J., von Lühmann, A., Blankertz, B., Kim, D.-W., Jeong, J.,
           Hwang, H.-J., & Müller, K.-R. (2017). Open Access Dataset for
           EEG+NIRS Single-Trial Classification. IEEE Transactions on Neural
           Systems and Rehabilitation Engineering, 25(10), 1735-1745.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsMentalArithmetic",
            task_type="mental arithmetic",
            events=MA_EVENTS,
            n_classes=len(MA_EVENTS),
            class_labels=list(MA_EVENTS),
            trial_duration=10.0,
            stimulus_type="subtraction problem and fixation cross",
            # Beeps mark the start and end of each task period.
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=2 * N_TRIALS_PER_CLASS,
            n_trials_per_class={label: N_TRIALS_PER_CLASS for label in MA_EVENTS},
            trials_context="per subject",
        ),
        # Shrinkage LDA on HbO, 0-10 s task period, 10x5-fold CV (supporting material).
        performance={"accuracy_percent": 75.3},
        sessions_per_subject=len(MA_SESSION_INDICES),
        file_format="MATLAB",
    )

    _session_indices = MA_SESSION_INDICES
    _class_name_map = MA_CLASS_NAME_MAP

    def __init__(self, accept=False, subjects=None, sessions=None):
        super().__init__(
            accept=accept,
            subjects=SUBJECTS,
            sessions_per_subject=len(MA_SESSION_INDICES),
            events=MA_EVENTS,
            code="Shin2016FnirsMA",
            interval=INTERVAL,
            paradigm="fnirsMentalArithmetic",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )


class Shin2016FnirsMI(BaseShin2016Fnirs):
    """Shin 2016 motor imagery fNIRS dataset.

    Left or right hand gripping imagery at 1 Hz for 10 s, cued by a 2 s arrow
    [1]_.

    References
    ----------
    .. [1] Shin, J., von Lühmann, A., Blankertz, B., Kim, D.-W., Jeong, J.,
           Hwang, H.-J., & Müller, K.-R. (2017). Open Access Dataset for
           EEG+NIRS Single-Trial Classification. IEEE Transactions on Neural
           Systems and Rehabilitation Engineering, 25(10), 1735-1745.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorImagery",
            task_type="hand gripping motor imagery",
            events=MI_EVENTS,
            n_classes=len(MI_EVENTS),
            class_labels=list(MI_EVENTS),
            trial_duration=10.0,
            stimulus_type="arrow and fixation cross",
            # Beeps mark the start and end of each task period.
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=2 * N_TRIALS_PER_CLASS,
            n_trials_per_class={label: N_TRIALS_PER_CLASS for label in MI_EVENTS},
            trials_context="per subject",
        ),
        # Shrinkage LDA on HbO, 0-10 s task period, 10x5-fold CV (supporting material).
        performance={"accuracy_percent": 58.1},
        sessions_per_subject=len(MI_SESSION_INDICES),
        file_format="MATLAB",
    )

    _session_indices = MI_SESSION_INDICES
    _class_name_map = MI_CLASS_NAME_MAP

    def __init__(self, accept=False, subjects=None, sessions=None):
        super().__init__(
            accept=accept,
            subjects=SUBJECTS,
            sessions_per_subject=len(MI_SESSION_INDICES),
            events=MI_EVENTS,
            code="Shin2016FnirsMI",
            interval=INTERVAL,
            paradigm="fnirsMotorImagery",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )
