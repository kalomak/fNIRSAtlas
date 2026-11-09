"""Shin et al. ternary mental-task fNIRS datasets."""

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
from fnirsatlas.datasets.preprocessing import interpolate_dist
from fnirsatlas.datasets.utils import concentration_length_to_molar, extract_archive

# Shimadzu LIGHTNIRS wavelengths 780, 805 and 830 nm, also stated in the paper.
# 780 and 830 nm best separate HbO and HbR.
WAVELENGTHS = (780, 830)
URL = "https://ndownloader.figshare.com/files/16753760"
CODE = "Shin2018FnirsTernary"
DOI = "10.3389/fninf.2018.00005"
PARADIGM = "fnirsMentalArithmetic"
SUBJECTS = list(range(1, 19))
# Onsets are at least 29.25 s apart.
INTERVAL = [-2, 15]
EVENTS = {"mental_arithmetic": 1, "motor_imagery": 2, "baseline": 3}
N_PAIRS = 16
TRIALS_PER_CLASS = 30
# The paper's modified Beer-Lambert equation stores HbO/HbR in mM cm, with every
# pair 30 mm apart. The files give no partial pathlength factor, so the MNE
# default of 6 is used to convert to M.
SOURCE_DETECTOR_DISTANCE_CM = 3.0
PARTIAL_PATHLENGTH_FACTOR = 6.0
MONTAGE_PATH = Path(__file__).parent / "montages" / "shin2018_ternary_montage_3d.json"
CHANNEL_MAP = {
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


class Shin2018FnirsTernary(BaseFNIRSDataset):
    """Shin et al. ternary mental-task dataset.

    Mental arithmetic, right-hand motor imagery or idle state for 10 s after a
    2 s instruction, with prefrontal fNIRS [1]_.

    Notes
    -----
    Subject 7 lacks both chromophore channels of one pair. They are interpolated
    from neighbouring channels.

    References
    ----------
    .. [1] Shin, J., Kwon, J., & Im, C.-H. (2018). A ternary hybrid EEG-NIRS
           brain-computer interface for the classification of brain activation
           patterns during mental arithmetic, motor imagery, and idle state.
           Frontiers in Neuroinformatics, 12, 5.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=40 / 3,
            channel_types={"hbo": N_PAIRS, "hbr": N_PAIRS},
            hardware="Shimadzu LIGHTNIRS",
            # Recorded at Hanyang University, Seoul.
            line_freq=60.0,
            montage="custom",
            wavelengths_nm=[780.0, 805.0, 830.0],
            n_sources=6,
            n_detectors=6,
            n_short_channels=0,
            # montage-toolkit positions, constructed from the paper's optode image
            # and Fpz/AFz/Fp1/Fp2/F7/F8 reference positions.
            sensor_positions="3d",
            # EEG and EOG were recorded but are not in the fNIRS data record.
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"male": 10, "female": 8},
            age_mean=23.8,
            age_std=2.5,
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="mental arithmetic, motor imagery and idle state",
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
            doi=DOI,
            investigators=["Jaeyoung Shin", "Jinuk Kwon", "Chang-Hwan Im"],
            institution="Hanyang University",
            country="KR",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.9198932",
            license="CC BY 4.0",
            publication_year=2018,
        ),
        data_structure=DataStructureMetadata(
            n_trials=len(EVENTS) * TRIALS_PER_CLASS,
            n_trials_per_class={label: TRIALS_PER_CLASS for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="MATLAB",
        # NIRS-only ternary sLDA, one-versus-one, 10 x 10-fold cross-validation.
        performance={"accuracy_percent": 64.1, "accuracy_std": 9.7},
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm=PARADIGM,
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        subject_file = self._subject_file(path, subject)
        if subject_file.is_file() and not force_update:
            return str(subject_file)
        self._download_data(path, force_update, verbose)
        if not subject_file.is_file():
            raise FileNotFoundError(
                f"Missing Shin2018FnirsTernary data for subject {subject}"
            )
        return str(subject_file)

    def _get_single_subject_data(self, subject):
        data = loadmat(self.data_path(subject), squeeze_me=True, struct_as_record=False)
        return self._single_run_session(self._convert_subject(data))

    def _convert_subject(self, source):
        sfreq = float(source["nfo"].fs)
        data, ch_names, missing = self._channel_data(source)
        samples = np.rint(np.atleast_1d(source["mrk"].time) * sfreq / 1000).astype(int)
        labels = np.atleast_1d(source["mrk"].event.desc).astype(int)
        stim = self._make_stim_raw(data.shape[1], sfreq, samples, labels)
        raw = self._make_raw(
            np.vstack([data, stim.get_data()]),
            ch_names + ["Stim"],
            [name.split()[-1] for name in ch_names] + ["stim"],
            sfreq,
            montage=self._load_montage_from_json(MONTAGE_PATH)[0],
        )
        # Subject 7 lacks one pair, which is interpolated from its neighbours.
        raw.info["bads"] = missing
        raw = interpolate_dist(raw)
        raw = self._populate_fnirs_metadata(raw, default_wavelengths=WAVELENGTHS)
        return self._standardize_raw(raw)

    @staticmethod
    def _channel_data(source):
        observed = {}
        for index, label in enumerate(np.atleast_1d(source["nfo"].clab), start=1):
            name = str(label)
            chromophore = "hbr" if "deoxy" in name else "hbo"
            original = name.replace("deoxy", "").replace("oxy", "")
            observed[f"{CHANNEL_MAP[original]} {chromophore}"] = source[f"ch{index}"]
        ch_names = [
            f"{pair} {kind}" for pair in CHANNEL_MAP.values() for kind in ("hbo", "hbr")
        ]
        missing = [name for name in ch_names if name not in observed]
        template = next(iter(observed.values()))
        rows = [observed.get(name, np.zeros_like(template)) for name in ch_names]
        data = concentration_length_to_molar(
            rows, SOURCE_DETECTOR_DISTANCE_CM, PARTIAL_PATHLENGTH_FACTOR
        )
        return data, ch_names, missing

    @staticmethod
    def _subject_file(path, subject):
        return (
            Path(dl.get_dataset_path(CODE, path))
            / f"MNE-{CODE.lower()}-data"
            / "matdata"
            / f"subject_{subject:02d}.mat"
        )

    def _download_data(self, path, force_update, verbose):
        base_dir = self._subject_file(path, 1).parents[1]
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        base_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, base_dir)
