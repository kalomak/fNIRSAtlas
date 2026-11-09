"""Shin et al. 2018 eyes-closed mental-arithmetic fNIRS dataset."""

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
from fnirsatlas.datasets.utils import extract_archive

# Recorded with a NIRx NIRScout, whose standard wavelengths are 760 and 850 nm.
WAVELENGTHS = (760, 850)
URL = "https://ndownloader.figshare.com/files/10515928"
CODE = "Shin2018bFnirsMA"
DOI = "10.1371/journal.pone.0196359"
PARADIGM = "fnirsMentalArithmetic"
SUBJECTS = list(range(1, 13))
# Recorded at 12.5 Hz, released at 10 Hz.
SFREQ = 10.0
EVENTS = {"mental_arithmetic": 1, "baseline": 2}
CLASS_NAME_MAP = {"MA": 1, "BL": 2}
N_TRIALS_PER_CLASS = 30
# Onsets are at least 28.4 s apart.
INTERVAL = [-2, 15]
CHANNEL_PAIRS = [
    "S1_D1",
    "S2_D1",
    "S2_D2",
    "S3_D1",
    "S3_D2",
    "S3_D3",
    "S4_D2",
    "S4_D3",
    "S5_D3",
]
MONTAGE_PATH = Path(__file__).parent / "montages" / "shin2018b_montage_3d.json"


class Shin2018bFnirsMA(BaseFNIRSDataset):
    """Shin et al. 2018 eyes-closed mental-arithmetic fNIRS dataset.

    Repeated one-digit subtraction (mental arithmetic) or relaxing (baseline)
    with eyes closed for 10 s, after a 2 s visual instruction [1]_.

    Notes
    -----
    The source files provide channel centers, not optode coordinates. The
    5-source, 3-detector montage is inferred from those centers and the
    published layout.

    References
    ----------
    .. [1] Shin, J., Müller, K.-R., & Hwang, H.-J. (2018). Eyes-closed hybrid
           brain-computer interface employing frontal brain activation. PLOS
           ONE, 13(5), e0196359.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"hbo": len(CHANNEL_PAIRS), "hbr": len(CHANNEL_PAIRS)},
            hardware="NIRx NIRScout",
            # Approved by the TU Berlin ethics committee, DE.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[float(wavelength) for wavelength in WAVELENGTHS],
            n_sources=5,
            n_detectors=3,
            n_short_channels=0,
            # montage-toolkit positions, constructed from the 10-05 spherical
            # positions of the channel-center labels, with optodes inferred at
            # 30 mm separation.
            sensor_positions="3d",
            other_modalities=["EEG", "EOG"],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"male": 5, "female": 7},
            age_mean=26.7,
            age_std=3.7,
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="mental arithmetic",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=10.0,
            stimulus_type="subtraction problem and fixation cross",
            # A beep marks the start and end of each task period.
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=["Jaeyoung Shin", "Klaus-Robert Müller", "Han-Jeong Hwang"],
            institution="Technische Universität Berlin",
            country="DE",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.5900842.v1",
            license="CC BY 4.0",
            publication_year=2018,
        ),
        data_structure=DataStructureMetadata(
            n_trials=N_TRIALS_PER_CLASS * len(EVENTS),
            n_trials_per_class={label: N_TRIALS_PER_CLASS for label in EVENTS},
            trials_context="per subject",
        ),
        # Shrinkage LDA on HbO and HbR features, 10 x 5-fold cross-validation.
        performance={"accuracy_percent": 75.9, "accuracy_std": 6.3},
        file_format="MATLAB",
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
        subject_dir = self._subject_dir(path, subject)
        if self._has_subject_files(subject_dir) and not force_update:
            return str(subject_dir)
        self._download_data(path, force_update, verbose)
        if not self._has_subject_files(subject_dir):
            raise FileNotFoundError(f"Missing Shin2018b data for subject {subject}")
        return str(subject_dir)

    def _get_single_subject_data(self, subject):
        subject_dir = Path(self.data_path(subject))
        cnt = loadmat(subject_dir / "cnt.mat", squeeze_me=True, struct_as_record=False)[
            "cnt"
        ]
        mrk = loadmat(subject_dir / "mrk.mat", squeeze_me=True, struct_as_record=False)[
            "mrk"
        ]
        return self._single_run_session(self._convert_subject(cnt, mrk))

    def _convert_subject(self, cnt, mrk):
        sfreq = float(cnt.oxy.fs)
        # cnt yUnit is mmol/L, MNE expects mol/L.
        data = np.vstack([cnt.oxy.x.T, cnt.deoxy.x.T]) * 1e-3
        samples = np.rint(np.atleast_1d(mrk.oxy.time) * sfreq / 1000).astype(int)
        class_names = np.atleast_1d(mrk.oxy.className).tolist()
        labels = [CLASS_NAME_MAP[class_names[i - 1]] for i in mrk.oxy.event.desc]
        stim = self._make_stim_raw(data.shape[1], sfreq, samples, labels)
        ch_names = (
            [f"{pair} hbo" for pair in CHANNEL_PAIRS]
            + [f"{pair} hbr" for pair in CHANNEL_PAIRS]
            + ["Stim"]
        )
        raw = self._make_raw(
            np.vstack([data, stim.get_data()]),
            ch_names,
            ["hbo"] * 9 + ["hbr"] * 9 + ["stim"],
            sfreq,
            montage=self._load_montage_from_json(MONTAGE_PATH)[0],
        )
        raw = self._populate_fnirs_metadata(raw, default_wavelengths=WAVELENGTHS)
        return self._standardize_raw(raw)

    @staticmethod
    def _subject_dir(path, subject):
        return (
            Path(dl.get_dataset_path(CODE, path))
            / f"MNE-{CODE.lower()}-data"
            / "NIRS"
            / f"VP{subject:03d}"
        )

    @staticmethod
    def _has_subject_files(subject_dir):
        return all(
            (subject_dir / filename).is_file()
            for filename in ("cnt.mat", "mrk.mat", "mnt.mat")
        )

    def _download_data(self, path, force_update, verbose):
        base_dir = self._subject_dir(path, 1).parents[1]
        archive_path = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        base_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, base_dir)
