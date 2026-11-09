"""Huang et al. 2021 mental-workload fNIRS dataset."""

from pathlib import Path

import numpy as np
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
)

from .base import BaseFNIRSDataset
from .metadata import FnirsAcquisitionMetadata

CODE = "Huang2021FnirsWorkload"
SUBJECTS = [
    1,
    5,
    7,
    13,
    14,
    15,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    29,
    31,
    32,
    34,
    35,
    36,
    37,
    38,
    40,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    51,
    52,
    54,
    55,
    56,
    57,
    58,
    60,
    61,
    62,
    63,
    64,
    65,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    91,
    92,
    93,
    94,
    95,
    97,
]
EVENTS = {"0-back": 1, "1-back": 2, "2-back": 3, "3-back": 4}
SFREQ = 5.2
N_EPOCHS = 16
EPOCH_SAMPLES = 426
WAVELENGTHS = (690, 830)
# The released epochs span 81.73 s and the last one ends with the recording.
# Resampling to 5 Hz can place its onset one sample late, which leaves 81.6 s.
INTERVAL = [0, 81.6]
DATA_URL = "https://tufts.app.box.com/s/1e0831syu1evlmk9zx2pukpl3i32md6r"
# Intensity-derived dual-slope estimates. The phase-derived estimates
# (*_PHI_*) are parallel estimates at the same locations and are not loaded.
SOURCE_COLUMNS = ["AB_I_O", "AB_I_DO", "CD_I_O", "CD_I_DO"]
CHANNEL_NAMES = ["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo", "S2_D2 hbr"]
MONTAGE_PATH = Path(__file__).parent / "montages" / "huang2021_montage_3d.json"


class Huang2021FnirsWorkload(BaseFNIRSDataset):
    """Huang2021 prefrontal fNIRS dataset with four n-back workload levels.

    Each subject completed 16 blocks of 40 digit n-back trials, 80 s per
    block, with n fixed to 0, 1, 2 or 3 per block [1]_.

    Notes
    -----
    Loads the released preprocessed epochs, one per block, concatenated into
    one Raw object as in MOABB's ``Hinss2021``. These are dual-slope HbO/HbR
    estimates, band-pass filtered at 0.001-0.2 Hz. Only the 68 eligible of 87
    recorded subjects are included. The authors excluded 5 for n-back
    performance, 5 for early sensor settings and 9 for abnormal oxygenation
    dynamics. The data must be downloaded manually.

    References
    ----------
    .. [1] Huang, Z., Wang, L., Blaney, G., Slaughter, C., McKeon, D., Zhou,
           Z., Jacob, R. J. K., & Hughes, M. C. (2021). The Tufts fNIRS Mental
           Workload Dataset & Benchmark for Brain-Computer Interfaces that
           Generalize. NeurIPS Datasets and Benchmarks Track.
           https://openreview.net/forum?id=QzNHE7QHhut
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"hbo": 2, "hbr": 2},
            hardware="ISS Imagent",
            # Recorded at Tufts University, Medford, MA, US.
            line_freq=60.0,
            montage="custom",
            wavelengths_nm=[float(w) for w in WAVELENGTHS],
            # Two probes, each a linear symmetric dual-slope set S-D-D-S
            # (paper Fig. C.1). Each set is loaded as one channel.
            n_sources=4,
            n_detectors=4,
            n_short_channels=0,
            # montage-toolkit positions, constructed from the paper's
            # channel-label image.
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"female": 39, "male": 26, "other": 2, "declined": 1},
            handedness={"right": 64, "left": 3, "unknown": 1},
            age_mean=21.71,
            age_std=4.01,
            age_min=18.0,
            age_max=44.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsNBack",
            task_type="n-back",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=80.0,
            stimulus_type="digits",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            investigators=[
                "Zhe Huang",
                "Liang Wang",
                "Giles Blaney",
                "Christopher Slaughter",
                "Devon McKeon",
                "Ziyu Zhou",
                "Robert J. K. Jacob",
                "Michael C. Hughes",
            ],
            institution="Tufts University",
            country="US",
            repository="Box",
            data_url=DATA_URL,
            # License source: https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html
            license="CC BY 4.0",
            publication_year=2021,
        ),
        data_structure=DataStructureMetadata(
            n_trials=N_EPOCHS,
            n_trials_per_class={label: N_EPOCHS // len(EVENTS) for label in EVENTS},
            trials_context="per subject",
        ),
        data_processed=True,
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
            preprocessing_applied=True,
            preprocessing_steps=["dual-slope HbO/HbR conversion", "band-pass"],
            bandpass=[0.001, 0.2],
            filter_type="butterworth",
            filter_order=3,
        ),
        file_format="CSV",
        # The paper and data have no DOI (dataset datasheet).
        external_links={
            "paper": "https://openreview.net/forum?id=QzNHE7QHhut",
            "website": "https://tufts-hci-lab.github.io/code_and_datasets/fNIRS2MW.html",
            "code": "https://github.com/tufts-ml/fNIRS-mental-workload-classifiers",
        },
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsNBack",
            doi=None,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return a manually downloaded subject CSV path."""
        del force_update, update_path, verbose
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        data_file = self._data_file(path, subject)
        if data_file.is_file():
            return str(data_file)
        raise FileNotFoundError(
            f"Download Huang2021 whole_data from {DATA_URL} and place "
            f"sub_{subject}.csv at {data_file}"
        )

    def _get_single_subject_data(self, subject):
        source = np.genfromtxt(
            self.data_path(subject), delimiter=",", names=True, encoding=None
        )
        raw = self._convert_subject(source)
        return self._single_run_session(raw)

    def _convert_subject(self, source):
        labels = np.asarray(source["label"], dtype=int)
        expected_samples = N_EPOCHS * EPOCH_SAMPLES
        if labels.size != expected_samples:
            raise ValueError(f"Expected {expected_samples} samples, got {labels.size}")
        epoch_labels = labels.reshape(N_EPOCHS, EPOCH_SAMPLES)
        event_labels = epoch_labels[:, 0] + 1
        if set(event_labels) != set(EVENTS.values()):
            raise ValueError(f"Unexpected event labels: {set(event_labels)}")
        # Source concentrations are in µM. MNE expects M.
        source_data = np.vstack([source[name] for name in SOURCE_COLUMNS]) * 1e-6
        # HACK: MOABB requires a leading zero to detect the first epoch without data loss.
        data = np.pad(source_data, ((0, 0), (1, 0)))
        raw = self._make_raw(
            data,
            CHANNEL_NAMES,
            ["hbo", "hbr"] * 2,
            SFREQ,
            self._load_montage_from_json(MONTAGE_PATH)[0],
        )
        # Fixed epoch boundaries, since consecutive epochs can share a label.
        samples = 1 + np.arange(N_EPOCHS) * EPOCH_SAMPLES
        raw.add_channels(
            [self._make_stim_raw(raw.n_times, SFREQ, samples, event_labels)]
        )
        return self._populate_fnirs_metadata(raw, default_wavelengths=WAVELENGTHS)

    @staticmethod
    def _data_file(path, subject):
        return (
            Path(dl.get_dataset_path(CODE, path))
            / f"MNE-{CODE.lower()}-data"
            / "whole_data"
            / f"sub_{subject}.csv"
        )
