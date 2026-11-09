"""BNCI Horizon 2020 fNIRS Mental Arithmetic dataset (003-2014)."""

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
from fnirsatlas.datasets.utils import concentration_length_to_molar

# Recorded with a Hitachi ETG-4000, which uses 695 and 830 nm.
WAVELENGTHS = (695, 830)
MONTAGE_PATH = Path(__file__).parent / "montages" / "bnci_montage_3d.json"

DOWNLOAD_CODE = "Bnci2014Fnirs"
BNCI_URL = "https://bnci-horizon-2020.eu/database/data-sets/003-2014/"
SESSIONS = {
    1: [1, 2, 3],
    2: [1, 2, 3],
    3: [1, 2, 3],
    4: [1, 2, 3, 4],
    5: [1, 2, 3, 4],
    6: [1, 2, 3, 4],
    7: [1, 2, 3, 4],
    8: [1, 2, 3, 4],
}
N_CHANNELS = 52
SFREQ = 10.0
SOURCE_DETECTOR_DISTANCE_CM = 3.0
SUBJECTS = list(range(1, 9))
EVENTS = {"mental_arithmetic": 1, "baseline": 2}
# Baseline markers are shifted 8 s later to separate them from the preceding
# arithmetic epoch. Shifted onsets are at least 20 s apart.
INTERVAL = [-2, 15]
DOI = "10.1007/s11517-011-0792-5"


class Bnci2014FnirsMA(BaseFNIRSDataset):
    """BNCI Horizon 2020 fNIRS mental arithmetic dataset (003-2014).

    Cue-guided serial subtraction of a one-digit number from a two-digit number
    for 12 s, followed by 28 s of rest [1]_.

    References
    ----------
    .. [1] Bauernfeind, G., Scherer, R., Pfurtscheller, G., & Neuper, C. (2011).
           Single-trial classification of antagonistic oxyhemoglobin responses
           during mental arithmetic. Medical & Biological Engineering &
           Computing, 49(9), 979-984.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"hbo": N_CHANNELS, "hbr": N_CHANNELS},
            hardware="Hitachi ETG-4000",
            # Recorded at Graz University of Technology, AT.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[float(w) for w in WAVELENGTHS],
            n_sources=17,
            n_detectors=16,
            n_short_channels=0,
            # montage-toolkit positions, constructed from Fig. 2 of the dataset
            # description (3 x 11 grid, 30 mm source-detector distance) and the
            # Fp1/Fp2 reference positions.
            sensor_positions="3d",
            other_modalities=[],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"male": 3, "female": 5},
            age_mean=26.0,
            age_std=2.8,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMentalArithmetic",
            task_type="mental arithmetic",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=12.0,
            stimulus_type="subtraction cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Günther Bauernfeind",
                "Reinhold Scherer",
                "Gert Pfurtscheller",
                "Christa Neuper",
            ],
            institution="Graz University of Technology",
            country="AT",
            repository="BNCI Horizon 2020",
            data_url=BNCI_URL,
            license="CC BY-ND 4.0",
            publication_year=2011,
        ),
        data_structure=DataStructureMetadata(
            # Trials per subject. Subjects 1-3 have 3 sessions and subjects 4-8
            # have 4 sessions of 6 trials per class. n_trials_per_class is for
            # subjects 4-8.
            n_trials={
                str(subject): 36 if subject <= 3 else 48 for subject in range(1, 9)
            },
            n_trials_per_class={label: 24 for label in EVENTS},
            trials_context="per subject",
        ),
        # Subjects 1-3 have 3 sessions, subjects 4-8 have 4.
        sessions_per_subject=3,
        file_format="MATLAB",
        # Per-subject antagonistic [oxy-Hb] two-channel features, from the abstract.
        performance={"accuracy_percent": 79.7},
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=3,
            events=EVENTS,
            code="Bnci2014FnirsMA",
            interval=INTERVAL,
            paradigm="fnirsMentalArithmetic",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        url = f"{BNCI_URL}S{subject:02d}.mat"
        return dl.data_dl(url, DOWNLOAD_CODE, path, force_update, verbose)

    def _get_single_subject_data(self, subject):
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True, struct_as_record=False)
        data_array = mat["data"]
        session_list = SESSIONS[subject]
        sessions = {}
        for sess_idx, sess_num in enumerate(session_list):
            raw = self._convert_session(data_array[sess_idx])
            self._add_run(sessions, f"{sess_num}fnirs", raw)
        return sessions

    def _convert_session(self, session_data):
        X = session_data.X  # (n_times, 156)
        trial_samples = session_data.trial.flatten()
        trial_labels = session_data.y.flatten()
        # Offset rest triggers by 8s to separate from preceding MA epoch
        rest_mask = trial_labels == 2
        trial_samples[rest_mask] += int(8 * SFREQ - 1)
        # HbO (0:52) + HbR (52:104), drop HbT. Values are mM mm, so /10 gives mM cm.
        ch_data = concentration_length_to_molar(
            X[:, :104].T / 10, SOURCE_DETECTOR_DISTANCE_CM
        )
        stim_raw = self._make_stim_raw(
            ch_data.shape[1], SFREQ, trial_samples, trial_labels
        )
        data = np.vstack([ch_data, stim_raw.get_data()])
        montage, channel_map = self._load_montage_from_json(MONTAGE_PATH)
        # Data column k is channel k of the dataset description figure.
        pairs = [channel_map[str(column)] for column in range(1, N_CHANNELS + 1)]
        ch_names = (
            [f"{pair} hbo" for pair in pairs]
            + [f"{pair} hbr" for pair in pairs]
            + ["Stim"]
        )
        ch_types = ["hbo"] * N_CHANNELS + ["hbr"] * N_CHANNELS + ["stim"]
        raw = self._make_raw(data, ch_names, ch_types, SFREQ, montage=montage)
        raw = self._populate_fnirs_metadata(raw, default_wavelengths=WAVELENGTHS)
        return self._standardize_raw(raw)
