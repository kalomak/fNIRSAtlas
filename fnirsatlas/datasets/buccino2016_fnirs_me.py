"""Buccino et al. 2016 EEG-fNIRS hybrid motor execution dataset."""

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

MONTAGE_PATH = Path(__file__).parent / "montages" / "buccino2016_montage_3d.json"
URL_S1_S8 = "https://ndownloader.figshare.com/files/2601537"
URL_S9_S15 = "https://ndownloader.figshare.com/files/2601481"
CODE = "Buccino2016FnirsME"
SFREQ = 10.42
N_CHANNELS = 34
EVENTS = {"right_arm": 3, "left_arm": 4, "right_hand": 7, "left_hand": 8}
ACTIVE_CODES = set(EVENTS.values())
ORIG2MNE = {
    "Fc3A": "S1_D1",
    "Fc1A": "S1_D2",
    "Fc3L": "S2_D1",
    "Fc3M": "S1_D3",
    "Fc1M": "S3_D2",
    "C3A": "S2_D3",
    "C1A": "S3_D3",
    "C3L": "S2_D4",
    "C3M": "S4_D3",
    "C1M": "S3_D5",
    "Cp3A": "S4_D4",
    "Cp1A": "S4_D5",
    "Cp3L": "S5_D4",
    "Cp3M": "S4_D6",
    "Cp1M": "S6_D5",
    "Cp3P": "S5_D6",
    "Cp1P": "S6_D6",
    "Fc2A": "S7_D7",
    "Fc4A": "S7_D8",
    "Fc2M": "S8_D7",
    "Fc4M": "S7_D9",
    "Fc4L": "S9_D8",
    "C2A": "S8_D9",
    "C4A": "S9_D9",
    "C2M": "S8_D10",
    "C4M": "S10_D9",
    "C4L": "S9_D11",
    "Cp2A": "S10_D10",
    "Cp4A": "S10_D11",
    "Cp2M": "S11_D10",
    "Cp4M": "S10_D12",
    "Cp4L": "S12_D11",
    "Cp2P": "S11_D12",
    "Cp4P": "S12_D12",
}
CH_NAMES_ORDERED = [
    "Fc3A",
    "Fc1A",
    "Fc3L",
    "Fc3M",
    "Fc1M",
    "C3A",
    "C1A",
    "C3L",
    "C3M",
    "C1M",
    "Cp3A",
    "Cp1A",
    "Cp3L",
    "Cp3M",
    "Cp1M",
    "Cp3P",
    "Cp1P",
    "Fc2A",
    "Fc4A",
    "Fc2M",
    "Fc4M",
    "Fc4L",
    "C2A",
    "C4A",
    "C2M",
    "C4M",
    "C4L",
    "Cp2A",
    "Cp4A",
    "Cp2M",
    "Cp4M",
    "Cp4L",
    "Cp2P",
    "Cp4P",
]


class Buccino2016FnirsME(BaseFNIRSDataset):
    """Buccino et al. 2016 EEG-fNIRS hybrid motor execution dataset.

    Four executed upper-limb movements: right/left arm raising and right/left
    hand gripping, each cued for 6 s after 6 s rest [1]_.

    References
    ----------
    .. [1] Buccino, A. P., Keles, H. O., & Omurtag, A. (2016).
           Hybrid EEG-fNIRS asynchronous brain-computer interface for
           multiple motor tasks. PloS one, 11(1), e0146610.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 2 * N_CHANNELS},
            hardware="NIRx NIRScout 8-16",
            # Recorded at the University of Houston, US.
            line_freq=60.0,
            montage="custom",
            wavelengths_nm=[760.0, 850.0],
            n_sources=12,
            n_detectors=12,
            n_short_channels=0,
            # montage-toolkit positions, constructed from the paper's channel-label
            # image and 10-20 reference positions.
            sensor_positions="3d",
            other_modalities=["EEG"],
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            gender={"male": 15},
            handedness={"right": 15},
            age_mean=27.4,
            age_std=7.7,
            age_min=23.0,
            age_max=54.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="upper-limb movement",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=6.0,
            stimulus_type="textual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0146610",
            investigators=[
                "Alessio Paolo Buccino",
                "Hasan Onur Keles",
                "Ahmet Omurtag",
            ],
            institution="University of Houston",
            country="US",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.1619640",
            license="CC BY 4.0",
            publication_year=2016,
        ),
        data_structure=DataStructureMetadata(
            n_trials=100,
            n_trials_per_class={label: 25 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="MATLAB",
        external_links={
            "data_s1_s8": "https://doi.org/10.6084/m9.figshare.1619640",
            "data_s9_s15": "https://doi.org/10.6084/m9.figshare.1619641",
        },
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            # Onsets are at least 14.011 s apart.
            interval=[-2, 11.8],
            paradigm="fnirsMotorExecution",
            doi="10.1371/journal.pone.0146610",
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
        del update_path
        base_dir = self._download_data(path, subject, force_update, verbose)
        return str(base_dir / f"S{subject}.mat")

    def _get_single_subject_data(self, subject):
        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True, struct_as_record=False)
        raw = self._convert_subject(mat)
        return self._single_run_session(raw)

    def _convert_subject(self, mat):
        data = mat["data"]
        nirs = data[1]
        sfreq = float(nirs[0])
        wl1 = nirs[4]  # (n_times, 34)
        wl2 = nirs[5]  # (n_times, 34)
        ch_data, ch_names = self._build_cw_channels(wl1, wl2)
        montage = self._load_montage_from_json(MONTAGE_PATH)[0]
        raw = self._make_raw(
            ch_data,
            ch_names,
            ["fnirs_cw_amplitude"] * len(ch_names),
            sfreq,
        )
        self._populate_fnirs_metadata(raw)
        raw.set_montage(montage)
        stim_raw = self._build_stim(data[2], sfreq, raw.n_times)
        raw.add_channels([stim_raw])
        return self._standardize_raw(raw)

    @staticmethod
    def _build_cw_channels(wl1, wl2):
        ch_names, rows = [], []
        for idx, orig_name in enumerate(CH_NAMES_ORDERED):
            pair = ORIG2MNE[orig_name]
            ch_names.extend([f"{pair} 760", f"{pair} 850"])
            rows.extend([wl1[:, idx], wl2[:, idx]])
        return np.asarray(rows), ch_names

    @staticmethod
    def _build_stim(events_raw, sfreq, n_times):
        mask = np.isin(events_raw[:, 1], list(ACTIVE_CODES))
        active = events_raw[mask]
        samples = np.rint(active[:, 0] * sfreq).astype(int)
        labels = active[:, 1].astype(float)
        return BaseFNIRSDataset._make_stim_raw(n_times, sfreq, samples, labels)

    def _download_data(self, path, subject, force_update, verbose):
        data_dir = Path(dl.get_dataset_path(CODE, path))
        base_dir = data_dir / f"MNE-{CODE.lower()}-data"
        if subject <= 8:
            subdir = "EEG-fNIRS hybrid SMR BCI data S1-S8"
            url = URL_S1_S8
        else:
            subdir = "EEG-fNIRS hybrid SMR BCI data S9-S15"
            url = URL_S9_S15
        subject_dir = base_dir / subdir
        marker = subject_dir / f"S{subject}.mat"
        if marker.is_file() and not force_update:
            return subject_dir
        archive_path = Path(dl.data_dl(url, CODE, path, force_update, verbose))
        base_dir.mkdir(parents=True, exist_ok=True)
        extract_archive(archive_path, base_dir)
        return subject_dir
