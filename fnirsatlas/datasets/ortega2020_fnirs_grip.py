"""Ortega et al. 2020 HYGRIP hand-grip fNIRS dataset."""

from pathlib import Path

import h5py
import numpy as np
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from fnirsatlas.datasets.base import BaseFNIRSDataset
from fnirsatlas.datasets.metadata import FnirsAcquisitionMetadata

URL = "https://ndownloader.figshare.com/files/22837760"
DATA_DOI_URL = "https://doi.org/10.6084/m9.figshare.12383639.v3"
DOI = "10.3389/fnins.2020.00919"
CODE = "Ortega2020FnirsGrip"
PARADIGM = "fnirsMotorExecution"
# Source subject IDs A to N map to 1 to 14.
SUBJECTS = list(range(1, 15))
EVENTS = {"left": 1, "right": 2}
# Onsets are at least 40.64 s apart. 25 s after the Go cue is the paper's
# validation epoch end.
INTERVAL = [-2, 25]
SFREQ = 12.5
WAVELENGTHS = (760, 850)
CHANNEL_PAIRS = [
    "S1_D1",
    "S2_D1",
    "S1_D2",
    "S3_D1",
    "S2_D3",
    "S3_D2",
    "S3_D3",
    "S4_D2",
    "S3_D4",
    "S5_D3",
    "S4_D4",
    "S5_D4",
    "S6_D6",
    "S7_D6",
    "S6_D7",
    "S8_D6",
    "S7_D8",
    "S8_D7",
    "S8_D8",
    "S9_D7",
    "S8_D9",
    "S10_D8",
    "S9_D9",
    "S10_D9",
]
MONTAGE_PATH = Path(__file__).parent / "montages" / "ortega2020_montage_3d.json"


def _interleave_wavelengths(wavelength_1_channel_data, wavelength_2_channel_data):
    """Interleave wavelength rows for each source-detector pair."""
    return np.stack(
        [wavelength_1_channel_data, wavelength_2_channel_data], axis=1
    ).reshape(-1, wavelength_1_channel_data.shape[-1])


class Ortega2020FnirsGrip(BaseFNIRSDataset):
    """Ortega et al. 2020 HYGRIP left- and right-hand dynamic grip dataset.

    Unimanual grip trials of 10 contraction (1.55 s) and relaxation (0.55 s)
    periods at 25-50 % of maximum voluntary contraction, with the hand announced
    by synthetic voice 2.5 s before the visual Go cue and 15-21 s rest between
    trials [1]_.

    References
    ----------
    .. [1] Ortega, P., Zhao, T., & Faisal, A. A. (2020). HYGRIP: Full-stack
           characterization of neurobehavioral signals (fNIRS, EEG, EMG, force,
           and breathing) during a bimanual grip force control task. Frontiers
           in Neuroscience, 14, 919.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": len(CHANNEL_PAIRS) * len(WAVELENGTHS)},
            hardware="NIRx NIRScout",
            # The paper's EEG pipeline notches the 50 Hz mains.
            line_freq=50.0,
            montage="custom",
            wavelengths_nm=[float(w) for w in WAVELENGTHS],
            n_sources=10,
            n_detectors=8,
            n_short_channels=0,
            # montage-toolkit positions, constructed from the paper's probe image
            # and the C3/C4 reference positions. Grid order follows
            # `scalp_chan_grid` and `scalp_chan_locs` in `hygrip.h5`.
            sensor_positions="3d",
            other_modalities=["EEG", "EMG", "EOG", "force", "respiration"],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            handedness={"right": len(SUBJECTS)},
        ),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            task_type="hand gripping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=21.0,
            stimulus_type="spoken hand cue and visual force target",
            stimulus_modalities=["auditory", "visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=["Pablo Ortega", "Tong Zhao", "A. Aldo Faisal"],
            institution="Imperial College London",
            country="GB",
            repository="figshare",
            data_url=DATA_DOI_URL,
            license="MIT",
            publication_year=2020,
        ),
        # Subjects 7 to 14 have 13 trials per class, subjects 1 to 6 have 10.
        data_structure=DataStructureMetadata(
            n_trials=26,
            n_trials_per_class={label: 13 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="HDF5",
        external_links={"code": "https://gitlab.doc.ic.ac.uk/bbl/hygrip.git"},
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
        data_file = self._data_file(path)
        if data_file.is_file() and not force_update:
            return str(data_file)
        downloaded = Path(dl.data_dl(URL, CODE, path, force_update, verbose))
        data_file.parent.mkdir(parents=True, exist_ok=True)
        if downloaded != data_file:
            downloaded.replace(data_file)
        return str(data_file)

    def _get_single_subject_data(self, subject):
        subject_id = chr(ord("A") + subject - 1)
        with h5py.File(self.data_path(subject), "r") as source:
            wavelength_1_channel_data = source[subject_id]["oi1"][:]
            wavelength_2_channel_data = source[subject_id]["oi2"][:]
            events = source[subject_id]["oi1"].attrs["events"]
        raw = self._convert_subject(
            wavelength_1_channel_data, wavelength_2_channel_data, events
        )
        return self._single_run_session(raw)

    def _convert_subject(
        self, wavelength_1_channel_data, wavelength_2_channel_data, events
    ):
        # [ch1 wl1, ch2 wl1, ...], [ch1 wl2, ch2 wl2, ...] > [ch1 wl1, ch1 wl2, ch2 wl1, ...]
        data = _interleave_wavelengths(
            wavelength_1_channel_data, wavelength_2_channel_data
        )
        ch_names = [
            f"{pair} {wavelength}"
            for pair in CHANNEL_PAIRS
            for wavelength in WAVELENGTHS
        ]
        raw = self._make_raw(
            data, ch_names, ["fnirs_cw_amplitude"] * len(ch_names), SFREQ
        )
        self._populate_fnirs_metadata(raw)
        raw.set_montage(self._load_montage_from_json(MONTAGE_PATH)[0])
        active = events[events[:, 1] >= 0]
        samples = np.rint(active[:, 0] * SFREQ).astype(int)
        labels = active[:, 1].astype(int) + 1
        raw.add_channels([self._make_stim_raw(raw.n_times, SFREQ, samples, labels)])
        return self._standardize_raw(raw)

    @staticmethod
    def _data_file(path):
        return (
            Path(dl.get_dataset_path(CODE, path))
            / f"MNE-{CODE.lower()}-data"
            / "hygrip.h5"
        )
