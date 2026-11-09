"""Gao2023 high-density fNIRS motor task dataset (OpenNeuro ds007420)."""

import math

from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from .bids_base import BaseBIDSFNIRSDataset, build_bids_fnirs_file_paths
from .metadata import FnirsAcquisitionMetadata
from .openneuro import download_openneuro_dataset_files
from .utils import override_raw_sfreq

_OPENNEURO_ID = "ds007420"
_SUBJECTS = [170, 171, 173, 174, 176, 177, 179, 181, 182, 183, 184, 185]
_EVENTS = {"Left": 1, "Right": 2}
_RUNS = ["01", "02", "03"]
_TASK = "BallSqueezing"
_SESSION = "02"
_CODE = "Gao2023FnirsSqueezing"
# Subject 179 is recorded at 11.625744 Hz and resampled to this rate.
SFREQ = 8.719308035714286
# Onsets within a run are at least 11.124 s apart.
INTERVAL = [-2, 9]
DOI = "10.1117/1.NPh.10.2.025007"


class Gao2023FnirsSqueezing(BaseBIDSFNIRSDataset):
    """High-density fNIRS ball-squeezing motor task.

    Left- or right-hand ball squeezing for 5 s, cued visually with a sound,
    followed by 5-15 s rest [1]_.

    Notes
    -----
    Only the three ball-squeezing runs of session 02 are loaded. The
    resting-state and motion-artifact tasks are excluded.

    References
    ----------
    .. [1] Gao, Y., Rogers, D., von Lühmann, A., Ortega-Martinez, A.,
           Boas, D. A., & Yücel, M. A. (2023). Short-separation regression
           incorporated diffuse optical tomography image reconstruction
           modeling for high-density functional near-infrared spectroscopy.
           Neurophotonics, 10(2), 025007.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 200},
            hardware="NIRx NIRSport2 (two cascaded 16x16)",
            # Recorded at Boston University, US.
            line_freq=60.0,
            montage="custom",
            # SNIRF probe wavelengths.
            wavelengths_nm=[760.0, 850.0],
            # Two patches of 7 sources and 16 detectors.
            n_sources=14,
            n_detectors=32,
            # The shortest channels are 19 mm. The paper averages them as a
            # "fake SS" regressor.
            n_short_channels=0,
            sensor_positions="3d",
            # SNIRF aux channels of three IMUs.
            other_modalities=["accelerometer", "gyroscope"],
        ),
        # OpenNeuro README.
        participants=ParticipantMetadata(
            n_subjects=len(_SUBJECTS),
            gender={"female": 6, "male": 6},
            age_mean=23.2,
            age_std=5.2,
            age_min=18.0,
            age_max=37.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="ball squeezing",
            events=_EVENTS,
            n_classes=len(_EVENTS),
            class_labels=list(_EVENTS),
            trial_duration=5.0,
            stimulus_type="visual cue with sound",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Yuanyuan Gao",
                "De'Ja Rogers",
                "Alexander von Lühmann",
                "Antonio Ortega-Martinez",
                "David A. Boas",
                "Meryem A. Yücel",
            ],
            institution="Boston University",
            country="US",
            repository="OpenNeuro",
            data_url="https://doi.org/10.18112/openneuro.ds007420.v1.0.5",
            license="CC0",
            publication_year=2023,
        ),
        data_structure=DataStructureMetadata(
            n_trials=90,
            n_trials_per_class={label: 45 for label in _EVENTS},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
        runs_per_session=len(_RUNS),
    )
    _bids_task = _TASK
    _bids_session = _SESSION
    _annotation_rename_map = {"1": "Left", "2": "Right"}  # noqa: RUF012

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=_SUBJECTS,
            sessions_per_subject=1,
            events=_EVENTS,
            code=_CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def _load_raw(self, bids_path, subject):
        raw = super()._load_raw(bids_path, subject)
        if math.isclose(raw.info["sfreq"], SFREQ, rel_tol=1e-6):
            return override_raw_sfreq(raw, SFREQ)
        raw.resample(SFREQ, verbose=False)
        return raw

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path, verbose
        return str(
            download_openneuro_dataset_files(
                _OPENNEURO_ID,
                self.code,
                path,
                self._openneuro_file_paths(subject),
                force_update,
            )
        )

    def _openneuro_file_paths(self, subject):
        bids_subject = self._bids_subject(subject)
        return [
            "dataset_description.json",
            *build_bids_fnirs_file_paths(
                bids_subject, _SESSION, [(_TASK, run) for run in _RUNS]
            ),
        ]
