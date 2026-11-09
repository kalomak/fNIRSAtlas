"""Carlton2026 high-density fNIRS ball-squeeze dataset (OpenNeuro ds006673)."""

import mne
import numpy as np
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

OPENNEURO_ID = "ds006673"
SUBJECTS = [
    568,
    577,
    580,
    581,
    583,
    586,
    587,
    588,
    592,
    613,
    618,
    619,
    621,
    633,
    638,
    639,
    640,
]
# To include resting state, also download task RS run 01, remove the _bids_task
# filter, and handle its trial_type "0" separately in _load_raw.
EVENTS = {"left": 1, "right": 2}
RUNS = ["01", "02", "03"]
TASK = "BS"
SFREQ = 4.324324324324325
DOI = "10.1117/1.NPh.13.2.025001"
DATA_DOI = "10.18112/openneuro.ds006673.v1.0.4"
WAVELENGTHS_NM = [760.0, 850.0]
# Onsets are at least 21.0512 s apart.
INTERVAL = [-2, 15]
TRIALS_PER_RUN = 14


class Carlton2026FnirsSqueezing(BaseBIDSFNIRSDataset):
    """Carlton2026 high-density fNIRS ball-squeezing dataset.

    Subjects squeezed a ball with the left or right hand for 5 s, as prompted
    on a screen, with 15-20 s jittered rest [1]_.

    Notes
    -----
    The resting-state runs are not loaded. Subject 580 run 01 has 13 trials and
    subject 638 run 01 has 9. The first sample of 33 runs contains NaNs and is
    dropped.

    References
    ----------
    .. [1] Carlton, L. B., Altınkaynak, M., Kelley, S. M., Zimmermann, B. B.,
           Kura, S., Middell, E., von Lühmann, A., Stephen, E. P., Yücel, M. A.,
           & Boas, D. A. (2026). Surface-based image reconstruction optimization
           for high-density functional near-infrared spectroscopy.
           Neurophotonics, 13(2), 025001.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            # 1220 pairs at 15-60 mm. The paper analyses the 567 pairs at 19 and
            # 33 mm.
            channel_types={"fnirs_cw_amplitude": 2440},
            hardware="NinjaNIRS2022",
            # Recorded at Boston University, US.
            line_freq=60.0,
            montage="custom",
            wavelengths_nm=WAVELENGTHS_NM,
            n_sources=56,
            # The BIDS sidecars state 56 detectors, the SNIRF probe and paper 144.
            n_detectors=144,
            n_short_channels=0,
            # SNIRF probe positions, identical for all subjects.
            sensor_positions="3d",
            # SNIRF aux "analog-1", "analog-2" and "digital" are unlabelled.
            other_modalities=[
                "accelerometer",
                "gyroscope",
                "temperature",
                "unlabelled aux",
            ],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            gender={"female": 7, "male": 10},
            handedness={"right": 17},
            # participants.tsv, one subject's age is n/a.
            age_mean=30.6,
            age_std=9.7,
            age_min=21.0,
            age_max=61.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="ball squeezing",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=5.0,
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Laura B. Carlton",
                "Miray Altınkaynak",
                "Shannon M. Kelley",
                "Bernhard B. Zimmermann",
                "Sreekanth Kura",
                "Eike Middell",
                "Alexander von Lühmann",
                "Emily P. Stephen",
                "Meryem A. Yücel",
                "David A. Boas",
            ],
            institution="Boston University",
            country="US",
            repository="OpenNeuro",
            data_url=f"https://doi.org/{DATA_DOI}",
            license="CC0",
            publication_year=2026,
        ),
        # Three runs of 14 trials. Subjects 580 and 638 have fewer.
        data_structure=DataStructureMetadata(
            n_trials=len(RUNS) * TRIALS_PER_RUN,
            n_trials_per_class={
                label: len(RUNS) * TRIALS_PER_RUN // len(EVENTS) for label in EVENTS
            },
            trials_context="per subject",
        ),
        runs_per_session=len(RUNS),
        file_format="BIDS SNIRF",
    )
    _bids_task = TASK

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code="Carlton2026FnirsSqueezing",
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
        )

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path, verbose
        return str(
            download_openneuro_dataset_files(
                OPENNEURO_ID,
                self.code,
                path,
                self._openneuro_file_paths(subject),
                force_update,
            )
        )

    def _load_raw(self, bids_path, subject):
        raw = super()._load_raw(bids_path, subject)

        # 33 recordings have a Nan at start
        first_sample = raw.get_data(picks="fnirs_cw_amplitude", stop=1)
        if not np.isfinite(first_sample).all():
            raw.crop(tmin=1 / raw.info["sfreq"])

        events = self._read_bids_events(bids_path)
        unknown = set(events["trial_type"]) - EVENTS.keys()
        if unknown:
            raise ValueError(
                f"Unknown Carlton2026FnirsSqueezing trial types: {sorted(unknown)}"
            )
        raw.set_annotations(
            mne.Annotations(
                events["onset"],
                events["duration"],
                events["trial_type"],
                orig_time=raw.info["meas_date"],
            )
        )
        return raw

    def _openneuro_file_paths(self, subject):
        files = build_bids_fnirs_file_paths(
            self._bids_subject(subject), None, [(TASK, run) for run in RUNS]
        )
        return [
            "dataset_description.json",
            "participants.json",
            "participants.tsv",
            *files,
        ]
