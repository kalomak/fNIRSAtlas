"""Guevara2024 Parkinson's disease finger-tapping fNIRS dataset."""

import shutil
import warnings
from pathlib import Path

import mne
from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)

from .base import BaseFNIRSDataset
from .metadata import FnirsAcquisitionMetadata
from .preprocessing import _crop_leading_clipped_samples
from .utils import (
    deduplicate_annotations,
    download_lock,
    extract_archive,
    override_raw_sfreq,
    paired_fnirs_channel_order,
    snap_raw_sfreq,
)

MONTAGE_PATH = Path(__file__).parent / "montages" / "guevara2024_montage_3d.json"
URL = "https://zenodo.org/api/records/7966830/files/fNIRS_Parkinson.zip/content"
CODE = "Guevara2024FnirsParkinson"
DOI = "10.1117/1.NPh.11.2.025004"
SUBJECTS = list(range(1, 21))
EVENTS = {"Tapping/Left": 1, "Tapping/Right": 2}
# Rest periods are 20 to 24 s. 3 of 378 consecutive onset intervals are
# shorter than 17 s.
INTERVAL = [-2, 15]
# Paper rate. Recordings are at 25, 50 or 75 Hz, Control18 at 12.5 Hz.
SFREQ = 25.0
N_PAIRS = 22
CONTROL18_SFREQ = 12.5
SFREQ_STEP = 0.1
SFREQ_RTOL = 1e-5
# Brite recordings start with samples at 2**-16, the lowest exported intensity.
BRITE_FLOOR = 2**-16


class Guevara2024FnirsParkinson(BaseFNIRSDataset):
    """Guevara2024 Parkinson's disease finger-tapping fNIRS dataset.

    Left- and right-hand thumb-to-finger tapping in 10 s blocks cued by video,
    with 20 to 24 s rest, recorded over bilateral motor cortex [1]_.

    Notes
    -----
    Only the 20 healthy controls are loaded. The 20 Parkinson's disease
    patients and the walking and resting-state recordings are excluded.

    References
    ----------
    .. [1] Guevara, E., Rivas-Ruvalcaba, F. J., Kolosovas-Machuca, E. S.,
           Ramírez-Elías, M., Díaz de León Zapata, R., Ramirez-GarciaLuna, J. L.,
           & Rodríguez-Leyva, I. (2024). Parkinson's disease patients show
           delayed hemodynamic changes in primary motor cortex in fine motor
           tasks and decreased resting-state interhemispheric functional
           connectivity: a functional near-infrared spectroscopy study.
           Neurophotonics, 11(2), 025004.
    """

    METADATA = DatasetMetadata(
        acquisition=FnirsAcquisitionMetadata(
            sampling_rate=SFREQ,
            channel_types={"fnirs_cw_amplitude": 2 * N_PAIRS},
            hardware="Artinis Brite MKII",
            # Recorded in San Luis Potosí, Mexico.
            line_freq=60.0,
            montage="custom",
            wavelengths_nm=[757.0, 843.0],
            n_sources=10,
            n_detectors=8,
            n_short_channels=2,
            # montage-toolkit positions, constructed from the paper's Fig. 1b,
            # the SNIRF 2D channel grid and measurementList order, with C3/C4
            # as reference positions.
            sensor_positions="3d",
            # 13 SNIRF aux channels named aux1 to aux13.
            other_modalities=["unlabelled aux"],
        ),
        participants=ParticipantMetadata(
            n_subjects=len(SUBJECTS),
            # 45% of the 20 controls are men.
            gender={"female": 11, "male": 9},
            handedness={"right": 20},
            age_mean=65.6,
            age_std=9.9,
        ),
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="finger tapping",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            trial_duration=10.0,
            stimulus_type="video cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi=DOI,
            investigators=[
                "Edgar Guevara",
                "Francisco Javier Rivas-Ruvalcaba",
                "Eleazar Samuel Kolosovas-Machuca",
                "Miguel Ramírez-Elías",
                "Ramón Díaz de León Zapata",
                "Jose Luis Ramirez-GarciaLuna",
                "Ildefonso Rodríguez-Leyva",
            ],
            institution="Universidad Autónoma de San Luis Potosí",
            country="MX",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.7966830",
            license="CC BY 4.0",
            publication_year=2024,
        ),
        # Planned design. The source files hold 398 tapping events in total.
        data_structure=DataStructureMetadata(
            n_trials=20,
            n_trials_per_class={label: 10 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="SNIRF",
    )

    _annotation_rename_map = {"L": "Tapping/Left", "R": "Tapping/Right"}  # noqa: RUF012

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=SUBJECTS,
            sessions_per_subject=1,
            events=EVENTS,
            code=CODE,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
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
        """Return the control subject's finger-tapping SNIRF path."""
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        root = self._download_data(path, subject, force_update, verbose)
        subject_id = f"Control{subject:02d}"
        return str(
            root / "finger_tapping" / "Control" / subject_id / "2_finger_tapping.snirf"
        )

    def _get_single_subject_data(self, subject):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message="The data only contains 2D location information.*",
            )
            raw = mne.io.read_raw_snirf(
                self.data_path(subject), preload=True, verbose=False
            )
        raw.set_montage(self._load_montage_from_json(MONTAGE_PATH)[0])
        raw.reorder_channels(paired_fnirs_channel_order(raw.ch_names))
        if subject == 18:
            # Its metadata rate is twice the rate supported by duration and spectra.
            raw = override_raw_sfreq(raw, CONTROL18_SFREQ)
        raw = snap_raw_sfreq(raw, SFREQ_STEP, SFREQ_RTOL)
        raw = _crop_leading_clipped_samples(raw, floor=BRITE_FLOOR)
        raw = deduplicate_annotations(raw)
        return self._single_run_session(self._standardize_raw(raw))

    def _download_data(self, path, subject, force_update, verbose):
        cache = Path(dl.get_dataset_path(CODE, path)) / f"MNE-{CODE.lower()}-data"
        base = cache / "Guevara2024"
        subject_id = f"Control{subject:02d}"
        expected = base / "data/finger_tapping/Control" / subject_id
        # All subjects share one archive.
        with download_lock(base):
            if expected.is_dir() and not force_update:
                return base / "data"
            archive_path = dl.data_dl(URL, CODE, path, force_update, verbose)
            if base.is_dir():
                shutil.rmtree(base)
            base.mkdir(parents=True)
            extract_archive(archive_path, base / "data")
            if not expected.is_dir():
                raise FileNotFoundError(f"Archive does not contain {expected}")
        return base / "data"
