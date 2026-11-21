"""Yucel2025 fNIRS ball-squeezing and resting-state datasets (OpenNeuro ds006377)."""

import logging

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
from .utils import combine_devices, deduplicate_annotations

_OPENNEURO_ID = "ds006377"
_SUBJECTS = [225, 229, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 252, 253, 259, 266, 269, 270, 274, 276, 277, 286, 287, 291, 293, 304, 326, 327, 332, 338, 341, 343, 349, 350, 356, 357, 362, 368, 369, 370, 371, 377, 380, 381, 386, 390, 391, 392, 394, 396, 397, 400, 401, 402, 403, 404, 405, 406, 407, 411, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 437, 438, 439, 440, 441, 442, 443, 449, 450, 451, 452, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 472, 473, 474, 475]  # fmt: skip
_DEVICES = ("LH", "RH")
DOI = "10.1038/s41562-025-02274-7"
DATA_DOI = "10.18112/openneuro.ds006377.v1.0.2"
# Device sampling rate stored in each SNIRF recording.
SFREQ = 10.172526041666666
EVENTS = {"left": 1, "right": 2}
# Onsets are at least 11 s apart.
INTERVAL = [-2, 9]
REST_EVENTS = {"rest": 1}
REST_PARADIGM = "fnirsRestingState"
# Two 3-min resting-state runs, after fast and after proper capping.
REST_RUNS = 2
# Describes one device recording, the default RH device. Both devices are NIRx
# NIRSport2 systems, RH with SiPD and LH with APD detectors.
ACQUISITION = FnirsAcquisitionMetadata(
    sampling_rate=SFREQ,
    channel_types={"fnirs_cw_amplitude": 52},
    hardware="NIRx NIRSport2",
    # Recorded at Boston University, US.
    line_freq=60.0,
    montage="custom",
    wavelengths_nm=[760.0, 850.0],
    # BIDS nirs.json. Detectors D9-D16 are the short-separation detectors.
    n_sources=8,
    n_detectors=16,
    # One 7-9 mm pair per source in the SNIRF files.
    n_short_channels=8,
    # 3D optode positions in the SNIRF files.
    sensor_positions="3d",
    other_modalities=[],
)
# OpenNeuro participants.tsv.
PARTICIPANTS = ParticipantMetadata(
    n_subjects=len(_SUBJECTS),
    gender={"female": 63, "male": 52},
    age_mean=26.4,
    age_std=10.3,
    age_min=18.0,
    age_max=74.0,
)
DOCUMENTATION = DocumentationMetadata(
    doi=DOI,
    investigators=[
        "Meryem A. Yücel",
        "Jessica E. Anderson",
        "De'Ja Rogers",
        "Parisa Hajirahimi",
        "Parya Farzam",
        "Yuanyuan Gao",
        "Rini I. Kaplan",
        "Emily J. Braun",
        "Nishaat Mukadam",
        "Sudan Duwadi",
        "Laura Carlton",
        "David Beeler",
        "Lindsay K. Butler",
        "Erin Carpenter",
        "Jaimie Girnis",
        "John Wilson",
        "Vaibhav Tripathi",
        "Yiwen Zhang",
        "Bettina Sorger",
        "Alexander von Lühmann",
        "David C. Somers",
        "Alice Cronin-Golomb",
        "Swathi Kiran",
        "Terry D. Ellis",
        "David A. Boas",
    ],
    institution="Boston University",
    country="US",
    repository="OpenNeuro",
    data_url=f"https://doi.org/{DATA_DOI}",
    # dataset_description.json License. Its DataLicense is PDDL.
    license="CC0",
    publication_year=2025,
)
EXTERNAL_LINKS = {"code": "https://github.com/mayucel/InclusionStudy"}

log = logging.getLogger(__name__)


class _Yucel2025Fnirs(BaseBIDSFNIRSDataset):
    """Shared loader for simultaneous LH and RH device recordings.

    Subclasses set ``_code`` and ``_task_runs``, the BIDS ``(task, run)``
    pairs recorded by each device. Runs of both devices are aligned and
    stacked with :func:`combine_devices`.
    """

    _code = ""
    _task_runs: dict[str, list[tuple[str, str]]] = {}  # noqa: RUF012

    def __init__(self, subjects, sessions, devices, **kwargs):
        if not devices or not set(devices) <= set(_DEVICES):
            raise ValueError(f"devices must be a subset of {list(_DEVICES)}")
        self.devices = tuple(sorted(set(devices)))
        super().__init__(
            subjects=_SUBJECTS,
            sessions_per_subject=1,
            code=self._device_code(),
            doi=DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=True,
            **kwargs,
        )

    def _device_code(self) -> str:
        """Dataset code for the selected devices, e.g. ``Yucel2025FnirsSqueezing-lh-rh``."""
        if self.devices == ("RH",):
            return self._code
        return "-".join([self._code, *(d.lower() for d in self.devices)])

    def _selected_task_runs(self):
        return [tr for d in self.devices for tr in self._task_runs[d]]

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        del update_path, verbose
        return str(
            download_openneuro_dataset_files(
                _OPENNEURO_ID,
                Yucel2025FnirsSqueezing._code,
                path,
                self._openneuro_file_paths(subject),
                force_update,
            )
        )

    def _openneuro_file_paths(self, subject):
        bids_subject = self._bids_subject(subject)
        subject_base = f"sub-{bids_subject}"
        return [
            "dataset_description.json",
            f"{subject_base}/{subject_base}_scans.tsv",
            *build_bids_fnirs_file_paths(
                bids_subject, None, self._selected_task_runs()
            ),
        ]

    def _get_path_search_params(self, subject):
        return {
            **super()._get_path_search_params(subject),
            "tasks": [task for task, _ in self._selected_task_runs()],
        }

    def _get_single_subject_data(self, subject):
        device_of_task = {
            task: d for d in self.devices for task, _ in self._task_runs[d]
        }
        runs = {}
        for bids_path in self.bids_paths(subject):
            raw = deduplicate_annotations(self._load_raw(bids_path, subject))
            runs.setdefault(bids_path.run, {})[device_of_task[bids_path.task]] = raw
        return {
            "0": {
                run: self._standardize_raw(self._combine(raws))
                for run, raws in sorted(runs.items())
                if self._can_combine(subject, run, raws)
            }
        }

    def _can_combine(self, subject, run, raws):
        """Whether the device recordings of one run can be aligned."""
        unmarked = [d for d, raw in raws.items() if not len(raw.annotations)]
        if len(raws) < 2 or not unmarked:
            return True
        log.warning(
            "Skipping subject %s run %s: %s recording has no events to align on",
            subject,
            run,
            "/".join(unmarked),
        )
        return False

    def _combine(self, raws):
        """Stack the selected device recordings of one run."""
        if self.devices == _DEVICES:
            return combine_devices(raws["RH"], raws["LH"])
        return raws[self.devices[0]]


class Yucel2025FnirsSqueezing(_Yucel2025Fnirs):
    """Yucel 2025 bilateral ball-squeezing fNIRS dataset.

    Left- or right-hand ball squeezing for 5 s at a self-selected rate, cued by
    written words with an audio signal, with 5-15 s rest. Two devices recorded
    the left and right hemispheres simultaneously [1]_.

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load. All when None.
    sessions : list of str | None
        Sessions to load. All when None.
    devices : tuple of {"LH", "RH"}
        Device recordings to load, named by the hemisphere they cover. With
        both, LH is aligned to RH and its sources and detectors are renumbered
        after the RH ones.

    Notes
    -----
    Subjects 253, 270 and 274 lack one right-hand event. With both devices,
    subjects 249, 252, 270, 274, 291, 370 and 394 lose the trials after their
    LH recording stops.

    References
    ----------
    .. [1] Yücel, M. A., Anderson, J. E., Rogers, D., et al. (2025). Quantifying
           the impact of hair and skin characteristics on fNIRS signal quality
           for enhanced inclusivity. Nature Human Behaviour, 9, 2651-2668.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm="fnirsMotorExecution",
            task_type="hand ball squeezing",
            events=EVENTS,
            n_classes=len(EVENTS),
            class_labels=list(EVENTS),
            # BIDS events store 10 s durations.
            trial_duration=5.0,
            stimulus_type="written words with audio signal",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=30,
            n_trials_per_class={label: 15 for label in EVENTS},
            trials_context="per subject",
        ),
        file_format="BIDS SNIRF",
        external_links=EXTERNAL_LINKS,
    )
    _code = "Yucel2025FnirsSqueezing"
    _task_runs = {  # noqa: RUF012
        "LH": [("009BallLH", "1")],
        "RH": [("010BallRH", "1")],
    }
    _annotation_rename_map = {"2": "left", "3": "right"}  # noqa: RUF012

    def __init__(self, subjects=None, sessions=None, devices=("RH",)):
        super().__init__(
            subjects,
            sessions,
            devices,
            events=EVENTS,
            interval=INTERVAL,
            paradigm="fnirsMotorExecution",
        )


class Yucel2025FnirsRest(_Yucel2025Fnirs):
    """Yucel 2025 resting-state fNIRS dataset.

    Two 3-min resting-state runs, after fast and after proper capping, with
    two devices recording the left and right hemispheres simultaneously [1]_.

    Parameters
    ----------
    subjects : list of int | None
        Subjects to load. All when None.
    sessions : list of str | None
        Sessions to load. All when None.
    devices : tuple of {"LH", "RH"}
        Device recordings to load, named by the hemisphere they cover. With
        both, LH is aligned to RH and its sources and detectors are renumbered
        after the RH ones.

    Notes
    -----
    With both devices, subject 236 run 2 is skipped because neither recording
    has events to align the devices on.

    References
    ----------
    .. [1] Yücel, M. A., Anderson, J. E., Rogers, D., et al. (2025). Quantifying
           the impact of hair and skin characteristics on fNIRS signal quality
           for enhanced inclusivity. Nature Human Behaviour, 9, 2651-2668.
    """

    METADATA = DatasetMetadata(
        acquisition=ACQUISITION,
        participants=PARTICIPANTS,
        experiment=ExperimentMetadata(
            paradigm=REST_PARADIGM,
            task_type="resting state",
            events=REST_EVENTS,
            n_classes=len(REST_EVENTS),
            class_labels=list(REST_EVENTS),
            trial_duration=180.0,
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        # Most runs mark the start and end of the 3-min rest with a "rest" event.
        data_structure=DataStructureMetadata(
            n_trials=2 * REST_RUNS,
            n_trials_per_class={label: 2 * REST_RUNS for label in REST_EVENTS},
            trials_context="per subject",
        ),
        runs_per_session=REST_RUNS,
        file_format="BIDS SNIRF",
        external_links=EXTERNAL_LINKS,
    )
    _code = "Yucel2025FnirsRest"
    _task_runs = {  # noqa: RUF012
        "LH": [("001RestingLH", "1"), ("003RestingLH", "2")],
        "RH": [("002RestingRH", "1"), ("004RestingRH", "2")],
    }
    _annotation_rename_map = {"1": "rest"}  # noqa: RUF012

    def __init__(self, subjects=None, sessions=None, devices=("RH",)):
        super().__init__(
            subjects,
            sessions,
            devices,
            events=REST_EVENTS,
            interval=None,
            paradigm=REST_PARADIGM,
        )
