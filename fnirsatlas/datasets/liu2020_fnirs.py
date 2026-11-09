"""Liu 2020 prefrontal hand-motion fNIRS dataset."""

import shutil
import subprocess
from pathlib import Path
from typing import ClassVar

import numpy as np
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
from .preprocessing import interpolate_zero
from .utils import download_lock, extract_archive

URL = "https://data.mendeley.com/public-api/zip/jhv73gj2d2/download/1"
DOWNLOAD_CODE = "Liu2020Fnirs"
SFREQ = 1 / 0.12288
N_PAIRS = 204
WAVELENGTHS = (780, 850)
# Selected onsets are at least 60.088 s apart. 14.99 s avoids rounding above
# the last available sample at this sampling frequency.
INTERVAL = [-2, 14.99]
# No paper describes this dataset, so the data DOI is used.
DOI = "10.17632/jhv73gj2d2.1"
PARADIGM = "fnirsMotorExecution"
# Control subject 10 is absent from the release.
CONTROL_SUBJECTS = list(range(1, 10))
STROKE_SUBJECTS = list(range(1, 11))
CONTROL_EVENTS = {"left hand": 1, "right hand": 2}
STROKE_EVENTS = {"affected hand": 1, "unaffected hand": 2}
ACQUISITION = FnirsAcquisitionMetadata(
    sampling_rate=SFREQ,
    channel_types={"fnirs_cw_amplitude": 2 * N_PAIRS},
    # Neither the record nor the CSV headers name the device.
    hardware="unknown",
    # Recorded in China.
    line_freq=50.0,
    montage="custom",
    # Column headers of the CSV files.
    wavelengths_nm=[float(w) for w in WAVELENGTHS],
    # No optode geometry is released, so all pairs use the fallback distance.
    n_short_channels=0,
    sensor_positions="none",
    # Accel_x/y/z and Gyro_x/y/z columns of the CSV files.
    other_modalities=["accelerometer", "gyroscope"],
)
DOCUMENTATION = DocumentationMetadata(
    doi=DOI,
    investigators=["Shixian Liu"],
    institution="Qingdao University of Science and Technology",
    country="CN",
    repository="Mendeley Data",
    data_url=f"https://doi.org/{DOI}",
    license="CC BY 4.0",
    publication_year=2020,
)


def _metadata(subjects, events, n_trials_per_class):
    """METADATA of one cohort."""
    return DatasetMetadata(
        acquisition=ACQUISITION,
        participants=ParticipantMetadata(n_subjects=len(subjects)),
        experiment=ExperimentMetadata(
            paradigm=PARADIGM,
            # Task blocks last 44.9 to 166.7 s between source onset and offset
            # markers, so no single trial duration is given.
            task_type="hand motion",
            events=events,
            n_classes=len(events),
            class_labels=list(events),
            mode="offline",
        ),
        documentation=DOCUMENTATION,
        data_structure=DataStructureMetadata(
            n_trials=n_trials_per_class * len(events),
            n_trials_per_class={label: n_trials_per_class for label in events},
            trials_context="per subject",
        ),
        file_format="CSV",
    )


class _Liu2020Fnirs(BaseFNIRSDataset):
    # No optode positions are released, so bad channels cannot be interpolated
    # by distance.
    preprocessing_options: ClassVar = {"interpolate_bads": interpolate_zero}
    cohort = ""
    SUBJECTS: ClassVar[list[int]] = []
    EVENTS: ClassVar[dict[str, int]] = {}

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=self.SUBJECTS,
            sessions_per_subject=1,
            events=self.EVENTS,
            code=self.code,
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
        """Return the extracted directory containing a subject's CSV files."""
        del update_path
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")
        return str(self._download_data(path, force_update, verbose))

    def _get_single_subject_data(self, subject):
        sessions = {}
        for run, condition in enumerate(self.EVENTS):
            source = self._subject_file(
                Path(self.data_path(subject)), subject, condition
            )
            raw = self._read_raw(source, self.event_id[condition])
            self._add_run(sessions, "0fnirs", raw, str(run))
        return sessions

    def _read_raw(self, source, event_code):
        # Each CSV starts with device metadata. The numeric table begins after
        # the row whose first column is "Time".
        skip_header = self._header_line(source) + 1
        values = np.genfromtxt(source, delimiter=",", skip_header=skip_header)
        if values.ndim != 2 or values.shape[1] < 2 + 2 * N_PAIRS:
            raise ValueError(f"Unexpected Liu2020 CSV shape {values.shape}: {source}")

        # Start from first non-zero row.
        values = self._drop_startup_samples(values)

        # Skip Time/Marker, select the 204 pairs at two wavelengths, and
        # transpose CSV (samples, channels) into MNE (channels, samples).
        data = values[:, 2 : 2 + 2 * N_PAIRS].T

        # The CSV only identifies CH1...CH204. MNE requires each wavelength
        # channel to use "S#_D# wavelength" syntax, so assign one synthetic
        # source-detector pair per exported channel. These are identifiers, not
        # anatomical geometry; fallback distances are populated below.
        names = [
            f"S{pair}_D{pair} {wavelength}"
            for pair in range(1, N_PAIRS + 1)
            for wavelength in WAVELENGTHS
        ]
        raw = self._make_raw(data, names, ["fnirs_cw_amplitude"] * len(names), SFREQ)
        self._populate_fnirs_metadata(raw)
        marker = values[:, 1]
        samples = np.flatnonzero(marker == 1)
        raw.add_channels(
            [
                self._make_stim_raw(
                    raw.n_times, SFREQ, samples, [event_code] * len(samples)
                )
            ]
        )
        return self._standardize_raw(raw)

    @staticmethod
    def _drop_startup_samples(values):
        signal = values[:, 2 : 2 + 2 * N_PAIRS]
        first_signal = np.flatnonzero(np.any(signal != 0, axis=1))[0]
        return values[first_signal:]

    @staticmethod
    def _header_line(source):
        with source.open(errors="replace") as handle:
            for index, line in enumerate(handle):
                if line.startswith("Time,"):
                    return index
        raise ValueError(f"Time header not found in {source}")

    def _subject_file(self, data_dir, subject, condition):
        prefix = f"no{subject}.{self.cohort}-{condition.replace(' ', '*')}"
        matches = sorted(data_dir.rglob(f"{prefix}*.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {prefix} CSV, found {len(matches)}")
        return matches[0]

    def _download_data(self, path, force_update, verbose):
        base = (
            Path(dl.get_dataset_path(DOWNLOAD_CODE, path))
            / f"MNE-{DOWNLOAD_CODE.lower()}-data"
        )
        marker = base / ".extracted"
        # The stroke and control cohorts share one archive.
        with download_lock(base):
            if marker.is_file() and not force_update:
                return base
            archive = Path(dl.data_dl(URL, DOWNLOAD_CODE, path, force_update, verbose))
            parts = base / "parts"
            parts.mkdir(parents=True, exist_ok=True)
            extract_archive(archive, parts)
            self._extract_split_archive(parts, base)
            if len(list(base.rglob("*.csv"))) != 38:
                raise RuntimeError(
                    "Liu2020 extraction did not produce all 38 CSV files"
                )
            shutil.rmtree(parts)
            marker.touch()
        return base

    @staticmethod
    def _extract_split_archive(parts, destination):
        for source in parts.glob("*.z??"):
            source.rename(parts / f"data{source.suffix}")
        executable = shutil.which("7z") or shutil.which("7zz")
        if executable is None:
            raise RuntimeError("Liu2020 extraction requires the 7z executable")
        result = subprocess.run(
            [executable, "x", str(parts / "data.zip"), f"-o{destination}", "-y"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Liu2020 extraction failed: {result.stderr.strip()}")


class Liu2020FnirsControl(_Liu2020Fnirs):
    """Liu 2020 prefrontal hand-motion fNIRS dataset, control cohort.

    Prefrontal recordings while healthy controls move their left or right hand
    in blocks, one recording per hand [1]_.

    Notes
    -----
    The release has no optode geometry or source-detector map, so pair names
    are synthetic and spatial analyses are unsupported. Leading all-zero
    startup samples are dropped. Controls and patients are separate datasets
    because their class meanings differ. Control subject 10 is absent from the
    release.

    References
    ----------
    .. [1] Liu, S. (2020). fNIRS DATA. Mendeley Data, V1.
           https://doi.org/10.17632/jhv73gj2d2.1
    """

    # Subjects 1 and 4 have an extra left-hand block.
    METADATA = _metadata(CONTROL_SUBJECTS, CONTROL_EVENTS, n_trials_per_class=5)
    code = "Liu2020FnirsControl"
    cohort = "normal subjects"
    SUBJECTS = CONTROL_SUBJECTS
    EVENTS = CONTROL_EVENTS


class Liu2020FnirsStroke(_Liu2020Fnirs):
    """Liu 2020 prefrontal hand-motion fNIRS dataset, stroke cohort.

    Prefrontal recordings while stroke patients move their affected or
    unaffected hand in blocks, one recording per hand [1]_.

    Notes
    -----
    The release has no optode geometry or source-detector map, so pair names
    are synthetic and spatial analyses are unsupported. Leading all-zero
    startup samples are dropped. Controls and patients are separate datasets
    because their class meanings differ.

    References
    ----------
    .. [1] Liu, S. (2020). fNIRS DATA. Mendeley Data, V1.
           https://doi.org/10.17632/jhv73gj2d2.1
    """

    METADATA = _metadata(STROKE_SUBJECTS, STROKE_EVENTS, n_trials_per_class=6)
    code = "Liu2020FnirsStroke"
    cohort = "stroke patients"
    SUBJECTS = STROKE_SUBJECTS
    EVENTS = STROKE_EVENTS
