"""Base class for BIDS fNIRS datasets.

Loads SNIRF files directly rather than via mne_bids.read_raw_bids, because
the SNIRF file is the authoritative source for fNIRS channel metadata.
BIDS channels.tsv often has naming mismatches with the SNIRF, and read_raw_bids
raises on mismatch by default. Using read_raw_snirf avoids this while still
leveraging BaseBIDSDataset.bids_paths() for BIDS file discovery.
"""

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path

import mne
import pandas as pd
from moabb.datasets.base import BaseBIDSDataset
from moabb.datasets.download import get_dataset_path

from .base import BaseFNIRSDataset

log = logging.getLogger(__name__)


def get_bids_cache_root(dataset_code, path=None):
    """Return the MOABB-managed BIDS dataset cache root."""
    return (
        Path(get_dataset_path(dataset_code, path)) / f"MNE-{dataset_code.lower()}-data"
    )


def build_bids_fnirs_file_paths(
    bids_subject: str,
    session: str | None,
    task_runs: Iterable[tuple[str, str | None]],
    include_sessions: bool = False,
) -> list[str]:
    """Return dataset-relative BIDS paths required for fNIRS recordings.

    Entity values are unprefixed and used verbatim. Session and run may be
    absent.
    """
    bids_sub_str = f"sub-{bids_subject}"
    session_base = bids_sub_str
    recording_base = bids_sub_str
    if session is not None:
        session_base += f"/ses-{session}"
        recording_base += f"_ses-{session}"
    nirs_base = f"{session_base}/nirs/{recording_base}"
    files = [
        f"{nirs_base}_coordsystem.json",
        f"{nirs_base}_optodes.json",
        f"{nirs_base}_optodes.tsv",
    ]
    if include_sessions:
        if session is None:
            raise ValueError("include_sessions requires a session")
        files.insert(0, f"{bids_sub_str}/{bids_sub_str}_sessions.tsv")
    for task, run in task_runs:
        run_base = f"{nirs_base}_task-{task}"
        if run is not None:
            run_base += f"_run-{run}"
        files.extend(
            f"{run_base}_{suffix}"
            for suffix in (
                "channels.tsv",
                "events.json",
                "events.tsv",
                "nirs.json",
                "nirs.snirf",
            )
        )
    return files


class BaseBIDSFNIRSDataset(BaseFNIRSDataset, BaseBIDSDataset):
    """BIDS dataset with fNIRS preprocessing (OD -> TDDR -> Beer-Lambert).

    Uses mne.io.read_raw_snirf instead of mne_bids.read_raw_bids because the
    SNIRF file is authoritative for fNIRS channel metadata. BIDS channels.tsv
    frequently has naming mismatches that cause read_raw_bids to raise.
    """

    _bids_task = None
    _bids_session = None
    _bids_subject_format = "{}"

    def _bids_subject(self, subject):
        """Convert a MOABB subject ID to its BIDS entity value."""
        return self._bids_subject_format.format(subject)

    def _bids_subject_str(self, subject):
        """Convert a MOABB subject ID to its prefixed BIDS subject string."""
        return f"sub-{self._bids_subject(subject)}"

    def _get_path_search_params(self, subject):
        """Return filters for this dataset's SNIRF recordings."""
        out = {"extensions": [".snirf"]}
        if self._bids_task is not None:
            out["tasks"] = self._bids_task
        if self._bids_session is not None:
            out["sessions"] = self._bids_session
        if subject is not None:
            out["subjects"] = self._bids_subject(subject)
        return out

    def _get_single_subject_data(self, subject):
        data = {}
        for bids_path in self.bids_paths(subject):
            raw = self._load_raw(bids_path, subject)
            raw = self._standardize_raw(raw)
            session = bids_path.session or "0"
            run = bids_path.run or "0"
            data.setdefault(session, {})[run] = raw
        return data

    def _load_raw(self, bids_path, subject):
        """Load one BIDS recording."""
        with warnings.catch_warnings():
            # A loader may still replace 2D file positions with a 3D montage, so
            # MNE cannot judge them yet. warn_if_optodes_are_planar checks the
            # loaded recording once the montage is settled.
            warnings.filterwarnings(
                "ignore", message=".*only contains 2D location information.*"
            )
            return mne.io.read_raw_snirf(bids_path.fpath, preload=True, verbose=False)

    @staticmethod
    def _read_bids_events(bids_path):
        """Read the events sidecar for a BIDS recording."""
        events_path = bids_path.copy().update(suffix="events", extension=".tsv").fpath
        events = pd.read_csv(events_path, sep="\t")
        if "trial_type" in events:
            events["trial_type"] = events["trial_type"].astype(str)
        return events

    def _events_with_full_epoch(self, events, raw):
        """Return the events whose epoch interval fits inside the recording."""
        start = events["onset"] + self.interval[0]
        stop = events["onset"] + self.interval[1]
        return events[(start >= 0) & (stop <= raw.times[-1])]
