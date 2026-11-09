"""Dataset loading utilities."""

import logging
import math
import re
import tarfile
import zipfile
from pathlib import Path

import mne
import numpy as np
from filelock import BaseFileLock, FileLock
from moabb.datasets.utils import (
    extract_rar,
    safe_extract_tar,
    safe_extract_zip,
)
from scipy.io import loadmat

logger = logging.getLogger(__name__)

_OPTODE_PATTERN = re.compile(r"S(\d+)_D(\d+)")


def extract_archive(archive_path, destination):
    """Extract a ZIP, TAR or RAR archive, then delete it.

    The archive is kept when extraction fails.

    Parameters
    ----------
    archive_path : path-like
        ZIP, RAR or optionally compressed TAR archive. The format is read from
        the file content, not its name. RAR needs ``unrar``, ``unar`` or ``7z``.
    destination : path-like
        Directory to extract into.
    """
    archive_path = Path(archive_path)
    with archive_path.open("rb") as handle:
        is_rar = handle.read(4) == b"Rar!"
    if is_rar:
        extract_rar(archive_path, destination)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            safe_extract_zip(archive, destination)
    else:
        with tarfile.open(archive_path, mode="r:*") as archive:
            safe_extract_tar(archive, destination)
    archive_path.unlink()


def download_lock(directory) -> BaseFileLock:
    """Return an inter-process lock for downloading into a directory.

    Hold it around the already-extracted check, the download and the extraction
    when several datasets or processes share one download directory, so each
    archive is extracted once.

    Parameters
    ----------
    directory : path-like
        Download or extraction directory. The lock file is ``<directory>.lock``
        next to it.
    """
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(directory.parent / f"{directory.name}.lock")


def paired_fnirs_channel_order(ch_names):
    """Return fNIRS channel names with wavelengths adjacent for each pair.

    Examples
    --------
    >>> paired_fnirs_channel_order(
    ...     ["S1_D1 760", "S2_D1 760", "S1_D1 850", "S2_D1 850"]
    ... )
    ['S1_D1 760', 'S1_D1 850', 'S2_D1 760', 'S2_D1 850']
    """
    pairs = list(dict.fromkeys(name.split()[0] for name in ch_names))
    wavelengths = list(dict.fromkeys(name.split()[1] for name in ch_names))
    return [f"{pair} {wavelength}" for pair in pairs for wavelength in wavelengths]


def homer_nirs_intensity(source):
    """Return the continuous-wave intensity channels of a Homer ``.nirs`` file.

    Parameters
    ----------
    source : dict
        The file loaded with ``scipy.io.loadmat(..., squeeze_me=True,
        struct_as_record=False)``.

    Returns
    -------
    data : ndarray, shape (n_channels, n_times)
        Intensity with the wavelengths of each source-detector pair adjacent.
    names : list of str
        ``"S<source>_D<detector> <wavelength>"`` channel names, pairs in
        ``SD.MeasList`` order.
    """
    data = np.asarray(source["d"], dtype=float)
    sd = source["SD"]
    # MeasList rows are (source, detector, unused, wavelength index from 1).
    measurements = np.asarray(sd.MeasList, dtype=int)
    wavelengths = np.atleast_1d(np.asarray(sd.Lambda, dtype=int))
    if data.ndim != 2 or data.shape[1] != len(measurements):
        raise ValueError(
            f"Intensity shape {data.shape} does not match MeasList {measurements.shape}"
        )
    pairs = [(src, det) for src, det, _, wave in measurements if wave == 1]
    if len(pairs) * len(wavelengths) != len(measurements):
        raise ValueError("Source-detector pairs differ between wavelengths")
    index = {(src, det, wave): i for i, (src, det, _, wave) in enumerate(measurements)}
    order = [
        index[(src, det, wave)]
        for src, det in pairs
        for wave in range(1, len(wavelengths) + 1)
    ]
    names = [f"S{src}_D{det} {wl}" for src, det in pairs for wl in wavelengths]
    return data.T[order], names


def concentration_length_to_molar(data, distance_cm, ppf=6.0):
    """Convert mM cm concentration-length values to M.

    Parameters
    ----------
    data : array-like
        Concentration changes multiplied by the unknown optical path length.
    distance_cm : float
        Source-detector distance in cm.
    ppf : float
        Partial pathlength factor. The default is MNE's ``beer_lambert_law``
        default.
    """
    return np.asarray(data) / (distance_cm * ppf) * 1e-3


def merge_repeated_annotations(annotations, max_interval):
    """Merge consecutive annotations that repeat one description.

    Use this when a source marks every stimulus of a block and one event per
    block is wanted.

    Parameters
    ----------
    annotations : mne.Annotations
        Annotations sorted by onset.
    max_interval : float
        Largest onset interval (s) between an annotation and the previous one
        for the two to merge. Both must have the same description.

    Returns
    -------
    merged : mne.Annotations
        One annotation per merged group. It starts at the group's first onset
        and ends where the group's last annotation ends.
    """
    groups = []
    for annotation in annotations:
        previous = groups[-1][-1] if groups else None
        if (
            previous is not None
            and annotation["description"] == previous["description"]
            and annotation["onset"] - previous["onset"] <= max_interval
        ):
            groups[-1].append(annotation)
        else:
            groups.append([annotation])
    return mne.Annotations(
        [group[0]["onset"] for group in groups],
        [g[-1]["onset"] + g[-1]["duration"] - g[0]["onset"] for g in groups],
        [group[0]["description"] for group in groups],
        orig_time=annotations.orig_time,
    )


def deduplicate_annotations(raw):
    """Remove annotations with identical onset, duration, and description."""
    seen = set()
    duplicates = []
    for index, annotation in enumerate(raw.annotations):
        key = (
            annotation["onset"],
            annotation["duration"],
            annotation["description"],
        )
        if key in seen:
            duplicates.append(index)
        seen.add(key)
    raw.annotations.delete(duplicates)
    return raw


def override_raw_sfreq(raw, sfreq):
    """Correct sampling-rate metadata without resampling the signal."""
    # Scale onset seconds inversely to preserve each annotation's sample index.
    factor = sfreq / raw.info["sfreq"]
    with raw.info._unlock():
        raw.info["sfreq"] = sfreq
    raw.annotations.onset /= factor
    return raw


def snap_raw_sfreq(raw, step, rtol):
    """Snap a timestamp-derived rate to a nearby nominal frequency grid."""
    sfreq = raw.info["sfreq"]
    nominal_sfreq = round(sfreq / step) * step
    if math.isclose(sfreq, nominal_sfreq, rel_tol=rtol):
        return override_raw_sfreq(raw, nominal_sfreq)
    return raw


def nirx_head_model_montage(probe_info_path):
    """Read the optode montage of a NIRx probe file that uses the NIRx head model.

    The positions are in the frame of the head model, not MNI. The montage
    carries the head model's fiducials, so ``set_montage`` moves the optodes
    to the head frame.

    Parameters
    ----------
    probe_info_path : Path
        ``*_probeInfo.mat`` file with a ``geom.NIRxHead`` head model.

    Returns
    -------
    mne.channels.DigMontage
        Optodes ``S<n>`` and ``D<n>`` with nasion, LPA and RPA.
    """
    info = loadmat(probe_info_path, squeeze_me=True, struct_as_record=False)[
        "probeInfo"
    ]
    # Rows are Nz, LPA, RPA, then other landmarks (nirsLAB manual, Appendix B).
    # Columns are an index and x, y, z in cm.
    nasion, lpa, rpa = info.geom.NIRxHead.mesh.fiducials[:3, 1:] / 100
    sources = info.probes.coords_s3 / 100
    detectors = info.probes.coords_d3 / 100
    ch_pos = {f"S{index}": pos for index, pos in enumerate(sources, start=1)}
    ch_pos |= {f"D{index}": pos for index, pos in enumerate(detectors, start=1)}
    return mne.channels.make_dig_montage(
        ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, coord_frame="mri"
    )


def warn_if_optodes_are_planar(raw):
    """Warn when all optodes sit at z = 0, which means the positions are 2D.

    Distances taken from planar positions ignore head curvature and therefore
    under-estimate the source-detector separation. Positions that are missing
    entirely are left to ``beer_lambert_law``.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to check. Not modified.

    Returns
    -------
    raw : mne.io.BaseRaw
        The input recording.
    """
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    if len(picks) == 0:
        return raw
    # loc[5] and loc[8] are the source and detector heights.
    heights = np.array([raw.info["chs"][pick]["loc"][[5, 8]] for pick in picks])
    known = heights[np.isfinite(heights)]
    if known.size and not known.any():
        logger.warning(
            "Optode positions are two-dimensional; source-detector distances "
            "ignore head curvature"
        )
    return raw


def _ensure_stim_channel(raw, event_id):
    if "stim" in raw.get_channel_types():
        return raw
    stim = np.zeros((1, raw.n_times))
    if raw.annotations:
        # Annotations outside event_id are left out of the stim channel.
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
        samples = events[:, 0] - raw.first_samp
        unique, counts = np.unique(samples, return_counts=True)
        if (counts > 1).any():
            logger.warning(
                "Events share onset samples %s; the stim channel keeps one code each",
                unique[counts > 1].tolist(),
            )
        stim[0, samples] = events[:, 2]
    stim_raw = mne.io.RawArray(
        stim, mne.create_info(["Stim"], raw.info["sfreq"], ["stim"]), verbose=False
    )
    raw.add_channels([stim_raw], force_update_info=True)
    return raw


def _rename_annotations(raw, annotation_rename_map):
    if annotation_rename_map is None:
        return
    present = set(raw.annotations.description)
    mapping = {
        key: value for key, value in annotation_rename_map.items() if key in present
    }
    if mapping:
        raw.annotations.rename(mapping)


def _annotations_after_end(raw: mne.io.BaseRaw) -> np.ndarray:
    """Indices of annotations that start after the last sample."""
    onsets = raw.annotations.onset - raw.first_time
    return np.flatnonzero(onsets > raw.times[-1])


def standardize_raw(raw, event_id, annotation_rename_map=None):
    """Standardize unprocessed raw to shared format."""
    _rename_annotations(raw, annotation_rename_map)
    late = _annotations_after_end(raw)
    if len(late):
        logger.warning("Dropping %d annotation(s) after the last sample", len(late))
        raw.annotations.delete(late)
    return _ensure_stim_channel(raw, event_id)


def _max_optodes(raw: mne.io.BaseRaw) -> tuple[int, int]:
    """Largest source and detector numbers in the channel names."""
    pairs = [tuple(map(int, _OPTODE_PATTERN.match(n).groups())) for n in raw.ch_names]
    return max(s for s, _ in pairs), max(d for _, d in pairs)


def _renumber_optodes(
    raw: mne.io.BaseRaw, source_offset: int, detector_offset: int
) -> None:
    """Shift source and detector numbers in the channel names."""
    raw.rename_channels(
        lambda name: _OPTODE_PATTERN.sub(
            lambda m: f"S{int(m[1]) + source_offset}_D{int(m[2]) + detector_offset}",
            name,
            count=1,
        )
    )


def _crop_samples(raw: mne.io.BaseRaw, start: int, n_times: int) -> None:
    """Keep ``n_times`` samples from sample ``start`` and the events starting in them."""
    raw.crop(tmin=raw.times[start], tmax=raw.times[start + n_times - 1])
    raw.annotations.delete(_annotations_after_end(raw))


def _check_matching_events(
    reference: mne.io.BaseRaw, other: mne.io.BaseRaw, tolerance: float
) -> None:
    """Raise if the aligned annotations of two recordings differ."""
    reference_onsets = reference.annotations.onset - reference.first_time
    other_onsets = other.annotations.onset - other.first_time
    if (
        reference.annotations.description.tolist()
        != other.annotations.description.tolist()
        or np.abs(reference_onsets - other_onsets).max() > tolerance
    ):
        raise ValueError("Device events do not match after alignment")


def combine_devices(
    reference: mne.io.BaseRaw, other: mne.io.BaseRaw, tolerance: float = 0.2
) -> mne.io.BaseRaw:
    """Align simultaneous recordings of two devices and stack their channels.

    ``other`` is shifted so its first event matches the first event of
    ``reference``, and both are cropped to their overlap. Sources and detectors
    of ``other`` are renumbered after the largest ``reference`` numbers.

    Parameters
    ----------
    reference, other : mne.io.BaseRaw
        fNIRS recordings with the same sampling rate and event sequence.
        Modified in place. Both need at least one event.
    tolerance : float
        Maximum onset difference, in seconds, between matching aligned events.

    Returns
    -------
    mne.io.BaseRaw
        ``reference`` channels followed by ``other`` channels, with
        ``reference`` annotations.
    """
    deduplicate_annotations(reference)
    deduplicate_annotations(other)
    if not len(reference.annotations) or not len(other.annotations):
        raise ValueError("A device recording has no events to align on")
    shift = round(
        (reference.annotations.onset[0] - reference.first_time)
        * reference.info["sfreq"]
        - (other.annotations.onset[0] - other.first_time) * other.info["sfreq"]
    )
    reference_start, other_start = max(shift, 0), max(-shift, 0)
    n_times = min(reference.n_times - reference_start, other.n_times - other_start)
    _crop_samples(reference, reference_start, n_times)
    _crop_samples(other, other_start, n_times)
    _check_matching_events(reference, other, tolerance)
    _renumber_optodes(other, *_max_optodes(reference))
    other.set_annotations(None)
    return reference.add_channels([other], force_update_info=True)
