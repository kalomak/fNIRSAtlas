"""Shared fNIRS preprocessing operations."""

import functools
import itertools
import logging
import warnings
from collections.abc import Callable
from operator import methodcaller

import mne
import numpy as np
from moabb.datasets.preprocessing import NamedFunctionTransformer
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve, periodogram
from sklearn.pipeline import make_pipeline

logger = logging.getLogger(__name__)


def mark_bad_segments_by_invalid_values(raw):
    """Annotate contiguous spans of invalid fNIRS samples as ``BAD_invalid``.

    Non-finite samples are invalid, and for continuous-wave amplitude also
    samples at or below zero, whose optical density is undefined.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to annotate. Modified in place.

    Returns
    -------
    raw : mne.io.BaseRaw
        Input recording with invalid spans annotated per channel.
    """
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    data = raw.get_data(picks=picks)
    mask = ~np.isfinite(data)
    cw = np.array(raw.get_channel_types(picks=picks)) == "fnirs_cw_amplitude"
    with np.errstate(invalid="ignore"):
        mask[cw] |= data[cw] <= 0
    edges = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rows, starts = np.nonzero(edges == 1)
    _, stops = np.nonzero(edges == -1)
    channels = [[raw.ch_names[picks[row]]] for row in rows]
    _annotate_bad_samples(raw, starts, stops, channels, "BAD_invalid")
    return raw


def mark_bad_segments_by_flat(raw, threshold=None, window=10.0):
    """Annotate constant CW windows as ``BAD_flat``, e.g. a disconnected optode.

    A window is flat when its standard deviation divided by the absolute mean
    is below ``threshold``. Flat windows pass the SNR check, whose SNR grows as
    the standard deviation falls. Non-finite windows are left to
    `mark_bad_segments_by_invalid_values`. The remainder after the last full
    window is not checked.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording containing continuous-wave amplitude channels. Modified in place.
    threshold : float | None
        Largest relative standard deviation of a flat window. ``None``
        disables the check.
    window : float
        Window length in seconds.

    Returns
    -------
    raw : mne.io.BaseRaw
        Input recording with flat windows annotated as ``BAD_flat``.
    """
    if threshold is None or "fnirs_cw_amplitude" not in raw.get_channel_types():
        return raw
    picks = mne.pick_types(raw.info, meg=False, fnirs="fnirs_cw_amplitude", exclude=[])
    data, length = _windows(raw.get_data(picks=picks), raw.info["sfreq"], window)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_std = data.std(axis=2) / np.abs(data.mean(axis=2))
    rows, indices = np.nonzero(relative_std < threshold)
    channels = [[raw.ch_names[picks[row]]] for row in rows]
    _annotate_bad_samples(
        raw, indices * length, (indices + 1) * length, channels, "BAD_flat"
    )
    return raw


def mark_bad_segments_by_snr(raw, threshold=None, window=10.0):
    """Annotate CW windows failing the amplitude SNR threshold as ``BAD_snr``.

    SNR is the amplitude mean divided by its standard deviation within a
    window. Zero variance and nonfinite SNR fail the check. The remainder
    after the last full window is not checked.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording containing continuous-wave amplitude channels. Modified in place.
    threshold : float | None
        Required SNR as a ratio. ``None`` disables rejection. In decibels the
        ratio is ``20 * log10(threshold)``, so ``10`` is the 20 dB of [1]_.
    window : float
        Window length in seconds.

    Returns
    -------
    raw : mne.io.BaseRaw
        Input recording with failing windows annotated as ``BAD_snr``.

    References
    ----------
    .. [1] Yücel, M. A. et al. (2021). Best practices for fNIRS publications.
       Neurophotonics, 8(1), 012101. https://doi.org/10.1117/1.NPh.8.1.012101
    """
    if threshold is None or "fnirs_cw_amplitude" not in raw.get_channel_types():
        return raw
    picks = mne.pick_types(raw.info, meg=False, fnirs="fnirs_cw_amplitude", exclude=[])
    snr, length = windowed_snr(raw, picks, window)
    rows, indices = np.nonzero(~(np.isfinite(snr) & (snr > threshold)))
    channels = [[raw.ch_names[picks[row]]] for row in rows]
    _annotate_bad_samples(
        raw, indices * length, (indices + 1) * length, channels, "BAD_snr"
    )
    return raw


def windowed_snr(
    raw: mne.io.BaseRaw, picks: np.ndarray, window: float
) -> tuple[np.ndarray, int]:
    """Mean divided by standard deviation of ``picks`` in consecutive windows.

    Returns
    -------
    snr : np.ndarray
        Shape ``(picks, windows)``. The remainder after the last full window
        is left out.
    length : int
        Window length in samples.
    """
    data, length = _windows(raw.get_data(picks=picks), raw.info["sfreq"], window)
    with np.errstate(divide="ignore", invalid="ignore"):
        return data.mean(axis=2) / data.std(axis=2), length


def _windows(data: ArrayLike, sfreq: float, window: float) -> tuple[np.ndarray, int]:
    """Split data into consecutive windows of ``window`` seconds.

    Parameters
    ----------
    data : array-like
        Shape ``(channels, times)``.
    sfreq : float
        Sampling frequency in Hz.
    window : float
        Window length in seconds.

    Returns
    -------
    windows : np.ndarray
        Shape ``(channels, windows, samples)``. The remainder after the last
        full window is left out.
    length : int
        Window length in samples.
    """
    data = np.asarray(data)
    length = int(np.ceil(window * sfreq))
    n_windows = data.shape[1] // length
    return data[:, : n_windows * length].reshape(len(data), n_windows, length), length


def _annotate_bad_samples(
    raw: mne.io.BaseRaw,
    starts: np.ndarray | list[int],
    stops: np.ndarray | list[int],
    channels: list[list[str]],
    description: str,
) -> None:
    """Annotate the sample spans ``[start, stop)`` of ``channels`` as bad.

    ``channels`` holds the channel names of each span. Sample 0 is the first
    sample of ``raw``.
    """
    if not channels:
        return
    starts, stops = np.asarray(starts), np.asarray(stops)
    sfreq = raw.info["sfreq"]
    raw.annotations.append(
        raw.first_time + starts / sfreq,
        (stops - starts) / sfreq,
        [description] * len(channels),
        channels,
    )


def _source_detector(ch_name: str) -> str:
    """Source-detector group of an fNIRS channel, e.g. ``S1_D1`` of ``S1_D1 hbo``."""
    return ch_name.rsplit(" ", 1)[0]


def _bad_spans(
    annotations: mne.Annotations,
) -> list[tuple[float, float, set[str]]]:
    """``BAD_`` annotations as ``(start, stop, source_detectors)``.

    Non-channel annotations have an empty set of source-detector groups.
    """
    return [
        (
            annotation["onset"],
            annotation["onset"] + annotation["duration"],
            {_source_detector(name) for name in annotation["ch_names"]},
        )
        for annotation in annotations
        if annotation["description"].lower().startswith("bad")
    ]


def _overlapping_spans(spans, start, stop):
    """Spans ``[on, off)`` that overlap the closed window ``[start, stop]``."""
    return [span for span in spans if span[0] <= stop and span[1] > start]


def _bad_channels(
    ch_names: list[str],
    spans: list[tuple[float, float, set[str]]],
    start: float,
    stop: float,
) -> list[str]:
    """Channels whose source-detector group has a bad span overlapping ``[start, stop]``."""
    groups = set().union(*(span[2] for span in _overlapping_spans(spans, start, stop)))
    return [name for name in ch_names if _source_detector(name) in groups]


def _epoch_windows(epochs: mne.BaseEpochs) -> list[tuple[float, float]]:
    """``(start, stop)`` of each epoch on the annotation time axis."""
    # Annotations keep the raw time axis, so events are placed with the raw rate.
    tzeros = epochs.events[:, 0] / epochs._raw_sfreq
    return [(tzero + epochs.tmin, tzero + epochs.tmax) for tzero in tzeros]


def convert_to_optical_density(raw):
    """Convert continuous-wave amplitude data to optical density."""
    if "fnirs_cw_amplitude" not in raw.get_channel_types():
        return raw
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Negative intensities encountered.*")
        return mne.preprocessing.nirs.optical_density(raw, verbose=False)


def beer_lambert_law(raw, ppf=6.0, fallback_distance=0.03):
    """Convert optical density to hemoglobin, using fallback sensor geometry."""
    picks = mne.pick_types(raw.info, meg=False, fnirs="fnirs_od", exclude=[])
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    if np.isfinite(distances).all() and (distances > 0).all():
        return mne.preprocessing.nirs.beer_lambert_law(raw, ppf=ppf)

    # TODO: this can be removed, MNE now supports fallback dist
    logger.warning(
        "Missing source-detector positions; using fallback distance of %.2f m",
        fallback_distance,
    )
    fallback_raw = raw.copy()
    original_locs = [fallback_raw.info["chs"][pick]["loc"].copy() for pick in picks]
    for pick in picks:
        loc = fallback_raw.info["chs"][pick]["loc"]
        loc[3:6] = 0.0
        loc[6:9] = (fallback_distance, 0.0, 0.0)
    converted = mne.preprocessing.nirs.beer_lambert_law(fallback_raw, ppf=ppf)
    for pick, loc in zip(picks, original_locs):
        converted.info["chs"][pick]["loc"] = loc
    return converted


def filter_raw(raw, l_freq, h_freq):
    """Band-pass raw data using the fnirsatlas Butterworth configuration."""
    if l_freq is None and h_freq is None:
        return raw
    return raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        l_trans_bandwidth=0.004,
        h_trans_bandwidth=0.01,
        filter_length="auto",
        method="iir",
        iir_params={"ftype": "butter", "order": 3},
        verbose=False,
    )


def mark_bad_segments_by_sci(raw, threshold=0.5, window=10.0):
    """Annotate optical-density windows with scalp coupling index below the threshold.

    SCI is the cardiac-band correlation of the two wavelengths of a
    source-detector group. Failing windows are annotated as ``BAD_sci``.
    ``None`` disables the check. The remainder after the last full window is
    not checked. Skipped with a warning when the sampling frequency is too low
    for the cardiac band.
    """
    if _skip_cardiac_check(raw, threshold, "SCI"):
        return raw
    windows, names, length = _cardiac_windows(raw, window)
    sci = _scalp_coupling_index(windows)
    groups, indices = np.nonzero(~np.isfinite(sci) | (sci < threshold))
    channels = [names[group] for group in groups]
    _annotate_bad_samples(
        raw, indices * length, (indices + 1) * length, channels, "BAD_sci"
    )
    return raw


def mark_bad_segments_by_psp(raw, threshold=0.02, window=10.0):
    """Annotate optical-density windows with peak spectral power below the threshold.

    PSP is the largest periodogram value of the cardiac-band cross-correlation
    of the two wavelengths of a source-detector group. Failing windows are
    annotated as ``BAD_psp``. ``None`` disables the check. The remainder after
    the last full window is not checked. Skipped with a warning when the
    sampling frequency is too low for the cardiac band.
    """
    if _skip_cardiac_check(raw, threshold, "PSP"):
        return raw
    windows, names, length = _cardiac_windows(raw, window)
    psp = _peak_spectral_power(windows, raw.info["sfreq"])
    groups, indices = np.nonzero(~np.isfinite(psp) | (psp < threshold))
    channels = [names[group] for group in groups]
    _annotate_bad_samples(
        raw, indices * length, (indices + 1) * length, channels, "BAD_psp"
    )
    return raw


def _skip_cardiac_check(raw: mne.io.BaseRaw, threshold, name: str) -> bool:
    """Whether a cardiac check is disabled or cannot run on ``raw``."""
    if threshold is None or "fnirs_od" not in raw.get_channel_types():
        return True
    sfreq = raw.info["sfreq"]
    if _cardiac_band(sfreq)[1] < 1.5:
        logger.warning(
            "Skipping %s check: sampling frequency %.2f Hz is too low for the "
            "cardiac band",
            name,
            sfreq,
        )
        return True
    return False


def _cardiac_band(sfreq: float) -> tuple[float, float]:
    """Cardiac band edges in Hz, the upper one lowered to stay below Nyquist."""
    # Pollonini et al. 2016 (PHOEBE), capped below Nyquist as in QT-NIRS.
    return 0.5, min(2.5, sfreq / 2 - 0.3)


def _cardiac_windows(
    raw: mne.io.BaseRaw, window: float
) -> tuple[np.ndarray, list[list[str]], int]:
    """Cardiac-band optical density split into source-detector groups and windows.

    The band is 0.5-2.5 Hz (30-150 bpm). Below 5.6 Hz sampling, the upper edge
    is lowered to stay below the Nyquist frequency.

    Returns
    -------
    windows : np.ndarray
        Shape ``(groups, wavelengths, windows, samples)``.
    names : list[list[str]]
        Channel names of each group.
    length : int
        Window length in samples.
    """
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, meg=False, fnirs="fnirs_od", exclude=[])
    groups: dict[str, list[int]] = {}
    for pick in picks:
        groups.setdefault(_source_detector(raw.ch_names[pick]), []).append(pick)
    rows = np.array(list(groups.values()))
    l_freq, h_freq = _cardiac_band(sfreq)
    data = mne.filter.filter_data(
        raw.get_data(picks=rows.ravel()),
        sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        l_trans_bandwidth=0.3,
        h_trans_bandwidth=0.3,
        verbose=False,
    )
    windows, length = _windows(data, sfreq, window)
    windows = windows.reshape(*rows.shape, *windows.shape[1:])
    names = [[raw.ch_names[pick] for pick in group] for group in rows]
    return windows, names, length


def _unit_windows(windows: np.ndarray) -> np.ndarray:
    """Zero-mean windows scaled to unit norm, same shape as ``windows``."""
    centered = windows - windows.mean(axis=-1, keepdims=True)
    # Zero variance gives NaN, which np.minimum keeps, so the group fails.
    with np.errstate(invalid="ignore", divide="ignore"):
        return centered / np.sqrt((centered**2).sum(-1, keepdims=True))


def _scalp_coupling_index(windows: np.ndarray) -> np.ndarray:
    """Lowest correlation over wavelength pairs, shape ``(groups, windows)``."""
    scaled = _unit_windows(windows)
    sci = np.full((windows.shape[0], windows.shape[2]), np.inf)
    wl_pairs = itertools.combinations(range(windows.shape[1]), 2)
    for wavelength_a, wavelength_b in wl_pairs:
        a, b = scaled[:, wavelength_a], scaled[:, wavelength_b]
        sci = np.minimum(sci, (a * b).sum(-1))
    return sci


def _peak_spectral_power(windows: np.ndarray, sfreq: float) -> np.ndarray:
    """Lowest cross-correlation peak power over wavelength pairs.

    Follows Pollonini et al. 2016 (PHOEBE) unbiased cross-correlation scaled so
    lag 0 is the SCI, power periodogram with a Hamming window. The peak is
    searched within the cardiac band. Two identical sinusoids give 0.5.

    Shape ``(groups, windows)``.
    """
    scaled = _unit_windows(windows)
    length = windows.shape[-1]
    lags = np.arange(1 - length, length)
    l_freq, h_freq = _cardiac_band(sfreq)
    psp = np.full((windows.shape[0], windows.shape[2]), np.inf)
    wl_pairs = itertools.combinations(range(windows.shape[1]), 2)
    for wavelength_a, wavelength_b in wl_pairs:
        a, b = scaled[:, wavelength_a], scaled[:, wavelength_b]
        # Equivalent to np.correlate(a, b, "full") per window. Convolution with b
        # reversed in time is the cross-correlation, and fftconvolve computes it
        # for all groups and windows at once, which np.correlate cannot.
        correlation = fftconvolve(a, b[..., ::-1], axes=-1)
        correlation = length * correlation / (length - np.abs(lags))
        freqs, power = periodogram(
            correlation, fs=sfreq, window="hamming", scaling="spectrum", axis=-1
        )
        in_band = (freqs >= l_freq) & (freqs <= h_freq)
        psp = np.minimum(psp, power[..., in_band].max(-1))
    return psp


def mark_bad_segments_by_std(
    raw: mne.io.BaseRaw, max_std: dict[str, float] | None = None, window: float = 10.0
) -> mne.io.BaseRaw:
    """Annotate channel windows whose std exceeds an absolute limit as ``BAD_std``.

    Channels in ``info["bads"]`` are skipped. The remainder after the last
    full window is not checked.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Hemoglobin recording in M. Modified in place.
    max_std : dict[str, float] | None
        Largest tolerated std in uM per channel type, e.g.
        ``{"hbo": 0.75, "hbr": 0.25}``. ``None`` disables the check.
    window : float
        Window length in seconds.

    Returns
    -------
    raw : mne.io.BaseRaw
        Input recording with high-std windows annotated.

    Notes
    -----
    The std depends on the band-pass. The example limits are about 10x the
    median std of 10 s windows band-passed to 0.01-0.1 Hz.
    """
    if max_std is None:
        return raw
    for kind, limit in max_std.items():
        picks = mne.pick_types(raw.info, meg=False, fnirs=kind, exclude="bads")
        data, length = _windows(raw.get_data(picks=picks), raw.info["sfreq"], window)
        rows, indices = np.nonzero(1e6 * data.std(axis=2) > limit)
        channels = [[raw.ch_names[picks[row]]] for row in rows]
        _annotate_bad_samples(
            raw, indices * length, (indices + 1) * length, channels, "BAD_std"
        )
    return raw


def bad_data_mask(
    raw: mne.io.BaseRaw, descriptions: list[str] | None = None
) -> np.ndarray:
    """Bad samples of each fNIRS channel, shape ``(channels, samples)``.

    A sample is bad when its channel is in ``info["bads"]`` or a
    channel-specific ``BAD_`` annotation of its source-detector group covers
    it. Every wavelength or chromophore of an annotated group is bad.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with bad channels and segments annotated.
    descriptions : list[str] | None
        Count only annotations with these descriptions and ignore
        ``info["bads"]``. ``None`` counts everything.
    """
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    names = [raw.ch_names[pick] for pick in picks]
    bad = np.zeros((len(names), raw.n_times), dtype=bool)
    annotations = raw.annotations
    if descriptions is None:
        bad[[i for i, name in enumerate(names) if name in raw.info["bads"]]] = True
    else:
        annotations = annotations[np.isin(annotations.description, descriptions)]
    for start, stop, groups in _bad_spans(annotations):
        rows = [i for i, name in enumerate(names) if _source_detector(name) in groups]
        first, last = raw.time_as_index(
            np.array([start, stop]) - raw.first_time, use_rounding=True
        ).clip(0, raw.n_times)
        bad[rows, first:last] = True
    return bad


def drop_recording_with_too_many_bads(raw, max_bad_fraction=None):
    """Drop a recording whose share of bad fNIRS data exceeds the limit.

    The share is the fraction of fNIRS channel samples that are in
    ``info["bads"]`` or covered by a channel-specific ``BAD_`` annotation.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording with bad channels and segments already annotated.
    max_bad_fraction : float | None
        Largest tolerated fraction of bad fNIRS data. ``None`` keeps every
        recording.

    Returns
    -------
    raw : mne.io.BaseRaw | None
        The input recording, or ``None`` when too much data is bad.
    """
    if max_bad_fraction is None:
        return raw
    bad = bad_data_mask(raw)
    if bad.size == 0:
        return raw
    fraction = bad.mean()
    if fraction <= max_bad_fraction:
        return raw
    logger.warning(
        "Dropping recording: %.0f%% of fNIRS data is bad (limit %.0f%%)",
        100 * fraction,
        100 * max_bad_fraction,
    )
    return None


def interpolate_dist(inst, temperature=0.005):
    """Replace bad fNIRS channels with a distance-weighted average of good ones.

    Each bad channel is rebuilt from the good channels sharing its wavelength or
    chromophore, weighted by ``softmax(-d / temperature)`` over the distances
    ``d`` between channel midpoints.

    Parameters
    ----------
    inst : mne.io.BaseRaw | mne.BaseEpochs
        Recording or epochs with bad channels marked. Modified in place.
    temperature : float
        Softmax temperature in meters. Lower values concentrate the weight on
        the nearest channel.

    Returns
    -------
    inst : mne.io.BaseRaw | mne.BaseEpochs
        Input with bad fNIRS channels replaced and ``info["bads"]`` cleared.

    Raises
    ------
    ValueError
        When a group with bad channels has no good channel, or a channel of
        such a group has no finite position.
    """
    picks = mne.pick_types(inst.info, meg=False, fnirs=True, exclude=[])
    if len(picks) == 0:
        return inst
    inst.load_data()
    positions = np.array([inst.info["chs"][pick]["loc"][:3] for pick in picks])
    unplaced = np.any(~np.isfinite(positions), axis=1)
    bads = set(inst.info["bads"])
    for key, members in _group_by_wavelength(inst, picks).items():
        is_bad = np.array([inst.ch_names[picks[m]] in bads for m in members])
        if not is_bad.any():
            continue
        if is_bad.all():
            raise ValueError(f"Every {key} channel is bad, nothing to interpolate from")
        if unplaced[members].any():
            names = [inst.ch_names[picks[m]] for m in members[unplaced[members]]]
            raise ValueError(
                f"{len(names)} fNIRS channels without positions, e.g. {names[:3]}"
            )
        good, bad = members[~is_bad], members[is_bad]
        weights = _softmax_over_distance(
            positions[bad], positions[good], temperature=temperature
        )
        source = inst._data[..., picks[good], :]
        inst._data[..., picks[bad], :] = np.einsum("bg,...gt->...bt", weights, source)
    inst.info["bads"] = []
    return inst


def _group_by_wavelength(inst, picks):
    """Map each wavelength or chromophore to its channel indices within picks."""
    groups = {}
    for index, pick in enumerate(picks):
        # MNE names fNIRS channels ``S1_D1 760`` or ``S1_D1 hbo``.
        groups.setdefault(inst.ch_names[pick].split()[-1], []).append(index)
    return {key: np.array(members) for key, members in groups.items()}


def _softmax_over_distance(bad_positions, good_positions, temperature):
    """Softmax weights over the negated distances from each bad to each good channel."""
    distances = np.linalg.norm(bad_positions[:, None] - good_positions[None], axis=-1)
    logits = -(distances - distances.min(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def interpolate_nearest(inst):
    """Replace bad fNIRS channels with their nearest matching channel."""
    if not inst.info["bads"]:
        return inst
    # MNE requires an origin before dispatch, but fNIRS interpolation ignores it.
    return inst.interpolate_bads(reset_bads=True, origin=(0.0, 0.0, 0.0), verbose=False)


def interpolate_zero(inst):
    """Replace bad fNIRS channels with zeros.

    Zero is neutral only for centred data, e.g. baseline-corrected epochs.
    """
    picks = mne.pick_types(inst.info, meg=False, fnirs=True, exclude=[])
    bads = [pick for pick in picks if inst.ch_names[pick] in inst.info["bads"]]
    inst.load_data()
    inst._data[..., bads, :] = 0.0
    inst.info["bads"] = []
    return inst


def repair_temporal_derivatives(raw, enabled=True):
    """Repair motion artifacts with temporal derivative distribution repair."""
    if not enabled:
        return raw
    return mne.preprocessing.nirs.temporal_derivative_distribution_repair(
        raw, verbose=False
    )


def convert_to_hemoglobin(raw, ppf=6.0):
    """Convert optical-density data to hemoglobin concentration."""
    if "fnirs_od" not in raw.get_channel_types():
        return raw
    return beer_lambert_law(raw, ppf=ppf)


def resample_raw(raw, sfreq=None):
    """Resample to a fixed rate."""
    if sfreq is None or raw.info["sfreq"] == sfreq:
        return raw
    return raw.resample(sfreq, verbose=False)


def interpolate_bads_step(method=interpolate_dist):
    """Pipeline step replacing bad fNIRS channels per epoch.

    In each epoch, every wavelength or chromophore of each source-detector
    group overlapped by a channel-specific ``BAD_`` annotation is replaced,
    as are the channels in ``info["bads"]``, which is cleared.

    Parameters
    ----------
    method : callable | None
        Function taking and returning ``mne.BaseEpochs``. It replaces the
        channels in ``info["bads"]`` and clears it, e.g.
        ``interpolate_nearest``, ``interpolate_zero`` or
        ``partial(interpolate_dist, temperature=0.01)``.
        Its name and bound
        arguments enter the MOABB cache key, so avoid lambdas. ``None`` keeps
        the bad data.
    """
    return _step(
        _interpolate_bad_channels_per_epoch, "Interpolate Bad Channels", method=method
    )


def crop_to_data(epochs, tmin, tmax):
    """Crop epochs to ``[tmin, tmax]``, ending at the last sample within ``tmax``."""
    return epochs.crop(tmin=tmin, tmax=min(tmax, epochs.tmax), verbose=False)


def crop_to_data_step(tmin, tmax):
    """Pipeline step cropping epochs without warning when ``tmax`` is between samples."""
    return _step(crop_to_data, "Crop", tmin=tmin, tmax=tmax)


def load_data_step():
    """Pipeline step loading epochs into memory."""
    return _step(methodcaller("load_data"), "Load Data")


def _bad_channels_per_epoch(epochs: mne.BaseEpochs) -> list[list[str]]:
    """Bad channels of each epoch: ``info["bads"]`` plus annotated groups overlapping it."""
    spans = _bad_spans(epochs.annotations)
    return [
        sorted(
            {*_bad_channels(epochs.ch_names, spans, start, stop), *epochs.info["bads"]}
        )
        for start, stop in _epoch_windows(epochs)
    ]


def _interpolate_bad_channels_per_epoch(
    epochs: mne.BaseEpochs, method: Callable[[mne.BaseEpochs], mne.BaseEpochs] | None
) -> mne.BaseEpochs:
    """Run ``method`` on groups of epochs sharing the same bad channels.

    MNE interpolates each ``info["bads"]`` for all epochs. Here a channel is
    bad only in the epochs its ``BAD_`` annotation overlaps, so epochs are
    grouped by bad set and each group is interpolated with its own
    ``info["bads"]``. ``method`` reads ``info["bads"]`` of the group and
    replaces those channels. ``info["bads"]`` of the result is cleared.
    ``None`` leaves the epochs as is.
    """
    if method is None:
        return epochs
    data = epochs.load_data()._data
    assert isinstance(data, np.ndarray)
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, bads in enumerate(_bad_channels_per_epoch(epochs)):
        if bads:
            groups.setdefault(tuple(bads), []).append(index)
    for bads, indices in groups.items():
        subset = epochs[indices]
        subset.info["bads"] = list(bads)
        data[indices] = method(subset).get_data(picks="all")
    epochs.info["bads"] = []
    return epochs


def drop_bad_epochs_step(max_bad_fraction=0.5, min_good_pairs=8):
    """Pipeline step dropping epochs with too many bad fNIRS channels.

    An epoch is dropped when its fraction of bad channels exceeds
    ``max_bad_fraction`` and fewer than ``min_good_pairs`` source-detector
    pairs have no bad channel. The defaults are very lenient to keep as much
    data as possible. The default ``min_good_pairs`` is about the smallest
    montage among the benchmark datasets.

    A channel is bad in an epoch when it is in ``info["bads"]`` or a
    channel-specific ``BAD_`` annotation of its source-detector group overlaps
    the epoch. These are the channels ``interpolate_bads_step`` replaces.
    Dropped epochs get the reason ``BAD_fraction``.

    Parameters
    ----------
    max_bad_fraction : float | None
        Largest tolerated fraction of bad fNIRS channels per epoch. ``None``
        keeps every epoch.
    min_good_pairs : int | None
        Number of good source-detector pairs that keeps an epoch regardless of
        its bad fraction. ``None`` drops on the fraction alone.
    """
    return _step(
        _drop_bad_epochs,
        "Drop Bad Epochs",
        max_bad_fraction=max_bad_fraction,
        min_good_pairs=min_good_pairs,
    )


def _drop_bad_epochs(
    epochs: mne.BaseEpochs,
    max_bad_fraction: float | None,
    min_good_pairs: int | None,
) -> mne.BaseEpochs:
    if max_bad_fraction is None:
        return epochs
    picks = mne.pick_types(epochs.info, meg=False, fnirs=True, exclude=[])
    names = {epochs.ch_names[pick] for pick in picks}
    pairs = {_source_detector(name) for name in names}
    bads_per_epoch = [
        names.intersection(bads) for bads in _bad_channels_per_epoch(epochs)
    ]
    too_bad = np.array(
        [len(bads) / len(names) > max_bad_fraction for bads in bads_per_epoch]
    )
    if min_good_pairs is not None:
        n_good_pairs = np.array(
            [
                len(pairs - {_source_detector(name) for name in bads})
                for bads in bads_per_epoch
            ]
        )
        too_bad &= n_good_pairs < min_good_pairs
    dropped = np.flatnonzero(too_bad)
    if len(dropped):
        logger.warning(
            "Dropping %d of %d epochs with more than %.0f%% bad fNIRS channels"
            " and fewer than %s good source-detector pairs",
            len(dropped),
            len(epochs),
            100 * max_bad_fraction,
            min_good_pairs,
        )
    return epochs.drop(dropped, reason="BAD_fraction", verbose=False)


def reject_by_annotation_step():
    """Pipeline step dropping epochs that overlap a non-channel ``BAD_`` annotation.

    Channel-specific annotations never drop an epoch. Their channels are
    handled by ``interpolate_bads_step``.
    """
    return _step(_reject_by_annotation, "Reject Bad Epochs")


def _reject_by_annotation(epochs):
    spans = [span for span in _bad_spans(epochs.annotations) if not span[2]]
    rejected = [
        index
        for index, (start, stop) in enumerate(_epoch_windows(epochs))
        if _overlapping_spans(spans, start, stop)
    ]
    return epochs.drop(rejected, reason="BAD_annotation", verbose=False)


def _skip_dropped(func):
    """Pass a dropped run through a step untouched.

    A run is dropped when a step returns ``None`` or empty epochs. MOABB
    skips runs whose processing returns ``None``.
    """

    @functools.wraps(func)
    def step(inst, **kw_args):
        if inst is None:
            return None
        out = func(inst, **kw_args)
        if isinstance(out, mne.BaseEpochs) and len(out) == 0:
            logger.warning("Dropping run: no epochs left")
            return None
        return out

    return step


def _step(func, display_name, **kw_args):
    """Wrap a preprocessing function as a named pipeline step.

    Parameters are appended to the step name because MOABB derives its epoch
    cache key and result keys from the pipeline repr. Without them, runs that
    differ only in preprocessing settings collide in the cache.
    """
    params = ", ".join(f"{name}={value}" for name, value in kw_args.items())
    name = f"{display_name} ({params})" if params else display_name
    return NamedFunctionTransformer(
        _skip_dropped(func), display_name=name, kw_args=kw_args
    )


def preprocess_fnirs(
    short_channel_threshold=None,
    flat_threshold=1e-6,
    snr_threshold=10.0,
    sci_threshold=0.5,
    psp_threshold=0.02,
    quality_window=10.0,
    tddr=True,
    ppf=6.0,
    l_freq=0.01,
    h_freq=0.1,
    resample=None,
    max_std=None,
):
    """Construct the fNIRS signal preprocessing pipeline."""
    steps = [
        _step(mark_bad_segments_by_invalid_values, "Invalid Value Check"),
        _step(
            mark_bad_segments_by_flat,
            "Flat Check",
            threshold=flat_threshold,
            window=quality_window,
        ),
        _step(
            mark_bad_segments_by_snr,
            "SNR Check",
            threshold=snr_threshold,
            window=quality_window,
        ),
        _step(convert_to_optical_density, "Optical Density"),
        _step(
            preprocess_short_channels,
            "Remove Short Channels",
            threshold=short_channel_threshold,
        ),
        _step(
            mark_bad_segments_by_sci,
            "Scalp Coupling Index",
            threshold=sci_threshold,
            window=quality_window,
        ),
        _step(
            mark_bad_segments_by_psp,
            "Peak Spectral Power",
            threshold=psp_threshold,
            window=quality_window,
        ),
        _step(repair_temporal_derivatives, "TDDR", enabled=tddr),
        _step(convert_to_hemoglobin, "Beer-Lambert Law", ppf=ppf),
        _step(
            filter_raw,
            "Band Pass Filter",
            l_freq=l_freq,
            h_freq=h_freq,
        ),
        _step(resample_raw, "Resample", sfreq=resample),
        _step(
            mark_bad_segments_by_std,
            "Std Check",
            max_std=max_std,
            window=quality_window,
        ),
    ]
    return make_pipeline(*steps)


def preprocess_short_channels(raw, threshold=None):
    """Remove fNIRS channels shorter than the distance threshold in meters."""
    if threshold is None:
        return raw
    # exclude=[] keeps bad channels, which interpolation repairs later.
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    if not np.isfinite(distances).any():
        logger.warning("No valid source-detector distances; keeping short channels")
        return raw
    return raw.drop_channels(
        [raw.ch_names[pick] for pick in picks[distances < threshold]]
    )


def _crop_leading_clipped_samples(raw, floor=-np.inf, ceiling=np.inf):
    """Crop leading samples where every fNIRS channel is at ``floor`` or ``ceiling``."""
    data = raw.get_data(picks="fnirs")
    clipped = ((data <= floor) | (data >= ceiling)).all(axis=0)
    if clipped.all():
        raise ValueError(f"All samples are at or outside [{floor}, {ceiling}]")
    n_leading = int(np.argmin(clipped))
    if n_leading == 0:
        return raw
    return raw.crop(tmin=raw.times[n_leading])


def _mark_bad_segments_by_hampel(raw, window_seconds=10, threshold=20):
    """Annotate channel-specific outlier segments using a Hampel detector.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to annotate. Modified in place.
    window_seconds : float
        Duration of the centered detection window in seconds.
    threshold : float
        Number of scaled median absolute deviations used as the cutoff.

    Returns
    -------
    raw : mne.io.BaseRaw
        Input recording with detected segments annotated as ``BAD_hampel``.
    """
    window = round(window_seconds * raw.info["sfreq"])
    window += not window % 2
    half_window = window // 2

    def outlier_segments(data):
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(data, half_window, mode="reflect"), window
        )
        median = np.median(windows, axis=1)
        mad = np.median(np.abs(windows - median[:, None]), axis=1)
        outliers = np.abs(data - median) > threshold * 1.4826 * mad
        edges = np.diff(np.pad(outliers.astype(int), (1, 1)))
        return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))

    starts, stops, channels = [], [], []
    for channel, data, channel_type in zip(
        raw.ch_names, raw.get_data(), raw.get_channel_types(), strict=True
    ):
        if channel_type == "stim":
            continue
        for start, stop in outlier_segments(data):
            starts.append(start)
            stops.append(stop)
            channels.append([channel])
    _annotate_bad_samples(raw, starts, stops, channels, "BAD_hampel")
    return raw


def _promote_bad_segments_to_bad_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Mark every channel of a source-detector group bad when any segment of it is bad.

    The channel-specific ``BAD_`` annotations stay as provenance.
    """
    groups = {
        _source_detector(name)
        for annotation in raw.annotations
        if annotation["description"].lower().startswith("bad")
        for name in annotation["ch_names"]
    }
    bads = [name for name in raw.ch_names if _source_detector(name) in groups]
    raw.info["bads"] = list(dict.fromkeys([*raw.info["bads"], *bads]))
    return raw


def _interpolate_bad_segments_temporal(raw, max_samples=1):
    """Linearly bridge channel-specific bad segments of at most max_samples temporally.

    Segments of at most ``max_samples`` are repaired and lose their
    annotation.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording to interpolate. Modified in place.
    max_samples : int
        Maximum segment length whose annotation is removed.

    Returns
    -------
    raw : mne.io.BaseRaw
        Input recording with bad segments bridged and short ones unannotated.
    """
    segments = {}
    repaired = np.zeros(len(raw.annotations), dtype=bool)
    for index, annotation in enumerate(raw.annotations):
        if (
            not annotation["description"].lower().startswith("bad")
            or not annotation["ch_names"]
        ):
            continue
        times = np.array(
            [annotation["onset"], annotation["onset"] + annotation["duration"]]
        )
        start, stop = raw.time_as_index(times - raw.first_time, use_rounding=True)
        repaired[index] = stop - start <= max_samples
        for channel in annotation["ch_names"]:
            segments.setdefault(channel, []).append((start, stop))

    def interpolate_channel(data, ch_name):
        for start, stop in segments[ch_name]:
            if start == 0:
                data[start:stop] = data[stop]
            elif stop >= data.size:
                data[start:stop] = data[start - 1]
            else:
                data[start:stop] = np.interp(
                    np.arange(start, stop), [start - 1, stop], data[[start - 1, stop]]
                )
        return data

    if segments:
        raw.load_data().apply_function(interpolate_channel, picks=list(segments))
    raw.set_annotations(raw.annotations[~repaired])
    return raw
