# Adapted from https://github.com/mne-tools/mne-python (BSD-3-Clause).

"""
Preprocessing functions for fNIRS data.
"""

import numpy as np
import mne
from itertools import compress
import logging
from fnirsatlas.config import get_preprocessing_config
from fnirsatlas.channels import (
    validate_ch_names,
    validate_ch_locations,
    to_mne_ch_order,
)
from typing import Dict, Any
from mne.preprocessing.nirs import temporal_derivative_distribution_repair

logger = logging.getLogger(__name__)


def beer_lambert_law(raw, ppf=6.0):
    """Convert optical density to haemoglobin concentration.

    Wraps `mne.preprocessing.nirs.beer_lambert_law`. Source-detector distances
    come from the sensor positions, or are 0.03 m for all channels if any
    position is missing.

    Parameters
    ----------
    raw : instance of Raw
        Optical density data.
    ppf : float
        Partial pathlength factor.

    Returns
    -------
    raw : instance of Raw
        Haemoglobin concentration data.
    """
    with mne.utils.use_log_level("ERROR"):
        distances = mne.preprocessing.nirs.source_detector_distances(raw.info, picks="all")
    sd_distances = 0.03 if np.isnan(distances).any() else None
    return mne.preprocessing.nirs.beer_lambert_law(raw, ppf=ppf, sd_distances=sd_distances)


def _get_mbll_distances(raw, picks, fallback=0.03):
    from mne.preprocessing.nirs import source_detector_distances

    with mne.utils.use_log_level("ERROR"):
        distances = source_detector_distances(raw.info, picks="all")
    if np.isnan(distances).any():
        logger.debug(f"Using default distance of {fallback}")
        distances = np.full(len(raw.ch_names), fallback)
    distances = distances.copy()
    bad = ~np.isfinite(distances[picks]) | (distances[picks] <= 0)
    if bad.any():
        logger.warning(
            "Source-detector distances are zero on NaN, some resulting "
            "concentrations will be zero. Consider setting a montage "
            "with raw.set_montage."
        )
    distances[picks[bad]] = 0.0
    return distances


def _get_hb_pair_picks(raw, picks):
    pair_to_picks = {}
    for pick in picks:
        pair = raw.ch_names[pick].split(" ")[0]
        pair_to_picks.setdefault(pair, []).append(pick)
    for pair, pair_picks in pair_to_picks.items():
        if len(pair_picks) != 2:
            raise ValueError(
                f"Expected 2 Hb channels for {pair}, found {len(pair_picks)}"
            )
        yield pair, sorted(pair_picks, key=lambda pick: raw.info["chs"][pick]["loc"][9])


def inverse_mbll(raw_hb, ppf=6.0):
    r"""Convert haemoglobin concentration data back to optical density."""
    from mne.io.constants import FIFF
    from mne.preprocessing.nirs import _validate_nirs_info
    from mne.preprocessing.nirs._beer_lambert_law import _load_absorption

    raw_hb = raw_hb.copy().load_data()
    picks = _validate_nirs_info(raw_hb.info, fnirs="hb", which="Inverse Beer-lambert")
    distances = _get_mbll_distances(raw_hb, picks)
    raw_od = raw_hb.copy()
    rename = {}
    for pair, pair_picks in _get_hb_pair_picks(raw_hb, picks):
        freqs = np.array(
            [raw_hb.info["chs"][pick]["loc"][9] for pick in pair_picks], float
        )
        abs_coef = _load_absorption(freqs)
        el_matrix = abs_coef * distances[pair_picks[0]] * float(ppf)
        raw_od._data[pair_picks] = el_matrix @ raw_hb._data[pair_picks] * 1e3
        for idx, freq in zip(pair_picks, freqs):
            ch = raw_od.info["chs"][idx]
            ch.update(coil_type=FIFF.FIFFV_COIL_FNIRS_OD, unit=FIFF.FIFF_UNIT_NONE)
            rename[raw_od.ch_names[idx]] = f"{pair} {int(freq)}"
    raw_od.rename_channels(rename)
    _validate_nirs_info(raw_od.info, fnirs="od", which="Inverse Beer-lambert output")
    return raw_od


def fill_missing_chs(raw, all_chs):
    """Adds empty channels to raw data if they are missing. Marks them as bad so they can be interpolated later."""
    missing_chs = set(all_chs) - set(raw.ch_names)
    if missing_chs:
        logger.info(f"Adding missing channels: {missing_chs}")
        for ch in missing_chs:
            missing_ch_info = mne.create_info(
                ch_names=[ch], sfreq=raw.info["sfreq"], ch_types=[ch.split(" ")[1]]
            )
            missing_ch_data = mne.io.RawArray(
                np.zeros((1, raw.n_times)), missing_ch_info, verbose=False
            )
            raw.add_channels([missing_ch_data], force_update_info=True)
            raw.info["bads"].append(ch)
    return raw


def apply_sci(raw, sci_threshold, verbose=False):
    """Helper function to mark channels as bad based on the scalp coupling index (SCI)."""
    if sci_threshold == 0:
        return raw
    with np.errstate(
        invalid="ignore"
    ):  # some channels have all zeros, they will be eliminated
        sci = mne.preprocessing.nirs.scalp_coupling_index(raw)
    below_threshold = np.isnan(sci) | (sci <= sci_threshold)
    elim_ratio = np.sum(below_threshold) / len(below_threshold)
    if elim_ratio > 0.5:
        logger.warning(f"Warning: {elim_ratio:.2f} channels eliminated due to low SCI")
    if elim_ratio == 1:
        logger.warning("Warning: All channels have low SCI, no channels marked as bad")
    else:
        raw.info["bads"] = list(compress(raw.ch_names, below_threshold))
        logger.debug(
            f"{len(raw.info['bads'])}/{len(raw.ch_names)} channels marked as bad"
        )
    return raw


def remove_short_channels(raw, min_length=0.01):
    """Remove channels with source-detector distance less than min_length."""
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    if dists is None or np.isnan(dists).all():
        logger.warning(
            "No valid source-detector distances (likely missing montage). Skipping short channel removal."
        )
        return raw
    elim_ix = dists < min_length
    logger.info(f"Removing {np.sum(elim_ix)} short channels")
    raw.pick(picks[~elim_ix])
    return raw


def _has_valid_montage(raw):
    """Check if raw has a valid montage with meaningful positions."""
    montage = raw.get_montage()
    if montage is None:
        logger.debug("No montage found")
        return False

    # Check if montage has any positions
    pos = getattr(montage, "get_positions", lambda: None)()
    if pos is None:
        logger.debug("No positions found in montage")
        return False

    ch_pos = pos.get("ch_pos", {})
    if not ch_pos:
        logger.debug("No channel positions found in montage")
        return False

    # If all positions are zero, treat as missing
    if all(np.allclose(v, 0) for v in ch_pos.values()):
        logger.debug("All positions are zero")
        return False

    return True


def process_raw(raw: mne.io.Raw, dataset_name: str = None, **kwargs):
    """
    Apply preprocessing to a raw object.

    1. Detects bad channels with SCI and interpolates them. Use `ch_interpolation`=None to not interpolate.
    2. Applies TDDR to the raw object.
    3. Converts the optical density data to haemoglobin concentration using the modified Beer-Lambert law (only if not already in HbO/HbR format).
    4. Applies a band-pass filter to the raw object

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object.
    dataset_name : str, optional
        Dataset name for dataset-specific preprocessing settings.
    **kwargs : dict
        Override any preprocessing parameters. Takes precedence over config values.

    Returns
    -------
    raw : mne.io.Raw
        Processed raw data.
    """
    # Get configuration with proper precedence: kwargs > dataset-specific > global
    config = _get_preprocessing_config_with_overrides(dataset_name, **kwargs)

    # Optional hard skip: return raw unchanged
    if config.get("skip_all", False):
        logger.info("Preprocessing skipped due to 'skip_all' in config")
        return raw

    # Channel validation and correction
    if not validate_ch_names(raw.ch_names):
        logger.debug("Found non-mne compatible channel names, reordering channels...")
        raw.reorder_channels(to_mne_ch_order(raw.info["chs"]))

    # Check montage availability
    has_montage = _has_valid_montage(raw)

    if not has_montage:
        logger.info(
            "No montage available. Spatial preprocessing steps (interpolation, short channel removal, regression) will be skipped."
        )
        montage = None
    elif has_montage and not validate_ch_locations(raw.info["chs"]):
        logger.info("Found non-mne compatible channel locations. Not using montage")
        has_montage = False
        montage = None
    else:
        montage = raw.get_montage()

    # Data format conversion
    current_channel_types = set(raw.get_channel_types())
    if "fnirs_cw_amplitude" in current_channel_types:
        logger.debug(
            "Data is in fnirs_cw_amplitude format, converting to optical density..."
        )
        if dataset_name and dataset_name.startswith("shin2016"):
            raw = _optical_density_with_warning_handling(raw)
        else:
            raw = mne.preprocessing.nirs.optical_density(raw)

        current_channel_types = set(raw.get_channel_types())

    # Short channel regression (if enabled and montage available)
    if has_montage and config.get("short_ch_regression") is not None:
        try:
            from mne_nirs.signal_enhancement import short_channel_regression

            raw = short_channel_regression(raw, max_dist=config["short_ch_regression"])
            logger.debug(
                f"Applied short channel regression with max_dist={config['short_ch_regression']}"
            )
        except ImportError:
            # TODO use mne_nirs as preprocessing dependency be default
            logger.warning("mne_nirs not available, skipping short channel regression")
        except Exception as e:
            logger.warning(
                f"Short channel regression failed (possibly missing montage): {e}. Skipping."
            )

    # Remove short channels (if enabled and montage available)
    if has_montage and config.get("remove_short") is not None:
        raw = remove_short_channels(raw, min_length=config["remove_short"])
        logger.debug(f"Removed short channels with distance < {config['remove_short']}")

    # Temporal Derivative Distribution Repair
    if config.get("tddr", True):
        raw = temporal_derivative_distribution_repair(raw)
        logger.debug("Applied TDDR motion correction")

    # Beer-Lambert Law conversion and SCI
    if "fnirs_od" in current_channel_types:
        # Apply SCI threshold
        if config.get("sci_threshold", 0) > 0:
            raw = apply_sci(
                raw, config["sci_threshold"], verbose=config.get("verbose", False)
            )
            logger.debug(f"Applied SCI threshold: {config['sci_threshold']}")

        # Convert to hemoglobin concentration
        raw = beer_lambert_law(raw, ppf=config.get("bll_ppf", 6))
        logger.debug(
            f"Converted to hemoglobin concentration with PPF={config.get('bll_ppf', 6)}"
        )

    # Channel interpolation
    ch_interpolation = config.get("ch_interpolation")
    if ch_interpolation == "interpolate_nearest" and has_montage and raw.info["bads"]:
        raw.interpolate_bads(method=dict(fnirs="nearest"), origin=(0.0, 0.0, 0.0))
        logger.debug("Interpolated bad channels using nearest neighbor")
    elif ch_interpolation == "drop":
        if raw.info["bads"]:
            raw.drop_channels(raw.info["bads"])
            logger.debug(f"Dropped {len(raw.info['bads'])} bad channels")
    elif ch_interpolation is None:
        logger.debug("Skipping channel interpolation")

    # Band-pass filtering
    filter_kwargs = config.get("filter_kwargs", {})
    raw = raw.filter(
        l_freq=config.get("l_freq", 0.01),
        h_freq=config.get("h_freq", 0.1),
        l_trans_bandwidth=config.get("l_trans_bandwidth", 0.004),
        h_trans_bandwidth=config.get("h_trans_bandwidth", 0.01),
        filter_length=config.get("filter_length", "auto"),
        verbose=False,
        **filter_kwargs,
    )
    logger.debug(
        f"Applied band-pass filter: {config.get('l_freq', 0.01)}-{config.get('h_freq', 0.1)} Hz"
    )

    # Restore montage after all preprocessing steps
    if montage is not None:
        raw.set_montage(montage)

    return raw


def _get_preprocessing_config_with_overrides(
    dataset_name: str = None, **kwargs
) -> Dict[str, Any]:
    """
    Get preprocessing configuration with proper precedence:
    1. Function kwargs (highest priority)
    2. Dataset-specific settings
    3. Global preprocessing settings (lowest priority)
    """
    from fnirsatlas.config import get_preprocessing_config, get_dataset_config

    # Start with global preprocessing config
    config = get_preprocessing_config().copy()

    # Override with dataset-specific settings if available
    if dataset_name:
        try:
            dataset_config = get_dataset_config(dataset_name)
            if "preprocessing" in dataset_config:
                config.update(dataset_config["preprocessing"])
                logger.debug(
                    f"Applied dataset-specific preprocessing settings for {dataset_name}"
                )
        except ValueError:
            logger.debug(f"No dataset-specific config found for {dataset_name}")

    # Override with function kwargs (highest priority)
    config.update(kwargs)

    return config


def _optical_density_with_warning_handling(raw):
    """Convert to optical density while catching and rerouting Shin2016 warnings."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        raw = mne.preprocessing.nirs.optical_density(raw)
        for warning in w:
            if "Negative intensities encountered" in str(warning.message):
                logger.warning(f"EXPECTED WARNING: {warning.message}")
            else:
                warnings.showwarning(
                    warning.message, warning.category, warning.filename, warning.lineno
                )
    return raw

