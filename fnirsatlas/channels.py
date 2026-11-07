import re
import logging
from typing import List, Dict
import mne
import numpy as np
from fnirsatlas.utils import cluster_means
from mne.io.constants import FIFF
from collections import defaultdict

COIL_TYPE_TO_CH_TYPE = {
    FIFF.FIFFV_COIL_FNIRS_OD: "fnirs_od",
    FIFF.FIFFV_COIL_FNIRS_HBO: "hbo",
    FIFF.FIFFV_COIL_FNIRS_HBR: "hbr",
    FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE: "fnirs_cw_amplitude",
    FIFF.FIFFV_COIL_EEG: "eeg",
    FIFF.FIFFV_COIL_EEG_BIPOLAR: "eeg",
    FIFF.FIFFV_COIL_EEG_CSD: "eeg",
    FIFF.FIFFV_STIM_CH: "stim",
    # Add more as needed
}
KIND_TO_STRING = {
    FIFF.FIFFV_FNIRS_CH: "fnirs",
}
UNIT_TO_STRING = {
    FIFF.FIFF_UNIT_MOL: "mol",
}
CH_TYPE_TO_COIL_TYPE = {v: k for k, v in COIL_TYPE_TO_CH_TYPE.items()}
COIL_TYPE_TO_INT = {k: int(k) for k in COIL_TYPE_TO_CH_TYPE.keys()}
INT_TO_COIL_TYPE = {v: k for k, v in COIL_TYPE_TO_INT.items()}
KIND_TO_INT = {k: int(k) for k in KIND_TO_STRING.keys()}
INT_TO_KIND = {v: k for k, v in KIND_TO_INT.items()}
UNIT_TO_INT = {k: int(k) for k in UNIT_TO_STRING.keys()}
INT_TO_UNIT = {v: k for k, v in UNIT_TO_INT.items()}

logger = logging.getLogger(__name__)


def handle_varying_wls(ch_names: List[str]) -> List[str]:
    """
    Renames channels with varying wavelengths to the closest cluster mean.
    """
    wls = [float(ch_name.split(" ")[1]) for ch_name in ch_names]
    wl_means = np.round(cluster_means(wls)).astype(int)
    new_ch_names = []
    for ci in range(len(ch_names)):
        ch_name = ch_names[ci]
        closest_wl = min(wl_means, key=lambda x: abs(x - wls[ci]))
        new_ch_name = f"{ch_name.split(' ')[0]} {closest_wl}"
        new_ch_names.append(new_ch_name)
    return new_ch_names


def format_channel_name(ch_name: str) -> str:
    ch_name = ch_name.replace(".0", "")
    ch_name = ch_name.replace("nm", "")
    ch_name = ch_name.replace("[", "")
    ch_name = ch_name.replace("]", "")
    ch_name = ch_name.replace("-", "_")
    ch_name = ch_name.replace("Rx", "D")
    ch_name = ch_name.replace("Tx", "S")
    
    s_match = re.search(r"S(\d+)", ch_name)
    d_match = re.search(r"D(\d+)", ch_name)
    wl_match = re.search(r"(\d+)$", ch_name)
    if s_match and d_match and wl_match:
        s = f"S{s_match.group(1)}"
        d = f"D{d_match.group(1)}"
        wl = wl_match.group(1)
        return f"{s}_{d} {wl}"
    else:
        raise ValueError(f"Could not format channel name: {ch_name}")


def format_sd_ids(ch_names: List[str]) -> List[str]:
    """Formats S and D ids to integers. e.g. S1_D1, S1a_D1, S2_D2 -> S1_D1, S3_D1, S2_D2"""
    sd_pairs = [ch_name.split(" ")[0] for ch_name in ch_names]
    ch_types = [ch_name.split(" ")[1] for ch_name in ch_names]
    sources = np.array([sd_pair.split("_")[0][1:] for sd_pair in sd_pairs])
    detectors = np.array([sd_pair.split("_")[1][1:] for sd_pair in sd_pairs])

    def remap_ids(ids: List[str]) -> List[str]:
        valid_ix = np.array([id.isdigit() for id in ids])
        valid_ids = ids[valid_ix]
        available_ids = np.setdiff1d(np.arange(1, len(ids) + 1), valid_ids.astype(int))
        invalid_ids = list(dict.fromkeys(ids[~valid_ix]))
        rename_map = dict(zip(invalid_ids, available_ids))
        for i in range(len(ids)):
            if ids[i] in rename_map:
                ids[i] = rename_map[ids[i]]
        return ids

    sources = remap_ids(sources)
    detectors = remap_ids(detectors)
    return [
        f"S{s}_D{d} {ch_type}" for ch_type, s, d in zip(ch_types, sources, detectors)
    ]


def chs_to_mne_format(raw: mne.io.Raw) -> mne.io.Raw:
    """Attempts to format the channel names to MNE format.

    1. Format individual channel names to MNE format. E.g. Rx1_Tx2 [760nm] -> S2_D1 760
    2. Format SD ids to integers. E.g. S1a_D1 -> S1_D1.
    3. Rename channels with varying wavelengths to the closest cluster mean. E.g. 849, 851 -> 850
    """
    old_names = [ch["ch_name"] for ch in raw.info["chs"]]
    new_names = [format_channel_name(old_name) for old_name in old_names]
    new_names = format_sd_ids(new_names)
    new_names = handle_varying_wls(new_names)
    ch_name_map = dict(zip(old_names, new_names))
    raw.rename_channels(ch_name_map)
    raw.reorder_channels(to_mne_ch_order(raw.info["chs"]))
    return raw


def set_channel_types_and_coils(raw: mne.io.Raw, ch_type_map: dict) -> mne.io.Raw:
    raw.set_channel_types(ch_type_map)
    for ch in raw.info["chs"]:
        typ = ch["ch_name"]
        t = ch_type_map.get(typ, None)
        if t and t in CH_TYPE_TO_COIL_TYPE:
            ch["kind"] = (
                FIFF.FIFFV_FNIRS_CH
                if t.startswith("fnirs") or t in ("hbo", "hbr")
                else ch["kind"]
            )
            ch["unit"] = (
                FIFF.FIFF_UNIT_MOL
                if t.startswith("fnirs") or t in ("hbo", "hbr")
                else ch["unit"]
            )
            ch["coil_type"] = CH_TYPE_TO_COIL_TYPE[t]
            try:
                wl = float(ch["ch_name"].split()[-1])
                ch["loc"][9] = wl
            except Exception:
                pass
    return raw


def is_split_ordered(chs):
    """
    Returns True if the channels are ordered such that all channels of one type (e.g., all 760 nm) are first, followed by all channels of the other type (e.g., all 850 nm), and for each i, chs[i] and chs[i+midway] have the same location.

    Returns:
        loc_matches: bool, True if the locations match between channel pairs
        name_matches: bool, True if the names match between channel pairs
    """
    n = len(chs)
    if n % 2 != 0:
        return False
    midway = n // 2
    loc_matches = []
    name_matches = []
    for i in range(midway):
        loc1 = np.round(chs[i]["loc"][:3], 8)
        loc2 = np.round(chs[i + midway]["loc"][:3], 8)
        loc_matches.append(np.allclose(loc1, loc2))
        ch1_sd, ch1_type = chs[i]["ch_name"].split(" ")
        ch2_sd, ch2_type = chs[i + midway]["ch_name"].split(" ")
        name_matches.append(ch1_sd == ch2_sd and ch1_type != ch2_type)
    return all(loc_matches), all(name_matches)


def validate_chs(chs):
    """MNE expects the channels to be in a specific order. This function checks that the channels are in the correct order and that the locations match between channel pairs."""
    ch_names = [ch["ch_name"] for ch in chs]
    if not validate_ch_names(ch_names):
        return False
    if not validate_ch_locations(chs):
        return False
    return True


def validate_ch_names(ch_names):
    """Expects the correct order of channels."""
    if isinstance(ch_names[0], dict):
        ch_names = [ch["ch_name"] for ch in ch_names]
    for ci in range(len(ch_names) // 2):
        ch1 = ch_names[ci * 2]
        ch2 = ch_names[ci * 2 + 1]
        ch1_loc, ch1_type = ch1.split(" ")
        ch2_loc, ch2_type = ch2.split(" ")
        if ch1_loc != ch2_loc:
            logger.debug(
                f"Consecutive channels have a different SD pair: {ch1_loc} != {ch2_loc}"
            )
            return False
        if ch1_type == ch2_type:
            logger.debug(f"Consecutive channels have same type: {ch1_type}")
            return False
    return True


def validate_ch_locations(chs: List[Dict]) -> bool:
    """Expects the correct order of channels."""
    # validate ch pair locations match
    for ci in range(len(chs) // 2):
        ch1 = chs[ci * 2]
        ch2 = chs[ci * 2 + 1]
        ch1_loc = np.array(ch1["loc"])[:3]
        ch2_loc = np.array(ch2["loc"])[:3]
        if not np.allclose(ch1_loc, ch2_loc):
            logger.debug(
                f"Consecutive channels have a different location: {ch1_loc} != {ch2_loc}"
            )
            return False
    # validate sensor-detector distance sensibility
    for ch in chs:
        ch_loc = ch["loc"][:3]
        s_pos = ch["loc"][3:6]
        d_pos = ch["loc"][6:9]
        dist = np.linalg.norm(s_pos - d_pos)
        if dist > 0.06:
            logger.debug(f"{ch['ch_name']}: sensor-detector distance is too large ({dist}m)")
            return False
    return True


def get_sensor_locs(raw: mne.io.Raw):
    """Returns a dictionary of sensor locations."""
    chs = raw.info["chs"]
    sensors = defaultdict(set)
    for ch in chs:
        sname, dname = ch["ch_name"].split(" ")[0].split("_")
        sensors[sname].add(tuple(ch["loc"][3:6]))
        sensors[dname].add(tuple(ch["loc"][6:9]))
    for sname in sensors:
        if len(sensors[sname]) > 1:
            logger.warning(f"Sensor {sname} has multiple locations: {sensors[sname]}")
    return sensors

def reorder_chs(chs, new_order):
    """Reorder channels according to new_order (list of channel names). Returns reordered channel list."""
    name2ch = {ch["ch_name"]: ch for ch in chs}
    return [name2ch[name] for name in new_order]


def to_mne_ch_order(chs):
    """Order channels by sensor-detector pair and type, i.e.['ch1 wl1', 'ch1 wl2', 'ch2 wl1', 'ch2 wl2']. Preserves ch order based on first appearance."""
    ch_names = [ch["ch_name"] for ch in chs]
    ch_types = list(dict.fromkeys([ch["ch_name"].split(" ")[1] for ch in chs]))
    sd_pairs = list(dict.fromkeys([ch_name.split(" ")[0] for ch_name in ch_names]))
    assert len(sd_pairs) * len(ch_types) == len(
        ch_names
    ), f"Expected {len(sd_pairs) * len(ch_types)} channels, got {len(ch_names)}"
    return [f"{sd_pair} {ch_type}" for sd_pair in sd_pairs for ch_type in ch_types]


def get_split_ordered_ch_names(chs):
    """
    Order channels by type and sensor-detector pair,
    i.e. ['ch1 wl1', 'ch2 wl1', ..., 'chN wl1', 'ch1 wl2', 'ch2 wl2', ..., 'chN wl2'].
    Preserves ch order based on first appearance.
    """
    ch_names = [ch["ch_name"] for ch in chs]
    sd_pairs = list(dict.fromkeys([ch_name.split(" ")[0] for ch_name in ch_names]))
    ch_types = list(dict.fromkeys([ch_name.split(" ")[1] for ch_name in ch_names]))
    assert len(sd_pairs) * len(ch_types) == len(
        ch_names
    ), f"Expected {len(sd_pairs) * len(ch_types)} channels, got {len(ch_names)}"
    return [f"{sd_pair} {ch_type}" for ch_type in ch_types for sd_pair in sd_pairs]


def get_ch_names_ordered_by_location(chs):
    """
    Returns a list of channel names in the order of the locations.
    Very risky. Assumes that the first appearance of a channel location is of the type that first appears in channels.
    """
    new_ch_names = np.empty(len(chs), dtype=object)
    ch_locs = np.array([ch["loc"][:3] for ch in chs])
    ch_names = np.array([ch["ch_name"] for ch in chs])
    sd_pairs = list(dict.fromkeys([ch_name.split(" ")[0] for ch_name in ch_names]))
    ch_types = list(dict.fromkeys([ch_name.split(" ")[1] for ch_name in ch_names]))
    _, idx = np.unique(ch_locs, axis=0, return_index=True)
    ch_locs_unique = ch_locs[np.sort(idx)]
    for i, ch_loc in enumerate(ch_locs_unique):
        sd_ix = np.where((ch_loc == ch_locs).all(axis=1))[0]
        assert len(sd_ix) == len(ch_types)
        for j, ix in enumerate(sd_ix):
            new_ch_names[ix] = f"{sd_pairs[i]} {ch_types[j]}"
    return list(new_ch_names)


def try_to_correct_ch_order(raw):

    def apply_and_validate(new_names):
        raw_c = raw.copy()
        raw_c.rename_channels(dict(zip(raw.ch_names, new_names)))
        raw_c.reorder_channels(to_mne_ch_order(raw_c.info["chs"]))
        return raw_c if validate_chs(raw_c.info["chs"]) else None

    loc_match, name_match = is_split_ordered(raw.info["chs"])
    if loc_match and not name_match:
        logger.debug(
            "BIDS channel locations are split ordered but channel names are not. Correcting."
        )
        corrected = apply_and_validate(get_split_ordered_ch_names(raw.info["chs"]))
        if corrected:
            logger.debug("Split-ordered channel correction successful.")
            return corrected
        else:
            logger.debug("Split-ordered channel correction failed.")

    logger.debug("Trying to correct channel order by location.")
    corrected = apply_and_validate(get_ch_names_ordered_by_location(raw.info["chs"]))
    if corrected:
        logger.debug("Location-ordered channel correction successful.")
        return corrected
    else:
        logger.debug("Location-ordered channel correction failed.")

    logger.warning("Failed to correct channel order.")
    return raw


def ch_constants_to_mne_format(chs):
    for ch in chs:
        ch["coil_type"] = INT_TO_COIL_TYPE[int(ch["coil_type"])]
        ch["kind"] = INT_TO_KIND[int(ch["kind"])]
        ch["unit"] = INT_TO_UNIT[int(ch["unit"])]
    return chs
