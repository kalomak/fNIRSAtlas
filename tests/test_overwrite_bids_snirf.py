import numpy as np
import mne
from pathlib import Path
from unittest.mock import patch
from fnirsatlas.parsers.bids import BIDSParser
from fnirsatlas.channels import (
    validate_chs,
    reorder_chs,
    to_mne_ch_order,
)


def create_test_raw(ch_names, ch_locs, snirf_path=None):
    """Create a test raw object with specified channels, locations, and standard non-channel data."""
    n_channels = len(ch_names)
    n_times = 1000
    data = np.random.randn(n_channels, n_times)
    ch_types = ["fnirs_od"] * n_channels
    
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=10.0)
    
    # Set channel locations
    for i, (ch, loc) in enumerate(zip(info["chs"], ch_locs)):
        ch["loc"][:3] = loc
    
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Add standard non-channel data (events, metadata)
    raw.info["temp"] = {"event_id": {1: "test"}}
    annotations = mne.Annotations([1.0], [0.5], ["test"])
    raw.set_annotations(annotations)
    
    # Create SNIRF file if path provided (just create empty file - will be mocked)
    if snirf_path:
        Path(snirf_path).touch()
        raw.filenames = [str(snirf_path)]
    else:
        raw.filenames = ["test.snirf"]
    
    return raw


def test_case1_bids_valid_no_override(tmpdir):
    """Case 1: BIDS channels are valid - no override needed."""
    snirf_path = Path(tmpdir) / "test.snirf"
    ch_names = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]
    loc1, loc2 = [0.05, 0.03, 0.13], [0.04, 0.05, 0.13]
    ch_locs = [loc1, loc1, loc2, loc2]  # Matching locations for pairs
    bids_raw = create_test_raw(ch_names, ch_locs, snirf_path)
    result = BIDSParser._maybe_override_bids_chs(bids_raw.copy())
    assert result.ch_names == bids_raw.ch_names
    assert "temp" in result.info
    assert len(result.annotations) == len(bids_raw.annotations)


def test_case2_bids_valid_after_reorder(tmpdir):
    """Case 2: BIDS channel order invalid, but valid after reordering."""
    snirf_path = Path(tmpdir) / "test.snirf"
    ch_names = ["S1_D1 760", "S2_D1 760", "S1_D1 850", "S2_D1 850"]
    loc1, loc2 = [0.05, 0.03, 0.13], [0.04, 0.05, 0.13]
    ch_locs = [loc1, loc2, loc1, loc2]
    bids_raw = create_test_raw(ch_names, ch_locs, snirf_path)
    assert not validate_chs(bids_raw.info["chs"])
    result = BIDSParser._maybe_override_bids_chs(bids_raw.copy())
    mne_order = to_mne_ch_order(bids_raw.info["chs"])
    assert validate_chs(result.info["chs"])
    assert result.ch_names == mne_order
    assert "temp" in result.info
    assert len(result.annotations) == len(bids_raw.annotations)


def test_case3_snirf_valid_override(tmpdir):
    """Case 3: BIDS invalid, SNIRF valid - override with SNIRF."""
    snirf_path = Path(tmpdir) / "test.snirf"
    loc1, loc2 = [0.05, 0.03, 0.13], [0.04, 0.05, 0.13]
    bids_ch_names = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]
    bids_ch_locs = [loc1, loc2, loc1, loc2]
    bids_raw = create_test_raw(bids_ch_names, bids_ch_locs, snirf_path)
    bids_name2ch = {ch["ch_name"]: ch for ch in bids_raw.info["chs"]}
    snirf_raw = create_test_raw(["S1_D1 760", "S2_D1 760", "S1_D1 850", "S2_D1 850"], [loc1, loc2, loc1, loc2], snirf_path)
    with patch("mne.io.read_raw_snirf", return_value=snirf_raw):
        result = BIDSParser._maybe_override_bids_chs(bids_raw.copy())
    snirf_mne_order = to_mne_ch_order(snirf_raw.info["chs"])
    assert result.ch_names == snirf_mne_order
    res_name2ch = {ch["ch_name"]: ch for ch in result.info["chs"]}
    snirf_name2ch = {ch["ch_name"]: ch for ch in snirf_raw.info["chs"]}
    bids_match = []
    for name in snirf_mne_order:
        assert np.allclose(res_name2ch[name]["loc"][:3], snirf_name2ch[name]["loc"][:3])
        bids_match.append(np.allclose(res_name2ch[name]["loc"][:3], bids_name2ch[name]["loc"][:3]))
    assert not all(bids_match)
    assert "temp" in result.info
    assert len(result.annotations) == len(bids_raw.annotations)


def test_case4_snirf_valid_override(tmpdir):
    """Case 4: BIDS invalid, SNIRF invalid - override with SNIRF."""
    snirf_path = Path(tmpdir) / "test.snirf"
    loc1, loc2 = [0.05, 0.03, 0.13], [0.04, 0.05, 0.13]
    bids_ch_names = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]
    bids_ch_locs = [loc1, loc2, loc1, loc2]
    bids_raw = create_test_raw(bids_ch_names, bids_ch_locs, snirf_path)
    snirf_raw = create_test_raw(["S1_D1 760", "S2_D1 760", "S1_D1 850", "S2_D1 850"], [loc1, loc1, loc2, loc2], snirf_path)
    with patch("mne.io.read_raw_snirf", return_value=snirf_raw):
        result = BIDSParser._maybe_override_bids_chs(bids_raw.copy())
    snirf_mne_order = to_mne_ch_order(snirf_raw.info["chs"])
    assert result.ch_names == snirf_mne_order
    res_name2ch = {ch["ch_name"]: ch for ch in result.info["chs"]}
    snirf_name2ch = {ch["ch_name"]: ch for ch in snirf_raw.info["chs"]}
    for name in snirf_mne_order:
        assert np.allclose(res_name2ch[name]["loc"][:3], snirf_name2ch[name]["loc"][:3])
    assert "temp" in result.info
    assert len(result.annotations) == len(bids_raw.annotations)
