"""Tests for shared fNIRS preprocessing operations."""

from functools import partial

import mne
import numpy as np
import pytest
from moabb.analysis.results import get_digest

from fnirsatlas.datasets.preprocessing import (
    _crop_leading_clipped_samples,
    _interpolate_bad_segments_temporal,
    _mark_bad_segments_by_hampel,
    _promote_bad_segments_to_bad_channels,
    bad_data_mask,
    beer_lambert_law,
    drop_bad_epochs_step,
    drop_recording_with_too_many_bads,
    interpolate_bads_step,
    interpolate_dist,
    interpolate_nearest,
    interpolate_zero,
    mark_bad_segments_by_flat,
    mark_bad_segments_by_invalid_values,
    mark_bad_segments_by_psp,
    mark_bad_segments_by_sci,
    mark_bad_segments_by_snr,
    preprocess_fnirs,
)


def test_mark_bad_segments_by_hampel():
    data = np.arange(22, dtype=float)
    data[10] = -1_000.0
    raw = mne.io.RawArray(
        np.vstack([data, np.arange(22)]),
        mne.create_info(["S1_D1 hbo", "Stim"], 1, ["hbo", "stim"]),
        verbose=False,
    )

    out = _mark_bad_segments_by_hampel(raw)

    assert out is raw
    assert np.array_equal(out.get_data(picks="S1_D1 hbo"), [data])
    assert np.array_equal(out.get_data(picks="stim"), [np.arange(22)])
    assert out.annotations.description.tolist() == ["BAD_hampel"]
    assert out.annotations.onset.tolist() == [10]
    assert out.annotations.duration.tolist() == [1]
    assert out.annotations.ch_names.tolist() == [("S1_D1 hbo",)]


def test_mark_bad_segments_by_invalid_values():
    """NaN in HbO, zero and negative CW amplitude are invalid. Negative HbO is not."""
    hbo = np.arange(8, dtype=float) - 4
    hbo[[2, 3, 6]] = np.nan
    cw = np.arange(8, dtype=float) + 1
    cw[[5]] = 0.0
    cw[[7]] = -1.0
    raw = mne.io.RawArray(
        np.vstack([hbo, cw, np.arange(8)]),
        mne.create_info(
            ["S1_D1 hbo", "S1_D1 760", "Stim"], 2, ["hbo", "fnirs_cw_amplitude", "stim"]
        ),
        verbose=False,
    )
    # Onsets stay on the acquisition clock after cropping.
    raw.crop(tmin=0.5)

    out = mark_bad_segments_by_invalid_values(raw)

    assert out is raw
    spans = [
        (annotation["onset"], annotation["duration"], annotation["ch_names"])
        for annotation in out.annotations
    ]
    assert set(out.annotations.description) == {"BAD_invalid"}
    assert sorted(spans) == [
        (1.0, 1.0, ("S1_D1 hbo",)),
        (2.5, 0.5, ("S1_D1 760",)),
        (3.0, 0.5, ("S1_D1 hbo",)),
        (3.5, 0.5, ("S1_D1 760",)),
    ]


def test_crop_leading_clipped_samples_keeps_from_first_unclipped_sample():
    # Samples 0-2 have every channel at a bound, sample 4 is clipped but not leading.
    data = np.array([[0, 9, 0, 5, 0], [9, 9, 0, 5, 9]], dtype=float)
    raw = mne.io.RawArray(
        data,
        mne.create_info(["S1_D1 760", "S1_D1 850"], 1, "fnirs_cw_amplitude"),
        verbose=False,
    )

    out = _crop_leading_clipped_samples(raw, floor=0, ceiling=9)

    assert out.get_data().tolist() == [[5, 0], [5, 9]]


def test_promote_bad_segments_to_bad_channels_marks_whole_groups():
    raw = mne.io.RawArray(
        np.ones((3, 22)),
        mne.create_info(["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo"], 1, "hbo"),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations([9], [2], ["BAD_hampel"], ch_names=[["S1_D1 hbo"]])
    )

    out = _promote_bad_segments_to_bad_channels(raw)

    assert out is raw
    assert out.info["bads"] == ["S1_D1 hbo", "S1_D1 hbr"]
    assert out.annotations.description.tolist() == ["BAD_hampel"]


def test_interpolate_bad_segments_temporal():
    short = np.arange(22, dtype=float)
    short[[0, 9]] = -1_000
    long = np.arange(22, dtype=float)
    long[14:17] = -1_000
    long[21] = -1_000
    raw = mne.io.RawArray(
        np.vstack([short, long, np.arange(22)]),
        mne.create_info(["S1_D1 hbo", "S2_D2 hbo", "Stim"], 1, ["hbo", "hbo", "stim"]),
        verbose=False,
    )
    raw.set_annotations(
        mne.Annotations(
            onset=[0, 9, 14, 19, 21],
            duration=[1, 1, 3, 1, 1],
            description=["BAD_hampel"] * 3 + ["BAD_other", "BAD_hampel"],
            ch_names=[["S1_D1 hbo"], ["S1_D1 hbo"], ["S2_D2 hbo"], [], ["S2_D2 hbo"]],
        )
    )

    out = _interpolate_bad_segments_temporal(raw)

    assert out is raw
    # Interior gaps are linear, edge gaps copy the nearest sample.
    expected_short = np.arange(22, dtype=float)
    expected_short[0] = 1
    expected_long = np.arange(22, dtype=float)
    expected_long[21] = 20
    assert np.array_equal(out.get_data(picks="S1_D1 hbo"), [expected_short])
    assert np.array_equal(out.get_data(picks="S2_D2 hbo"), [expected_long])
    assert np.array_equal(out.get_data(picks="stim"), [np.arange(22)])
    # Only segments longer than max_samples and non-channel annotations remain.
    assert out.annotations.onset.tolist() == [14, 19]
    assert out.annotations.description.tolist() == ["BAD_hampel", "BAD_other"]


def _snr_raw(data):
    names = ["S1_D1 760", "S2_D2 850", "S1_D1 850", "S2_D2 760"]
    return mne.io.RawArray(
        np.asarray(data, dtype=float),
        mne.create_info(names, 10, ["fnirs_cw_amplitude"] * 4),
        verbose=False,
    )


PASSING_SNR = [19.0, 21.0, 19.0, 21.0]
FAILING_SNR = [9.0, 11.0, 9.0, 11.0]
S1_D1 = ["S1_D1 850"]


@pytest.mark.parametrize(
    ("third_channel", "bads"),
    [
        (PASSING_SNR, []),
        # SNR equal to the threshold fails.
        (FAILING_SNR, [(0.0, S1_D1), (0.2, S1_D1)]),
        # Only the failing window is marked.
        ([19.0, 21.0, 9.0, 11.0], [(0.2, S1_D1)]),
        ([1.0] * 4, [(0.0, S1_D1), (0.2, S1_D1)]),
        ([np.nan, 1.0, 19.0, 21.0], [(0.0, S1_D1)]),
        ([19.0, 21.0, np.inf, 1.0], [(0.2, S1_D1)]),
    ],
)
def test_mark_bad_segments_by_snr(third_channel, bads):
    """Two 0.2 s windows of two samples each."""
    raw = _snr_raw([PASSING_SNR, PASSING_SNR, third_channel, PASSING_SNR])

    out = mark_bad_segments_by_snr(raw, threshold=10, window=0.2)

    assert out is raw
    marked = {}
    for annotation in out.annotations:
        assert annotation["description"] == "BAD_snr"
        assert annotation["duration"] == pytest.approx(0.2)
        marked.setdefault(round(annotation["onset"], 6), []).extend(
            annotation["ch_names"]
        )
    assert [(onset, sorted(names)) for onset, names in marked.items()] == bads


def test_mark_bad_segments_by_flat():
    """Two 1 s windows at 10 Hz. Relative SD 1e-8 is flat, 1e-4 is not."""
    rng = np.random.default_rng(0)
    data = 1 + rng.normal(0, 1e-4, (4, 20))
    data[2, 10:] = 1 + rng.normal(0, 1e-8, 10)

    out = mark_bad_segments_by_flat(_snr_raw(data), threshold=1e-6, window=1.0)

    spans = [
        (annotation["onset"], annotation["duration"], annotation["ch_names"])
        for annotation in out.annotations
    ]
    assert spans == [(1.0, 1.0, ("S1_D1 850",))]
    assert out.annotations.description.tolist() == ["BAD_flat"]


def test_interpolate_nearest_uses_nearest_matching_chromophore():
    info = mne.create_info(
        ["S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr"],
        10,
        ["hbo", "hbr", "hbo", "hbr"],
    )
    raw = mne.io.RawArray(
        np.array([[0.0] * 10, [1.0] * 10, [2.0] * 10, [3.0] * 10]),
        info,
        verbose=False,
    )
    for channel in raw.info["chs"][:2]:
        channel["loc"][:3] = (0.01, 0.0, 0.0)
    for channel in raw.info["chs"][2:]:
        channel["loc"][:3] = (0.03, 0.0, 0.0)
    raw.info["bads"] = ["S1_D1 hbo", "S1_D1 hbr"]

    out = interpolate_nearest(raw)

    assert np.array_equal(out.get_data(picks="S1_D1 hbo"), [[2.0] * 10])
    assert np.array_equal(out.get_data(picks="S1_D1 hbr"), [[3.0] * 10])
    assert out.info["bads"] == []


def test_beer_lambert_law_uses_three_cm_when_distances_are_missing(caplog):
    info = mne.create_info(["S1_D1 760", "S1_D1 850"], 10, ["fnirs_od", "fnirs_od"])
    info["chs"][0]["loc"][9] = 760
    info["chs"][1]["loc"][9] = 850
    raw = mne.io.RawArray(np.arange(40).reshape(2, 20), info, verbose=False)
    expected_raw = raw.copy()
    for channel in expected_raw.info["chs"]:
        channel["loc"][3:6] = 0.0
        channel["loc"][6:9] = (0.03, 0.0, 0.0)
    expected = mne.preprocessing.nirs.beer_lambert_law(expected_raw)
    out = beer_lambert_law(raw)

    assert np.allclose(out.get_data(), expected.get_data())
    assert np.isnan(out.info["chs"][0]["loc"][3:9]).all()
    assert "using fallback distance of 0.03 m" in caplog.text


@pytest.mark.parametrize(
    "option", [{"snr_threshold": 40}, {"max_std": {"hbo": 0.75, "hbr": 0.25}}]
)
def test_preprocessing_options_change_the_pipeline_digest(option):
    """MOABB keys its epoch cache on the pipeline repr, so options must show there."""
    assert get_digest(preprocess_fnirs()) != get_digest(preprocess_fnirs(**option))


def _noisy_segment_raw() -> mne.io.RawArray:
    """Four pairs with 0.1 uM HbO and 0.03 uM HbR noise at 5 Hz, S2_D2 hbr 20x at 15-30 s."""
    ch_names = [f"S{i}_D{i} {kind}" for i in range(4) for kind in ("hbo", "hbr")]
    info = mne.create_info(ch_names, 5, [name.split()[-1] for name in ch_names])
    scale = np.array([1e-7 if name.endswith("hbo") else 3e-8 for name in ch_names])
    data = np.random.default_rng(0).normal(size=(len(ch_names), 1000)) * scale[:, None]
    data[ch_names.index("S2_D2 hbr"), 75:150] *= 20
    return mne.io.RawArray(data, info, verbose=False)


def test_std_check_annotates_segments_of_good_channels_above_the_limit():
    pipeline = preprocess_fnirs(
        snr_threshold=None,
        tddr=False,
        l_freq=None,
        h_freq=None,
        max_std={"hbo": 0.75, "hbr": 0.25},
        quality_window=5.0,
    )
    raw = _noisy_segment_raw()
    raw._data[raw.ch_names.index("S1_D1 hbo")] *= 100
    raw.info["bads"] = ["S1_D1 hbo", "S1_D1 hbr"]

    out = pipeline.fit_transform(raw)

    assert [
        (a["onset"], a["duration"], a["description"], tuple(a["ch_names"]))
        for a in out.annotations
    ] == [
        (15.0, 5.0, "BAD_std", ("S2_D2 hbr",)),
        (20.0, 5.0, "BAD_std", ("S2_D2 hbr",)),
        (25.0, 5.0, "BAD_std", ("S2_D2 hbr",)),
    ]


@pytest.mark.parametrize(
    ("bad", "max_bad_fraction", "min_good_pairs", "kept"),
    [
        # A 0.1 s segment on one channel marks its group, 2 of 4 channels, in epoch 0.
        ({"segment": 0.1}, 0.5, None, [0, 1, 2]),
        ({"segment": 0.1}, 0.4, None, [1, 2]),
        # Over the fraction limit, an epoch is kept with enough good pairs left.
        ({"segment": 0.1}, 0.4, 1, [0, 1, 2]),
        ({"segment": 0.1}, 0.4, 2, [1, 2]),
        ({"segment": 0.1}, 0.5, 2, [0, 1, 2]),
        ({"bads": ["S1_D1 760", "S1_D1 850"]}, 0.5, None, [0, 1, 2]),
        # One bad wavelength makes its pair bad.
        ({"bads": ["S1_D1 760", "S1_D1 850", "S2_D2 760"]}, 0.5, 1, None),
        # Without epochs left, the run is dropped.
        ({"bads": ["S1_D1 760", "S1_D1 850"]}, 0.4, None, None),
    ],
)
def test_drop_bad_epochs(bad, max_bad_fraction, min_good_pairs, kept):
    """Three 1 s epochs of two 760/850 nm groups at 10 Hz, from a cropped raw."""
    names = [f"{group} {wl}" for group in ("S1_D1", "S2_D2") for wl in (760, 850)]
    info = mne.create_info(names, 10, "fnirs_cw_amplitude")
    raw = mne.io.RawArray(np.ones((4, 40)), info, verbose=False).crop(tmin=0.5)
    if "segment" in bad:
        raw.annotations.append(
            raw.first_time, bad["segment"], "BAD_test", [["S1_D1 760"]]
        )
    raw.info["bads"] = bad.get("bads", [])
    events = np.array([[sample, 0, 1] for sample in (5, 15, 25)])
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=0.9,
        baseline=None,
        reject_by_annotation=False,
        preload=True,
        verbose=False,
    )

    out = drop_bad_epochs_step(max_bad_fraction, min_good_pairs).transform(epochs)

    assert (None if out is None else out.selection.tolist()) == kept


@pytest.mark.parametrize(
    ("bad", "max_bad_fraction", "kept"),
    [
        # A 0.5 s segment on one channel marks 2 of 4 channels for half of 1 s.
        ({"segment": 0.5}, 0.25, True),
        ({"segment": 0.5}, 0.2, False),
        ({"bads": ["S1_D1 760", "S1_D1 850"]}, 0.5, True),
        ({"bads": ["S1_D1 760", "S1_D1 850"]}, 0.4, False),
    ],
)
def test_drop_recording_with_too_many_bads(bad, max_bad_fraction, kept):
    raw = _snr_raw([[1.0] * 10] * 4)
    if "segment" in bad:
        raw.annotations.append(0.0, bad["segment"], "BAD_test", [["S1_D1 760"]])
    raw.info["bads"] = bad.get("bads", [])

    out = drop_recording_with_too_many_bads(raw, max_bad_fraction=max_bad_fraction)

    assert (out is raw) == kept


def _cw_raw() -> mne.io.RawArray:
    """Six 760/850 nm groups at 10 Hz for 120 s sharing a 1 Hz cardiac wave.

    S1_D1 and S10_D10 are clean, except S1_D1 760 is independent noise during
    20-30 s.

    S3_D3 760 is negative during 0-24 s, S4_D4 760 has SNR 2, S5_D5
    has independent noise per wavelength and S6_D6 is a 1 cm short channel.
    """
    rng = np.random.default_rng(0)
    times = np.arange(1200) / 10
    groups = [f"S{i}_D{i}" for i in (1, 10, 3, 4, 5, 6)]
    names = [f"{group} {wl}" for group in groups for wl in (760, 850)]
    data = 1 + 0.01 * np.sin(2 * np.pi * times) + rng.normal(0, 1e-3, (12, 1200))
    data[1, 200:300] = 1 + rng.normal(0, 0.01, 100)
    data[4, :240] = -1.0
    data[6] = 1 + rng.normal(0, 0.5, 1200)
    data[8:10] = 1 + rng.normal(0, 0.01, (2, 1200))
    info = mne.create_info(names, 10, "fnirs_cw_amplitude")
    for index, channel in enumerate(info["chs"]):
        group, distance = index // 2, 0.01 if index >= 10 else 0.03
        channel["loc"][3:6] = (0.1 * group, 0.0, 0.0)
        channel["loc"][6:9] = (0.1 * group + distance, 0.0, 0.0)
        channel["loc"][:3] = (0.1 * group + distance / 2, 0.0, 0.0)
        channel["loc"][9] = (760, 850)[index % 2]
    return mne.io.RawArray(data, info, verbose=False)


def _bad_descriptions(raw):
    bad = {}
    for annotation in raw.annotations:
        for name in annotation["ch_names"]:
            bad.setdefault(name, set()).add(annotation["description"])
    return bad


@pytest.mark.parametrize("check", [mark_bad_segments_by_sci, mark_bad_segments_by_psp])
def test_coupling_checks_use_every_wavelength(check):
    """Three wavelengths at 10 Hz for 20 s. S2_D2 850 is flat."""
    rng = np.random.default_rng(0)
    times = np.arange(200) / 10
    names = [f"S{i}_D{i} {wl}" for i in (1, 2) for wl in (730, 808, 850)]
    data = 0.01 * np.sin(2 * np.pi * times) + rng.normal(0, 1e-3, (6, 200))
    data[5] = 0.0
    raw = mne.io.RawArray(data, mne.create_info(names, 10, "fnirs_od"), verbose=False)

    out = check(raw, threshold=0.1, window=10.0)

    assert {name for names in out.annotations.ch_names for name in names} == set(
        names[3:]
    )
    assert out.annotations.onset.tolist() == [0.0, 10.0]


@pytest.mark.parametrize(
    "threshold, failing", [(0.45, {"S2_D2", "S3_D3"}), (0.55, None)]
)
def test_mark_bad_segments_by_psp(threshold, failing):
    """S1_D1 is an ideal 1 Hz pulse in both wavelengths, PSP 0.5 (PHOEBE).

    S2_D2 has identical noise in both wavelengths. S3_D3 is a 0.3 Hz
    oscillation below the cardiac band.
    """
    rng = np.random.default_rng(0)
    times = np.arange(200) / 10
    noise = rng.normal(0, 1e-3, 200)
    slow = np.sin(2 * np.pi * 0.3 * times)
    data = [np.sin(2 * np.pi * times)] * 2 + [1 + noise] * 2 + [slow] * 2
    groups = ("S1_D1", "S2_D2", "S3_D3")
    names = [f"{group} {wl}" for group in groups for wl in (760, 850)]
    raw = mne.io.RawArray(data, mne.create_info(names, 10, "fnirs_od"), verbose=False)

    out = mark_bad_segments_by_psp(raw, threshold=threshold, window=10.0)

    bad = {name.split()[0] for names in out.annotations.ch_names for name in names}
    assert bad == (failing or set(groups))


@pytest.mark.parametrize("check", [mark_bad_segments_by_sci, mark_bad_segments_by_psp])
@pytest.mark.parametrize("sfreq, pulse", [(10, 2.2), (3.9, 1.6)])
def test_coupling_checks_keep_fast_pulses(check, sfreq, pulse):
    """A pulse up to 150 bpm, or up to the Nyquist limit, passes."""
    rng = np.random.default_rng(0)
    times = np.arange(int(20 * sfreq)) / sfreq
    data = np.sin(2 * np.pi * pulse * times) + rng.normal(0, 0.1, (2, len(times)))
    info = mne.create_info(["S1_D1 760", "S1_D1 850"], sfreq, "fnirs_od")
    raw = mne.io.RawArray(data, info, verbose=False)

    out = check(raw, threshold=0.1, window=10.0)

    assert len(out.annotations) == 0


@pytest.mark.parametrize("check", [mark_bad_segments_by_sci, mark_bad_segments_by_psp])
def test_coupling_checks_skip_sampling_without_cardiac_band(check, caplog):
    info = mne.create_info(["S1_D1 760", "S1_D1 850"], 2.5, "fnirs_od")
    raw = mne.io.RawArray(np.ones((2, 50)), info, verbose=False)

    out = check(raw)

    assert len(out.annotations) == 0
    assert "too low for the cardiac band" in caplog.text


def test_preprocess_fnirs_cw_pipeline():
    out = preprocess_fnirs(short_channel_threshold=0.02).fit_transform(_cw_raw())

    assert out.get_channel_types() == ["hbo", "hbr"] * 5
    assert not any(name.startswith("S6_D6") for name in out.ch_names)
    assert np.isfinite(out.get_data()).all()
    # Annotations follow the channel renaming.
    bad = _bad_descriptions(out)
    assert not any(name.startswith("S10_D10") for name in bad)
    assert "BAD_invalid" in bad["S3_D3 hbo"]
    assert "BAD_snr" in bad["S4_D4 hbo"]
    for kind in ("hbo", "hbr"):
        assert bad[f"S5_D5 {kind}"] == {"BAD_sci", "BAD_psp"}
    # A bad channel makes its whole source-detector group bad.
    for description, group in (("BAD_invalid", "S3_D3"), ("BAD_snr", "S4_D4")):
        rows = [i for i, name in enumerate(out.ch_names) if name.startswith(group)]
        assert bad_data_mask(out, [description])[rows].any(axis=1).all()
    # Windowed checks mark only the failing 10 s window.
    windows = [
        (annotation["onset"], annotation["duration"], annotation["description"])
        for annotation in out.annotations
        if "S1_D1 hbo" in annotation["ch_names"]
    ]
    assert sorted(windows) == [(20.0, 10.0, "BAD_psp"), (20.0, 10.0, "BAD_sci")]
    # S3_D3 760 is negative during 0-24 s.
    invalid = [
        (annotation["onset"], annotation["duration"])
        for annotation in out.annotations
        if annotation["description"] == "BAD_invalid"
        and annotation["ch_names"] == ("S3_D3 hbo",)
    ]
    assert invalid == [(0.0, 24.0)]


def _hb_raw_with_positions(positions, data):
    """Raw with one hbo/hbr pair per position, hbo first."""
    names = []
    types = []
    for index in range(len(positions)):
        names += [f"S{index + 1}_D1 hbo", f"S{index + 1}_D1 hbr"]
        types += ["hbo", "hbr"]
    raw = mne.io.RawArray(
        np.array(data, dtype=float), mne.create_info(names, 10, types), verbose=False
    )
    for index, position in enumerate(positions):
        raw.info["chs"][2 * index]["loc"][:3] = position
        raw.info["chs"][2 * index + 1]["loc"][:3] = position
    return raw


def test_interpolate_dist_weights_nearer_channels_more():
    raw = _hb_raw_with_positions(
        [(0.0, 0, 0), (0.01, 0, 0), (0.05, 0, 0)],
        [[0.0] * 10, [0.0] * 10, [1.0] * 10, [10.0] * 10, [3.0] * 10, [30.0] * 10],
    )
    raw.info["bads"] = ["S1_D1 hbo", "S1_D1 hbr"]

    out = interpolate_dist(raw, temperature=0.01)

    hbo = out.get_data(picks="S1_D1 hbo")[0, 0]
    hbr = out.get_data(picks="S1_D1 hbr")[0, 0]
    # Weights are exp(-d / temperature) over [1 cm, 5 cm].
    weights = np.exp(-np.array([0.01, 0.05]) / 0.01)
    weights /= weights.sum()
    assert np.isclose(hbo, weights @ [1.0, 3.0])
    assert np.isclose(hbr, weights @ [10.0, 30.0])
    assert out.info["bads"] == []


@pytest.mark.parametrize(
    ("positions", "bads", "match"),
    [
        ([(0.0, 0, 0), (0.01, 0, 0)], ["S1_D1 hbo", "S2_D1 hbo"], "Every hbo"),
        ([(0.0, 0, 0), (np.nan,) * 3], ["S1_D1 hbo"], "S2_D1 hbo"),
    ],
)
def test_interpolate_dist_raises_without_good_channels_or_positions(
    positions, bads, match
):
    raw = _hb_raw_with_positions(
        positions, [[0.0] * 10, [0.0] * 10, [1.0] * 10, [10.0] * 10]
    )
    raw.info["bads"] = bads

    with pytest.raises(ValueError, match=match):
        interpolate_dist(raw)


def test_interpolate_zero_zeroes_only_bad_channels():
    raw = _hb_raw_with_positions(
        [(0.0, 0, 0), (0.01, 0, 0)],
        [[5.0] * 10, [6.0] * 10, [1.0] * 10, [10.0] * 10],
    )
    raw.info["bads"] = ["S1_D1 hbo", "S2_D1 hbr"]

    out = interpolate_zero(raw)

    assert np.all(out.get_data(picks=["S1_D1 hbo", "S2_D1 hbr"]) == 0)
    assert np.all(out.get_data(picks="S1_D1 hbr") == 6.0)
    assert np.all(out.get_data(picks="S2_D1 hbo") == 1.0)
    assert out.info["bads"] == []


def test_interpolate_bads_step_none_keeps_bads():
    raw = _hb_raw_with_positions(
        [(0.0, 0, 0), (0.01, 0, 0)],
        [[0.0] * 10, [0.0] * 10, [1.0] * 10, [10.0] * 10],
    )
    raw.info["bads"] = ["S1_D1 hbo"]

    out = interpolate_bads_step(None).transform(raw)

    assert out.info["bads"] == ["S1_D1 hbo"]


def test_interpolate_bads_step_digest_depends_on_method():
    digests = {
        get_digest(interpolate_bads_step(method))
        for method in [
            interpolate_nearest,
            interpolate_zero,
            partial(interpolate_dist, temperature=0.01),
            partial(interpolate_dist, temperature=0.02),
            None,
        ]
    }

    assert len(digests) == 5
