"""Tests for the Medvedeva2026FnirsStroke dataset."""

from collections import Counter
from pathlib import Path

import mne
import numpy as np
import pytest

from fnirsatlas.datasets.medvedeva2026_fnirs_stroke import Medvedeva2026FnirsStroke

SUBJECTS = list(range(1, 17))
SFREQ = 13.56336784362793
# Subject 2, session 2, source run "right1" is the second recorded run of the
# session. Right flashes prompt right-hand presses and left flashes are ignored.
SUBJECT = 2
SESSION = "2fnirs"
RUN = "1right1"
SOURCE = "R_02_session2_right1.fif"
SOURCE_LABELS = {"flash_r": "right_hand", "flash_l": "ignore_left"}


def _stim_labels(dataset, raw):
    names = {code: name for name, code in dataset.event_id.items()}
    events = mne.find_events(raw, stim_channel="Stim", verbose=False)
    return events[:, 0], [names[code] for code in events[:, 2]]


@pytest.fixture(scope="module")
def dataset():
    return Medvedeva2026FnirsStroke()


@pytest.fixture(scope="module")
def raw(dataset):
    return dataset.get_data(subjects=[SUBJECT])[SUBJECT][SESSION][RUN]


@pytest.mark.data
def test_labels_follow_source_flashes(dataset, raw):
    source = mne.io.read_raw_fif(
        Path(dataset.data_path(SUBJECT)) / SOURCE, verbose="error"
    )
    expected = [
        (source.time_as_index(onset, use_rounding=True)[0], SOURCE_LABELS[flash])
        for onset, flash in zip(
            source.annotations.onset, source.annotations.description, strict=True
        )
    ]
    samples, names = _stim_labels(dataset, raw)
    assert list(zip(samples, names, strict=True)) == expected


@pytest.mark.data
def test_raw_properties(raw):
    assert raw.info["sfreq"] == pytest.approx(SFREQ)
    picks = mne.pick_types(raw.info, fnirs="fnirs_od")
    assert len(picks) == 140
    assert "Stim" in raw.ch_names
    # Wavelengths of each pair are adjacent, 760 nm first.
    names = [raw.ch_names[pick] for pick in picks]
    assert all(name.endswith(" 760") for name in names[::2])
    assert [name[:-4] for name in names[::2]] == [name[:-4] for name in names[1::2]]
    assert np.isfinite(raw.get_data(picks=picks)).all()
    assert not np.allclose(raw.get_data(picks=picks), 0)
    assert not np.isnan([raw.info["chs"][pick]["loc"][:9] for pick in picks]).any()


@pytest.mark.data
@pytest.mark.parametrize("subject", SUBJECTS)
def test_all_subjects_load(dataset, subject):
    sessions = dataset.get_data(subjects=[subject])[subject]
    assert 3 <= len(sessions) <= 6
    counts = Counter()
    for runs in sessions.values():
        assert len(runs) >= 2
        for raw in runs.values():
            counts.update(_stim_labels(dataset, raw)[1])
    # Left-hand runs ignore right flashes and right-hand runs ignore left ones.
    # Two source runs miss one flash.
    assert abs(counts["ignore_right"] - counts["left_hand"]) <= 1
    assert abs(counts["ignore_left"] - counts["right_hand"]) <= 1
    assert counts["left_hand"] > 0
    assert counts["right_hand"] > 0
