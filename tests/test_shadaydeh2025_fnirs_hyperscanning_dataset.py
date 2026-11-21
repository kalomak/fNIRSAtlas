"""Tests for the Shadaydeh2025FnirsHyperscanning dataset."""

import mne

from fnirsatlas.datasets.shadaydeh2025_fnirs_hyperscanning import (
    Shadaydeh2025FnirsHyperscanning,
)


def test_missing_actor_trigger_does_not_break_mapping(monkeypatch):
    raw = mne.io.RawArray([[0.0] * 10], mne.create_info(["stim"], 1, "stim"))
    raw.set_annotations(mne.Annotations([1, 2], [1, 1], ["4", "5"]))
    monkeypatch.setattr(
        "mne.preprocessing.nirs.temporal_derivative_distribution_repair",
        lambda raw: raw,
    )

    Shadaydeh2025FnirsHyperscanning()._standardize_raw(raw)

    assert raw.annotations.description.tolist() == ["Finger", "Foot"]
