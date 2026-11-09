from typing import ClassVar

import mne
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import check_scoring

from fnirsatlas import (
    Bak2019FnirsTapping,
    Bnci2014FnirsMA,
    Emotion,
    FingerTapping,
    FNIRSClassification,
    Huang2021FnirsWorkload,
    Luke2021FnirsTapping,
    MentalArithmetic,
    MotorExecution,
    MotorImagery,
    NBack,
    Nemo2024FnirsImagery,
    Shin2016FnirsMI,
    Yi2025FnirsMI,
)
from fnirsatlas.datasets import BENCHMARK_DATASETS
from fnirsatlas.datasets.base import BaseFNIRSDataset
from fnirsatlas.datasets.preprocessing import interpolate_dist, interpolate_nearest
from fnirsatlas.paradigms import epochs_to_array
from fnirsatlas.paradigms.fnirs import DEFAULT_SCORING


def test_epochs_to_array_returns_labels_metadata_and_info():
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], 5.0, ["hbo", "hbr"])
    metadata = pd.DataFrame({"subject": [1, 1], "session": ["0", "0"]})
    epochs = mne.EpochsArray(
        np.ones((2, 2, 4)),
        info,
        events=np.array([[0, 0, 1], [4, 0, 2]]),
        event_id={"left": 1, "right": 2},
        metadata=metadata,
        tmin=-0.2,
        verbose=False,
    )

    X, y, returned_metadata, epoch_info = epochs_to_array(epochs, unit_factor=2.0)

    np.testing.assert_array_equal(X, 2.0)
    np.testing.assert_array_equal(y, ["left", "right"])
    assert returned_metadata.equals(metadata)
    assert epoch_info["sfreq"] == 5.0
    assert epoch_info["ch_names"] == ["S1_D1 hbo", "S1_D1 hbr"]
    assert epoch_info["cond2label"] == {"left": 1, "right": 2}
    assert epoch_info["tmin"] == epochs.tmin
    assert epoch_info["tmax"] == epochs.tmax
    assert epoch_info["chs"] is not epochs.info["chs"]


def test_fnirs_paradigm_rejects_multiple_filters():
    with pytest.raises(AssertionError, match="exactly one filter"):
        FNIRSClassification(filters=([0.01, 0.1], [0.1, 0.2]))


def test_fnirs_classification_paradigm_rejects_recording_only_dataset():
    class Dataset:
        paradigm = "fnirsRestingState"
        interval = None
        event_id: ClassVar = {"event": 1}

    assert not FNIRSClassification().is_valid(Dataset())


def test_dataset_paradigms_are_bids_safe():
    for dataset in FNIRSClassification().datasets:
        assert dataset.paradigm.startswith("fnirs")
        assert dataset.paradigm.isalnum()


@pytest.mark.parametrize(
    "options",
    [{}, {"short_channel_threshold": 0.02}, {"snr_threshold": 20, "ppf": 6.0}],
)
def test_raw_pipeline_combines_dataset_and_paradigm_options(monkeypatch, options):
    class Dataset:
        preprocessing_options: ClassVar = options
        interval: ClassVar = [-2, 15]

    received = {}
    pipeline = object()

    def construct(**kwargs):
        received.update(kwargs)
        return pipeline

    monkeypatch.setattr("fnirsatlas.paradigms.fnirs.preprocess_fnirs", construct)
    paradigm = FNIRSClassification(
        preprocessing={
            "snr_threshold": 10,
            "sci_threshold": 0.2,
            "tddr": False,
            "ppf": 5.0,
            "interpolate_bads": None,
        }
    )
    paradigm.prepare_process(Dataset())

    result = paradigm._get_raw_pipelines()

    assert result == [pipeline]
    # Dataset options replace the paradigm arguments of the same name.
    assert received == {
        "short_channel_threshold": 0.015,
        "flat_threshold": 1e-6,
        "snr_threshold": 10,
        "sci_threshold": 0.2,
        "psp_threshold": 0.02,
        "quality_window": 10.0,
        "tddr": False,
        "ppf": 5.0,
        "l_freq": 0.01,
        "h_freq": 0.1,
        "resample": 5.0,
        **options,
    }


def test_unknown_dataset_option_raises():
    class Dataset:
        code = "Dataset"
        preprocessing_options: ClassVar = {"snr_treshold": 20}

    with pytest.raises(ValueError, match="snr_treshold"):
        FNIRSClassification().prepare_process(Dataset())


class _SyntheticFnirs(BaseFNIRSDataset):
    """Two subjects of the same recording, acquired at different sampling rates.

    S1_D1 and S2_D2 are 3 cm long. S1_D1 hbo has a bad segment overlapping
    only the first epoch, and a non-channel bad segment overlaps only the
    second epoch. S3_D3 is a 1 cm short channel. With the default interval,
    the first epoch starts 0.07 s after the recording start.
    """

    SFREQS: ClassVar = {1: 25.0, 2: 12.5}
    ONSETS: ClassVar = [2.05, 60.0, 100.0]
    PAIRS: ClassVar = {"S1_D1": 0.03, "S2_D2": 0.03, "S3_D3": 0.01}

    def __init__(self, interval=(-1.98, 15)):
        # The default start is off the sample grid at every rate used here.
        super().__init__(
            subjects=[1, 2],
            sessions_per_subject=1,
            events={"left": 1, "right": 2},
            code="SyntheticFnirs",
            interval=list(interval),
            paradigm="fnirsMotorExecution",
            return_all_modalities=True,
        )

    def _get_single_subject_data(self, subject):
        sfreq = self.SFREQS[subject]
        times = np.arange(int(150 * sfreq)) / sfreq
        wave = np.sin(0.05 * 2 * np.pi * times) * 1e-6
        ch_names = [f"{pair} {kind}" for pair in self.PAIRS for kind in ("hbo", "hbr")]
        raw = self._make_raw(
            data=np.vstack([wave * (scale + 1) for scale in range(len(ch_names))]),
            ch_names=ch_names,
            ch_types=[name.split()[1] for name in ch_names],
            sfreq=sfreq,
        )
        for channel in raw.info["chs"]:
            # midpoint, then the source and detector positions fNIRS needs
            pair = channel["ch_name"].split()[0]
            offset = 0.03 * list(self.PAIRS).index(pair) + 0.03
            half_distance = self.PAIRS[pair] / 2
            channel["loc"][:3] = (offset, 0.0, 0.0)
            channel["loc"][3:6] = (offset - half_distance, 0.0, 0.0)
            channel["loc"][6:9] = (offset + half_distance, 0.0, 0.0)
        raw.set_annotations(
            mne.Annotations(
                onset=self.ONSETS,
                duration=[6.0, 6.0, 6.0],
                description=["left", "right", "left"],
            )
        )
        raw.annotations.append(raw.first_time, 10.0, "BAD_test", [["S1_D1 hbo"]])
        raw.annotations.append(raw.first_time + 70.0, 1.0, "BAD_block")
        return self._single_run_session(self._standardize_raw(raw))

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return []


def test_epoch_length_does_not_depend_on_acquisition_rate():
    dataset = _SyntheticFnirs()
    paradigm = FNIRSClassification(preprocessing={"tddr": False})

    lengths = {
        subject: paradigm.get_data(dataset=dataset, subjects=[subject])[0].shape[2]
        for subject in dataset.subject_list
    }

    assert len(set(lengths.values())) == 1, lengths


@pytest.mark.parametrize("subject", [1, 2])
def test_epochs_keep_every_event_onset(subject):
    dataset = _SyntheticFnirs()
    paradigm = FNIRSClassification(preprocessing={"tddr": False})

    epochs = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)[
        0
    ]

    sfreq = epochs.info["sfreq"]
    onsets = epochs.events[:, 0] / sfreq
    np.testing.assert_allclose(onsets, dataset.ONSETS, atol=0.5 / sfreq)


def test_short_channels_are_removed():
    epochs = FNIRSClassification(preprocessing={"tddr": False}).get_data(
        dataset=_SyntheticFnirs(), subjects=[1], return_epochs=True
    )[0]

    assert not any(name.startswith("S3_D3") for name in epochs.ch_names)


@pytest.mark.parametrize("method", [interpolate_nearest, interpolate_dist])
def test_bad_segments_are_interpolated_only_in_overlapping_epochs(method):
    dataset = _SyntheticFnirs()
    paradigm = FNIRSClassification(
        resample=None, preprocessing={"tddr": False, "interpolate_bads": method}
    )

    epochs = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)[0]

    assert epochs.info["bads"] == []
    for kind in ("hbo", "hbr"):
        interpolated = epochs.get_data(picks=f"S1_D1 {kind}", verbose=False)
        source = epochs.get_data(picks=f"S2_D2 {kind}", verbose=False)
        assert np.array_equal(interpolated[0], source[0])
        assert not np.allclose(interpolated[1:], source[1:])


class _SyntheticFnirsAllBad(_SyntheticFnirs):
    """``_SyntheticFnirs`` with every long channel bad for 1 s of the third epoch."""

    def _get_single_subject_data(self, subject):
        data = super()._get_single_subject_data(subject)
        raw = data["0fnirs"]["0"]
        raw.annotations.append(
            raw.first_time + 100.0, 1.0, "BAD_test", [["S1_D1 hbo", "S2_D2 hbo"]]
        )
        return data


@pytest.mark.parametrize(
    ("dataset", "reject", "max_bad_fraction", "events"),
    [
        (_SyntheticFnirs, False, 0.5, [1, 2, 1]),
        # BAD_block overlaps the second epoch.
        (_SyntheticFnirs, True, 0.5, [1, 1]),
        # BAD_test marks S1_D1, half of the long channels, in the first epoch.
        (_SyntheticFnirs, False, 0.4, [2, 1]),
        # Nothing is left to interpolate from in the third epoch.
        (_SyntheticFnirsAllBad, False, 0.5, [1, 2]),
    ],
)
def test_bad_annotations_drop_epochs(dataset, reject, max_bad_fraction, events):
    dataset = dataset()
    paradigm = FNIRSClassification(
        resample=None,
        reject_by_annotation=reject,
        preprocessing={"tddr": False, "max_bad_fraction": max_bad_fraction},
    )

    epochs = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)[0]

    assert epochs.events[:, 2].tolist() == events


class _SyntheticFnirsBadSubject(_SyntheticFnirs):
    """``_SyntheticFnirs`` with S2_D2, half of the long channels, bad in subject 1."""

    def _get_single_subject_data(self, subject):
        data = super()._get_single_subject_data(subject)
        if subject == 1:
            data["0fnirs"]["0"].info["bads"] = ["S2_D2 hbo", "S2_D2 hbr"]
        return data


@pytest.mark.parametrize("return_epochs", [False, True])
def test_run_without_epochs_is_skipped(return_epochs):
    paradigm = FNIRSClassification(
        resample=None, preprocessing={"tddr": False, "max_bad_fraction": 0.4}
    )

    *_, metadata = paradigm.get_data(
        dataset=_SyntheticFnirsBadSubject(),
        subjects=[1, 2],
        return_epochs=return_epochs,
    )

    # Subject 2 keeps the epochs without the S1_D1 bad segment.
    assert metadata["subject"].tolist() == [2, 2]


def test_epochs_end_on_last_sample_when_interval_is_off_the_sample_grid():
    # 12.9 s is not a multiple of the 0.2 s sample spacing at 5 Hz.
    dataset = _SyntheticFnirs(interval=(-2, 12.9))
    paradigm = FNIRSClassification(resample=5.0, preprocessing={"tddr": False})
    epochs = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)[0]
    assert np.isclose(epochs.tmax, 12.8)


@pytest.mark.parametrize("interval", [(-1.98, 15), (0, 15)])
def test_epoch_window_is_event_time_with_prestimulus_baseline(interval):
    paradigm = FNIRSClassification(
        resample=5.0, tmin=interval[0], tmax=10, preprocessing={"tddr": False}
    )
    epochs = paradigm.get_data(
        dataset=_SyntheticFnirs(interval=interval), subjects=[1], return_epochs=True
    )[0]
    assert np.isclose(epochs.tmin, interval[0], atol=0.1)
    assert np.isclose(epochs.tmax, 10)
    in_baseline = (epochs.times >= interval[0]) & (epochs.times <= 0)
    baseline = epochs.get_data(copy=False)[..., in_baseline]
    np.testing.assert_allclose(baseline.mean(axis=-1), 0, atol=1e-12)


@pytest.mark.parametrize("window", [{"tmin": -3}, {"tmax": 20}])
def test_window_outside_dataset_interval_is_invalid(window, caplog):
    dataset = _SyntheticFnirs(interval=(-1.98, 15))
    paradigm = FNIRSClassification(**window)
    assert not paradigm.is_valid(dataset)
    assert "interval" in caplog.text
    with pytest.raises(AssertionError, match="not valid"):
        paradigm.get_data(dataset=dataset, subjects=[1])


@pytest.mark.data
def test_default_epochs_cover_poststimulus_interval():
    dataset = Bnci2014FnirsMA(subjects=[1])
    paradigm = FNIRSClassification(resample=5.0)

    X, _, _ = paradigm.get_data(
        dataset=dataset,
        subjects=[1],
    )

    assert X.shape[1:3] == (2, 52)
    assert X.shape[3] == 76


def test_motor_execution_uses_bak_tapping_events():
    dataset = Bak2019FnirsTapping(subjects=[1])
    paradigm = MotorExecution()

    assert isinstance(paradigm, FNIRSClassification)
    assert paradigm.used_events(dataset) == {
        "Tapping/Right": 1,
        "Tapping/Left": 2,
        "Tapping/Foot": 3,
    }
    assert list(paradigm.scoring) == list(DEFAULT_SCORING)


@pytest.mark.parametrize(
    ("paradigm", "dataset"),
    [
        (MentalArithmetic(), Bnci2014FnirsMA()),
        (Emotion(), Nemo2024FnirsImagery()),
        (FingerTapping(), Luke2021FnirsTapping()),
        (MotorExecution(), Bak2019FnirsTapping()),
        (MotorImagery(), Shin2016FnirsMI()),
        (NBack(), Huang2021FnirsWorkload()),
    ],
)
def test_paradigm_discovers_compatible_dataset(paradigm, dataset):
    assert dataset.code in {d.code for d in paradigm.datasets}


def test_paradigm_uses_only_default_datasets():
    assert {type(d) for d in MotorImagery().datasets} <= set(BENCHMARK_DATASETS)


@pytest.mark.parametrize(
    ("paradigm", "dataset"),
    [
        (Emotion(), Nemo2024FnirsImagery()),
        (FingerTapping(), Luke2021FnirsTapping()),
        (MentalArithmetic(), Bnci2014FnirsMA()),
        (MotorImagery(), Shin2016FnirsMI()),
        (MotorImagery(), Yi2025FnirsMI()),
        (NBack(), Huang2021FnirsWorkload()),
    ],
)
def test_task_paradigms_select_dataset_events(paradigm, dataset):
    assert paradigm.used_events(dataset) == dataset.event_id
    assert isinstance(paradigm, FNIRSClassification)


def test_default_scoring_reports_all_metrics():
    X = np.zeros((6, 2))
    y = np.array([0, 0, 1, 1, 2, 2])
    estimator = DummyClassifier(strategy="most_frequent").fit(X, y)

    scores = check_scoring(estimator, FNIRSClassification().scoring)(estimator, X, y)

    assert list(scores) == list(DEFAULT_SCORING)
    assert scores["accuracy"] == pytest.approx(1 / 3)
    assert scores["kappa"] == pytest.approx(0.0)


def test_scorer_overrides_default_scoring():
    assert FNIRSClassification(scorer="roc_auc").scoring == "roc_auc"


def test_paradigm_returns_a_chromophore_axis():
    """Arrays carry hbo and hbr in their own dimension, ordered by type."""
    dataset = _SyntheticFnirs()

    X = FNIRSClassification(preprocessing={"tddr": False}).get_data(
        dataset=dataset, subjects=[1]
    )[0]
    epochs = FNIRSClassification(preprocessing={"tddr": False}).get_data(
        dataset=dataset, subjects=[1], return_epochs=True
    )[0]

    assert epochs.ch_names == ["S1_D1 hbo", "S1_D1 hbr", "S2_D2 hbo", "S2_D2 hbr"]
    assert X.shape == (len(epochs), 2, 2, X.shape[-1])
    data = epochs.get_data(copy=True) * dataset.unit_factor
    np.testing.assert_allclose(X[:, 0], data[:, 0::2])
    np.testing.assert_allclose(X[:, 1], data[:, 1::2])
