"""fNIRS Paradigms."""

import abc
import logging

from moabb.datasets.base import BaseDataset
from moabb.datasets.preprocessing import (
    FixedPipeline,
    ForkPipelines,
    NamedFunctionTransformer,
    RawToEpochs,
    make_fixed_pipeline,
)
from moabb.paradigms.base import BaseParadigm
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

from fnirsatlas.datasets import BENCHMARK_DATASETS
from fnirsatlas.datasets.preprocessing import (
    crop_to_data_step,
    drop_bad_epochs_step,
    interpolate_bads_step,
    interpolate_dist,
    load_data_step,
    preprocess_fnirs,
    reject_by_annotation_step,
)
from fnirsatlas.pipelines.transformations import ChannelSplitter

log = logging.getLogger(__name__)

# Metrics recorded for every evaluation. The first one is also stored as the
# primary ``score`` column, the rest as ``score_<name>``.
DEFAULT_SCORING = {
    "accuracy": "accuracy",
    "kappa": make_scorer(cohen_kappa_score),
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
    "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
}


# Preprocessing defaults. See docs/preprocessing.md.
PREPROCESSING = {
    # Source-detector distance in meters below which a channel is removed.
    "short_channel_threshold": 0.015,
    # Largest relative standard deviation of a flat CW amplitude window.
    "flat_threshold": 1e-6,
    # Required CW amplitude SNR, scalp coupling index and cardiac peak spectral
    # power per window.
    "snr_threshold": 10.0,
    "sci_threshold": 0.5,
    # Lenient, PHOEBE uses 0.1. See docs/preprocessing.md.
    "psp_threshold": 0.02,
    # Window length in seconds of the flat, SNR, SCI and PSP checks.
    "quality_window": 10.0,
    # Temporal derivative distribution repair.
    "tddr": True,
    # Partial pathlength factor of the modified Beer-Lambert law.
    "ppf": 6.0,
    # Epochs with a larger fraction of bad channels and fewer good pairs are
    # dropped. See drop_bad_epochs_step.
    "max_bad_fraction": 0.5,
    "min_good_pairs": 8,
    # Replaces bad channels per epoch. See interpolate_bads_step.
    "interpolate_bads": interpolate_dist,
}


def _split_chromophores(epochs):
    """Return epoch data shaped as `(trials, chromophores, channels, timepoints)`."""
    return ChannelSplitter().fit_transform(epochs)


class BaseFNIRS(BaseParadigm):
    """Base fNIRS paradigm.

    Parameters
    ----------
    filters : list of list of float
        Single band-pass band ``[l_freq, h_freq]`` in Hz. Exactly one band is
        required.
    events : list of str | None
        Event names to keep. ``None`` keeps every event of the dataset.
    tmin : float
        Epoch start in seconds relative to the event onset.
    tmax : float | None
        Epoch end in seconds relative to the event onset. ``None`` uses the
        end of each dataset's interval. Datasets whose interval does not
        contain ``[tmin, tmax]`` are not valid for the paradigm.
    channels : list of str | None
        Channels to keep. ``None`` keeps all of them.
    resample : float
        Target sampling rate in Hz, applied to the raw during preprocessing.
    scorer : str | dict | None
        Scoring passed to the evaluation. ``None`` uses ``DEFAULT_SCORING``.
    overlap : float | None
        Sliding-window overlap in percent. ``None`` epochs on events only.
    reject_by_annotation : bool
        Drop epochs overlapping non-channel ``BAD_`` annotations.
        Channel-specific annotations never drop epochs. Their channels are
        interpolated instead.
    preprocessing : dict | None
        Values replacing those of ``PREPROCESSING``. A dataset's
        ``preprocessing_options`` replace both for that dataset.

    Notes
    -----
    Epochs are baseline-corrected on the pre-stimulus part of each dataset's
    interval, ``[interval[0], 0]``.

    This differs from MOABB paradigms, where ``tmin`` and ``tmax`` are relative
    to the dataset interval start and one ``baseline`` applies to all datasets.
    """

    def __init__(
        self,
        filters=([0.01, 0.1],),
        events=None,
        tmin=0.0,
        tmax=None,
        channels=None,
        resample=5.0,
        scorer=None,
        overlap=None,
        reject_by_annotation=False,
        preprocessing=None,
        **kwargs,
    ):
        self.preprocessing = {**PREPROCESSING, **(preprocessing or {})}
        super().__init__(
            filters=filters,
            events=events,
            channels=channels,
            # The baseline depends on the dataset interval.
            baseline=None,
            tmin=tmin,
            tmax=tmax,
            # Applied to the raw in preprocessing.
            resample=resample,
            overlap=overlap,
            scorer=scorer,
            reject_by_annotation=reject_by_annotation,
            **kwargs,
        )
        assert len(self.filters) == 1, "fNIRS requires exactly one filter"

    def options(self, dataset):
        """Preprocessing options with the dataset's ``preprocessing_options`` applied.

        Raises
        ------
        ValueError
            When an option is not a key of ``PREPROCESSING``.
        """
        options = {**self.preprocessing, **dataset.preprocessing_options}
        unknown = sorted(set(options) - set(PREPROCESSING))
        if unknown:
            raise ValueError(f"{dataset.code}: unknown preprocessing options {unknown}")
        return options

    def prepare_process(self, dataset):
        """Keep the dataset's options for the raw pipeline, which gets no dataset."""
        self._options = self.options(dataset)

    def _get_raw_pipelines(self):
        options = self._options
        fmin, fmax = self.filters[0]
        return [
            preprocess_fnirs(
                short_channel_threshold=options["short_channel_threshold"],
                flat_threshold=options["flat_threshold"],
                snr_threshold=options["snr_threshold"],
                sci_threshold=options["sci_threshold"],
                psp_threshold=options["psp_threshold"],
                quality_window=options["quality_window"],
                tddr=options["tddr"],
                ppf=options["ppf"],
                l_freq=fmin,
                h_freq=fmax,
                resample=self.resample,
            )
        ]

    def _get_epochs_pipeline(self, return_epochs, return_raws, dataset):
        """Epoch around the event onset, drop mostly bad epochs, then repair bad channels.

        Interpolating here rather than on the raw recording allows us to
        interpolate bad segments instead of entire channels.
        """
        if return_raws:
            return None
        options = self.options(dataset)
        start, end = dataset.interval
        tmax = end if self.tmax is None else self.tmax
        steps = [
            # Epochs include the baseline window, which is cropped afterwards.
            ("epoching", self._epoching_step(dataset, start, tmax, (start, 0.0))),
        ]
        if self.reject_by_annotation:
            steps.append(("reject", reject_by_annotation_step()))
        steps.append(
            (
                "drop_bad",
                drop_bad_epochs_step(
                    options["max_bad_fraction"], options["min_good_pairs"]
                ),
            )
        )
        interpolate = interpolate_bads_step(options["interpolate_bads"])
        steps.append(("interpolate", interpolate))
        if start < self.tmin:
            steps.append(("crop", crop_to_data_step(self.tmin, tmax)))
        if return_epochs:
            steps.append(("load_data", load_data_step()))
        return FixedPipeline(steps)

    def _epoching_step(
        self,
        dataset: BaseDataset,
        tmin: float,
        tmax: float,
        baseline: tuple[float, float],
    ) -> FixedPipeline:
        """Raw to baseline-corrected epochs, in seconds relative to the onset.

        Replaces the epoching of ``BaseParadigm._get_epochs_pipeline``, which
        shifts the window and baseline by the dataset interval start and so
        cannot express an onset-relative window or a per-dataset baseline.
        """
        events = ForkPipelines(
            [
                ("raw", make_fixed_pipeline(None)),
                ("events", self._get_events_pipeline(dataset)),
            ]
        )
        epochs = RawToEpochs(
            event_id=self.used_events(dataset),
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            channels=self.channels,  # ty: ignore[invalid-argument-type]
            interpolate_missing_channels=self.interpolate_missing_channels,
            return_all_modalities=dataset.return_all_modalities,
            # MNE would also reject on channel-specific annotations.
            reject_by_annotation=False,
        )
        return FixedPipeline([("extract events", events), ("create epochs", epochs)])

    def _get_array_pipeline(
        self, return_epochs, return_raws, dataset, processing_pipeline
    ):
        """Standard array pipeline + split chromophores > (n, ctypes, cgroups, t)"""
        pipeline = super()._get_array_pipeline(
            return_epochs, return_raws, dataset, processing_pipeline
        )
        if pipeline is None:
            return pipeline
        return pipeline.set_params(
            get_data=NamedFunctionTransformer(
                _split_chromophores, display_name="Epochs to Chromophore Array"
            )
        )

    def is_valid(self, dataset):
        return (
            dataset.interval is not None
            and dataset.paradigm.startswith("fnirs")
            and (not self.events or set(self.events) <= dataset.event_id.keys())
            and self._window_fits(dataset)
        )

    def _window_fits(self, dataset: BaseDataset) -> bool:
        """Whether the dataset interval contains the epoch window."""
        start, end = dataset.interval
        fits = start <= self.tmin and (self.tmax is None or self.tmax <= end)
        if not fits:
            log.warning(
                "%s interval %s does not contain the epoch window [%s, %s]",
                dataset.code,
                dataset.interval,
                self.tmin,
                self.tmax,
            )
        return fits

    @abc.abstractmethod
    def used_events(self, dataset):
        pass

    @property
    def datasets(self):
        datasets = (cls() for cls in BENCHMARK_DATASETS)
        return [dataset for dataset in datasets if self.is_valid(dataset)]

    @property
    def scoring(self):
        if self.scorer is not None:
            return self.scorer
        return dict(DEFAULT_SCORING)


class FNIRSClassification(BaseFNIRS):
    """fNIRS classification paradigm. Uses all events from dataset."""

    dataset_paradigm = None

    def used_events(self, dataset):
        out = {}
        if self.events is None:
            for k, v in dataset.event_id.items():
                out[k] = v
        else:
            for event in self.events:
                if event in dataset.event_id:
                    out[event] = dataset.event_id[event]
        return out

    def is_valid(self, dataset):
        return (
            self.dataset_paradigm is None or dataset.paradigm == self.dataset_paradigm
        ) and super().is_valid(dataset)


class Emotion(FNIRSClassification):
    """Emotion classification paradigm."""

    dataset_paradigm = "fnirsEmotion"


class MotorExecution(FNIRSClassification):
    """Motor-execution classification paradigm."""

    dataset_paradigm = "fnirsMotorExecution"


class MotorImagery(FNIRSClassification):
    """Motor-imagery classification paradigm."""

    dataset_paradigm = "fnirsMotorImagery"


class NBack(FNIRSClassification):
    """N-back workload classification paradigm."""

    dataset_paradigm = "fnirsNBack"


class FingerTapping(MotorExecution):
    """Finger-tapping classification paradigm."""


class MentalArithmetic(FNIRSClassification):
    """Mental-arithmetic classification paradigm."""

    dataset_paradigm = "fnirsMentalArithmetic"
