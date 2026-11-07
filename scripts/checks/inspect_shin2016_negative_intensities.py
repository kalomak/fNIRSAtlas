"""Inspect Shin2016 negative-intensity warnings and plot a comparison segment.

The signal drops below zero very briefly and not by much so probably fine.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pandas as pd

from fnirsatlas.config import get_results_dir
from fnirsatlas.load_parser import load_parser
from fnirsatlas.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def _load_raw(parser, subject_id: str, session) -> mne.io.BaseRaw:
    """Load one subject/session as raw amplitude data."""
    data = parser.load_data(subject_id, session=session)
    return parser._raw_from_session(data)


def _detect_negative_warning(raw: mne.io.BaseRaw) -> bool:
    """Return True when MNE emits the negative-intensity warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mne.preprocessing.nirs.optical_density(raw.copy())
    return any("Negative intensities encountered" in str(w.message) for w in caught)


def _first_negative_hit(raw: mne.io.BaseRaw) -> dict:
    """Return metadata for the first negative sample in fNIRS channels."""
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    data = raw.get_data(picks=picks)
    neg = data < 0
    first_ch, first_t = neg.argmax() // neg.shape[1], neg.argmax() % neg.shape[1]
    ch_ix = picks[first_ch]
    return {
        "channel": raw.ch_names[ch_ix],
        "sample": first_t,
        "time_s": first_t / raw.info["sfreq"],
        "value": float(data[first_ch, first_t]),
        "n_negative": int(neg.sum()),
        "min_value": float(data.min()),
    }


def _good_segment(raw: mne.io.BaseRaw) -> dict:
    """Return a stable, positive segment from the first fNIRS channel."""
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    ch_ix = picks[0]
    data = raw.get_data(picks=[ch_ix])[0]
    sample = min(len(data) // 2, max(0, len(data) - 1))
    return {
        "channel": raw.ch_names[ch_ix],
        "sample": sample,
        "time_s": sample / raw.info["sfreq"],
        "value": float(data[sample]),
        "min_value": float(data.min()),
    }


def _segment(raw: mne.io.BaseRaw, channel: str, center_sample: int, width: int) -> tuple:
    """Return times and values for a centered channel segment."""
    ch_ix = raw.ch_names.index(channel)
    data = raw.get_data(picks=[ch_ix])[0]
    start = max(0, center_sample - width)
    stop = min(len(data), center_sample + width)
    times = raw.times[start:stop]
    return times, data[start:stop]


def inspect_dataset(dataset_name: str) -> tuple[pd.DataFrame, dict | None]:
    """Scan a Shin2016 dataset and return warning rows plus one example hit."""
    parser = load_parser(dataset_name, download="no")
    rows = []
    example = None
    for subject_id in parser.get_subject_ids():
        for session in parser.get_sessions(subject_id):
            raw = _load_raw(parser, subject_id, session)
            warned = _detect_negative_warning(raw)
            if not warned:
                continue
            hit = _first_negative_hit(raw)
            row = {"dataset": dataset_name, "subject": subject_id, "session": session, **hit}
            rows.append(row)
            if example is None:
                example = {"subject": subject_id, "session": session, "raw": raw, **hit}
    return pd.DataFrame(rows), example


def load_luke2021_example() -> dict:
    """Load one clean Luke2021 raw segment for comparison."""
    parser = load_parser("luke2021", download="no")
    subject_id = parser.get_subject_ids()[0]
    session = parser.get_sessions(subject_id)[0]
    raw = _load_raw(parser, subject_id, session)
    segment = _good_segment(raw)
    return {"subject": subject_id, "session": session, "raw": raw, **segment}


def save_plot(shin_example: dict, luke_example: dict, output_path: Path, width: int) -> None:
    """Save side-by-side raw amplitude segments for bad and good examples."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
    specs = [
        (axes[0], shin_example, "Shin2016 negative-intensity segment", True),
        (axes[1], luke_example, "Luke2021 comparison segment", False),
    ]
    for ax, example, title, mark_center in specs:
        times, values = _segment(example["raw"], example["channel"], example["sample"], width)
        ax.plot(times, values, linewidth=1.5)
        if mark_center:
            ax.axvline(example["time_s"], color="crimson", linestyle="--", linewidth=1)
            ax.scatter([example["time_s"]], [example["value"]], color="crimson", zorder=3)
        ax.set_title(
            f"{title}: subject={example['subject']} session={example['session']} channel={example['channel']}"
        )
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    """Run the inspection and save a summary plus comparison plot."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=get_results_dir() / "checks" / "negative_intensity_inspection",
    )
    parser.add_argument("--width", type=int, default=200)
    args = parser.parse_args()

    frames = []
    examples = {}
    for dataset_name in ("shin2016__ma", "shin2016__mi"):
        df, example = inspect_dataset(dataset_name)
        frames.append(df)
        examples[dataset_name] = example
    result = pd.concat(frames, ignore_index=True)
    csv_path = args.save_dir / "shin2016_negative_intensity_hits.csv"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)
    logger.info("Saved hit summary to %s", csv_path)
    luke_example = load_luke2021_example()
    for dataset_name, example in examples.items():
        if example is None:
            logger.warning("No negative-intensity warnings found for %s", dataset_name)
            continue
        plot_path = args.save_dir / f"{dataset_name}_vs_luke2021.png"
        save_plot(example, luke_example, plot_path, args.width)
        logger.info("Saved comparison plot to %s", plot_path)
    if result.empty:
        logger.info("No negative-intensity warnings found in Shin2016")
        return 0
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
