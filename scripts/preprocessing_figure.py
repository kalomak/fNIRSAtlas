"""Plot channel pairs after each step of the paradigm's preprocessing.

The panels follow the raw pipeline of ``FNIRSClassification``, so added or
reordered steps appear without changes here. Steps that leave the recording
unchanged are skipped. Segments marked bad are shaded and labelled with the
check that marked them. The last row shows the epochs of the same run at their
onsets. The pairs are stacked under each other for every step.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.ticker import MaxNLocator

from fnirsatlas import datasets
from fnirsatlas.benchmark import results_dir
from fnirsatlas.paradigms import FNIRSClassification

# Channel type -> (y-axis label, scale applied to the data).
UNITS = {
    "fnirs_cw_amplitude": ("Intensity (V)", 1.0),
    "fnirs_od": ("OD", 1.0),
    "hbo": ("Hb (µM)", 1e6),
    "hbr": ("Hb (µM)", 1e6),
}


def trace(raw, pipeline):
    """Return ``(title, raw)`` for the input and after each step that changes it."""
    snapshots = [("Raw", raw)]
    for step in pipeline.named_steps.values():
        previous = snapshots[-1][1]
        current = step.fit_transform(previous.copy())
        if not same_recording(previous, current):
            snapshots.append((step.display_name, current))
    return snapshots


def same_recording(first, second):
    """Whether two recordings have equal channels, data and annotations."""
    return (
        first.ch_names == second.ch_names
        and first.info["sfreq"] == second.info["sfreq"]
        and np.array_equal(first.get_data(), second.get_data(), equal_nan=True)
        and first.annotations == second.annotations
    )


def pair_picks(inst, pair):
    """Return the names of the channels of a source-detector pair."""
    picks = [name for name in inst.ch_names if name.split(" ")[0] == pair]
    if not picks:
        raise ValueError(f"No channels of {pair} in {inst.ch_names[:6]}...")
    return picks


def bad_spans(raw, pair, start, stop):
    """Return ``{check: [(left, right), ...]}`` of merged bad marks on the pair."""
    annotations = raw.annotations
    spans = {}
    for onset, duration, description, channels in zip(
        annotations.onset - raw.first_time,
        annotations.duration,
        annotations.description,
        annotations.ch_names,
        strict=True,
    ):
        covers = not channels or any(name.split(" ")[0] == pair for name in channels)
        if description.startswith("BAD_") and covers:
            check = description.removeprefix("BAD_").upper()
            spans.setdefault(check, []).append((onset, onset + duration))
    return {
        check: [
            (max(left, start), min(right, stop))
            for left, right in merge(intervals)
            if left < stop and right > start
        ]
        for check, intervals in spans.items()
    }


def merge(intervals):
    """Merge overlapping or touching intervals."""
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def check_order(snapshots):
    """Return the bad-mark checks in the order the pipeline first adds them."""
    order = []
    for _, raw in snapshots:
        for description in raw.annotations.description:
            check = description.removeprefix("BAD_").upper()
            if description.startswith("BAD_") and check not in order:
                order.append(check)
    return order


def shade_bad(ax, raw, pair, start, stop, order):
    """Shade bad marks on the pair and label each span with its check.

    Each check's label has a fixed height given by its position in ``order``.
    """
    for check, spans in bad_spans(raw, pair, start, stop).items():
        for left, right in spans:
            ax.axvspan(left, right, color="tab:red", alpha=0.15, linewidth=0)
            ax.text(
                (left + right) / 2,
                0.95 - 0.16 * order.index(check),
                check,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6,
                color="darkred",
            )


def plot_channels(ax, inst, pair, data, times):
    """Plot the pair's channels in their units and set the y-axis label."""
    names = pair_picks(inst, pair)
    label, scale = UNITS[inst.get_channel_types(picks=names)[0]]
    for index, name in enumerate(names):
        values = data(name) * scale
        for segment, (time, value) in enumerate(zip(times, values, strict=True)):
            line_label = name.split(" ")[1] if segment == 0 else None
            ax.plot(time, value, color=f"C{index}", label=line_label, linewidth=0.8)
    ax.set_ylabel(label, fontsize=8)


def plot_raw(ax, raw, pair, start, stop, order):
    """Plot the pair's channels of one snapshot between start and stop seconds."""
    window = raw.copy().crop(start, stop)
    plot_channels(
        ax, raw, pair, lambda name: window.get_data(picks=name), [window.times + start]
    )
    shade_bad(ax, raw, pair, start, stop, order)


def plot_epochs(ax, epochs, raw, pair, start, stop):
    """Plot each epoch of the pair that lies within the window at its onset.

    Each epoch is labelled with its event name.
    """
    onsets = epochs.events[:, 0] / epochs.info["sfreq"] - raw.first_time
    inside = [
        index
        for index, onset in enumerate(onsets)
        if onset + epochs.times[0] >= start and onset + epochs.times[-1] <= stop
    ]
    times = [onsets[index] + epochs.times for index in inside]
    plot_channels(
        ax, epochs, pair, lambda name: epochs.get_data(picks=name)[inside, 0], times
    )
    for index, time in zip(inside, times, strict=True):
        ax.text(
            time[0],
            0.95,
            epochs.metadata["label"].iloc[index],
            transform=ax.get_xaxis_transform(),
            va="top",
            fontsize=6,
        )


def run_epochs(paradigm, dataset, subject, session, run):
    """Return the paradigm's epochs of one run, with a ``label`` metadata column."""
    epochs, labels, metadata = paradigm.get_data(
        dataset, subjects=[subject], return_epochs=True
    )
    epochs.metadata = metadata.assign(label=labels).reset_index(drop=True)
    keep = (metadata["session"] == session) & (metadata["run"] == run)
    return epochs[np.flatnonzero(keep.to_numpy())]


def draw(snapshots, epochs, pairs, start, stop):
    """Return the figure with one group of stacked pairs per stage."""
    stages = [*snapshots, ("Epochs", None)]
    # Each stage is its pair rows plus an empty spacer row that holds its title.
    ratios = [0.6, *[1] * len(pairs)] * len(stages)
    fig, axes = plt.subplots(
        len(ratios),
        1,
        sharex=True,
        figsize=(7, 0.95 * sum(ratios)),
        gridspec_kw={"height_ratios": ratios, "hspace": 0},
    )
    rows = iter(axes)
    order = check_order(snapshots)
    for title, snapshot in stages:
        spacer = next(rows)
        spacer.axis("off")
        spacer.text(
            0, 0.3, title, transform=spacer.transAxes, fontsize=8, style="italic"
        )
        for pair in pairs:
            ax = next(rows)
            if snapshot is None:
                plot_epochs(ax, epochs, snapshots[-1][1], pair, start, stop)
            else:
                plot_raw(ax, snapshot, pair, start, stop, order)
            ax.set_ylabel(f"{pair}\n{ax.get_ylabel()}", fontsize=8)
            # Drop edge ticks so labels of touching panels do not collide.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
            if pair != pairs[-1]:
                ax.tick_params(axis="x", bottom=False)
    axes[-1].set_xlim(start, stop)
    axes[-1].set_xlabel("Time (s)")
    fig.align_ylabels(axes)
    return fig


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Luke2021FnirsSpeech")
    parser.add_argument("--subject", type=int, default=6)
    parser.add_argument(
        "--pairs", nargs="+", default=["S7_D7", "S11_D10"], help="Source-detector pairs"
    )
    parser.add_argument("--start", type=float, default=677.0, help="Seconds")
    parser.add_argument("--duration", type=float, default=100.0, help="Seconds")
    parser.add_argument("--output", type=Path, help="Output image path")
    return parser.parse_args(argv)


def main(argv=None):
    """Trace the first run of a subject and save the figure."""
    args = parse_args(argv)
    mne.set_log_level("ERROR")
    dataset = datasets.get_dataset(args.dataset)
    paradigm = FNIRSClassification()
    paradigm.prepare_process(dataset)
    sessions = dataset.get_data(subjects=[args.subject])[args.subject]
    session, runs = next(iter(sessions.items()))
    run, raw = next(iter(runs.items()))
    snapshots = trace(raw, paradigm._get_raw_pipelines()[0])
    epochs = run_epochs(paradigm, dataset, args.subject, session, run)
    stop = args.start + args.duration
    fig = draw(snapshots, epochs, args.pairs, args.start, stop)
    output = args.output or results_dir() / "benchmark" / "preprocessing.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
