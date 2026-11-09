"""Print a pipeline comparison table from benchmark result files."""

import argparse
from pathlib import Path

import h5py
import pandas as pd
from mne import get_config
from moabb.analysis.results import Results as MoabbResults

from fnirsatlas.results import subject_mean

INDEX_COLUMNS = ["dataset", "paradigm", "evaluation"]
REQUIRED_COLUMNS = {*INDEX_COLUMNS, "pipeline", "subject", "score"}


def make_table(results, evaluation=None):
    """Return mean pipeline scores as percentages, every subject weighted equally."""
    missing = REQUIRED_COLUMNS - set(results.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if evaluation:
        results = results[results["evaluation"] == evaluation]
    if results.empty:
        raise ValueError("No matching results")
    return (
        subject_mean(results, [*INDEX_COLUMNS, "pipeline"])
        .unstack()
        .mul(100)
        .reset_index()
        .rename_axis(columns=None)
    )


def load_results(paths):
    """Load analysis CSVs or MOABB HDF5 caches."""
    frames = []
    for path in paths:
        if path.suffix == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix == ".hdf5":
            with h5py.File(path) as cache:
                frame = MoabbResults._to_dataframe_from_file(cache)
            frame["paradigm"] = path.parent.parent.name
            frame["evaluation"] = path.parent.name.removesuffix("Evaluation")
        else:
            raise ValueError(f"Unsupported result file: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def default_paths():
    """Find analysis CSVs, falling back to unsuffixed MOABB caches."""
    paths = sorted(Path("benchmark").glob("**/analysis/data.csv"))
    if paths:
        return paths
    cache_root = get_config("MOABB_RESULTS")
    if not isinstance(cache_root, str):
        return []
    return sorted(Path(cache_root).glob("results/**/results_.hdf5"))


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/results_table.py "
            "--evaluation CrossSubject /path/to/results_.hdf5"
        ),
    )
    parser.add_argument("paths", nargs="*", type=Path, help="CSV or HDF5 result files")
    parser.add_argument("--evaluation", help="Only include this evaluation")
    return parser.parse_args(argv)


def main(argv=None):
    """Load benchmark results and print their comparison table."""
    args = parse_args(argv)
    paths = args.paths or default_paths()
    if not paths:
        raise FileNotFoundError("No benchmark result files found")
    results = load_results(paths)
    table = make_table(results, args.evaluation)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.1f}"))


if __name__ == "__main__":
    main()
