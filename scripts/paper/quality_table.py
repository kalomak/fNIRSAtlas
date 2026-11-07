"""
Create a summary table of channel quality (SCI) across datasets.

For each dataset, loads or computes SCI values and reports:
- number of unique channels
- mean/median SCI
- elimination percentages at thresholds 0.2, 0.5, and 0.7

Example usage:
    python scripts/paper/quality_table.py
    python scripts/paper/quality_table.py --datasets luke2021 luke2024b
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from fnirsatlas.format import format_dataset_name
from fnirsatlas.logging import setup_logging
from fnirsatlas.parsers import get_available_datasets

sys.path.insert(0, str(Path(__file__).parent.parent / "checks"))
from check_sci_quality import analyze_dataset_sci  # noqa: E402

setup_logging(level="WARNING")
logger = logging.getLogger(__name__)

from fnirsatlas.config import get_results_cache_dir

QUALITY_DIR = get_results_cache_dir() / "quality"
SCI_THRESHOLDS = [0.2, 0.5, 0.7]


def load_or_compute(dataset_name: str) -> pd.DataFrame | None:
    """Load pre-computed SCI CSV or run analysis if missing."""
    csv_path = QUALITY_DIR / f"{dataset_name}_sci_values.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    logger.info(f"No cached SCI for {dataset_name}, computing...")
    try:
        return analyze_dataset_sci(dataset_name, plot=False)
    except Exception as e:
        logger.error(f"Failed to compute SCI for {dataset_name}: {e}")
        return None


def summarize(dataset_name: str, df: pd.DataFrame) -> dict:
    """Compute summary statistics from per-channel SCI DataFrame."""
    sci = df["sci"]
    row = {
        "Dataset": format_dataset_name(dataset_name),
        "#Ch": df["channel"].nunique(),
        "Mean SCI": round(sci.mean(), 3),
        "Median SCI": round(sci.median(), 3),
    }
    for t in SCI_THRESHOLDS:
        pct = 100 * (sci < t).sum() / len(sci)
        row[f"Elim%(<{t})"] = round(pct, 1)
    return row


def format_latex_table(df: pd.DataFrame) -> str:
    formatters = {
        "#Ch": "{:.0f}",
        "Mean SCI": "{:.3f}",
        "Median SCI": "{:.3f}",
    }
    for t in SCI_THRESHOLDS:
        formatters[f"Elim%(<{t})"] = "{:.1f}"
    latex_output = df.style.hide(axis="index").format(formatters).to_latex()
    return latex_output.replace(r"[c]", "[t]")


def main():
    parser = argparse.ArgumentParser(description="Channel quality summary table")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets to include (default: all available)",
    )
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    datasets = args.datasets or get_available_datasets()

    rows = []
    for ds in datasets:
        df = load_or_compute(ds)
        if df is None or df.empty:
            logger.warning(f"Skipping {ds}: no data")
            continue
        rows.append(summarize(ds, df))

    if not rows:
        print("No data collected.")
        return

    result = pd.DataFrame(rows)
    if args.latex:
        print(format_latex_table(result))
    else:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
