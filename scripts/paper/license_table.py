"""
Script for creating a LaTeX license table for all datasets.

Example usage:
    python scripts/paper/license_table.py --output './../paper/nirsatlas/tex/results/license_table.tex'

"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from fnirsatlas.parsers import get_available_datasets, get_suite_for_dataset, SUITE_MAP
from fnirsatlas.logging import setup_logging
from scripts.paper.ds_table import (
    get_title,
    suite_display_mapping,
)

setup_logging()
logger = logging.getLogger(__name__)


def load_licenses():
    """Load license data from JSON file."""
    path = Path(__file__).with_name("licenses.json")
    with open(path) as f:
        return json.load(f)


def create_license_table(output_path=None):
    licenses = load_licenses()
    datasets = get_available_datasets()
    rows = []
    for ds in datasets:
        if ds not in licenses:
            logger.warning(f"No license for {ds}")
            continue
        suite = get_suite_for_dataset(ds) or ""
        suite_pretty = suite_display_mapping.get(suite, suite)
        rows.append({
            "Suite": suite_pretty,
            "Dataset": get_title(ds),
            "License": licenses[ds],
            "_suite_raw": suite,
        })
    df = pd.DataFrame(rows)
    suite_order = [suite_display_mapping.get(s, s) for s in SUITE_MAP]
    df["Suite"] = pd.Categorical(df["Suite"], categories=suite_order, ordered=True)
    df = df.sort_values(["Suite", "Dataset"]).drop(columns=["_suite_raw"])
    df = df.set_index(["Suite", "Dataset"])
    latex_output = df.style.to_latex()
    latex_output = latex_output.replace(r"[c]", "[t]")
    if output_path:
        Path(output_path).write_text(latex_output)
        logger.info(f"LaTeX table saved to {output_path}")
    else:
        print(latex_output)


def main():
    parser = argparse.ArgumentParser(description="Create LaTeX license table")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    create_license_table(output_path=args.output)


if __name__ == "__main__":
    main()
