"""
Script to load processed datasets and create a pandas DataFrame with basic information.

Example usage:
    python scripts/paper/ds_table.py --latex-output './../paper/nirsatlas/tex/results/big_dataset_table.tex'
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from fnirsatlas.processed_dataset import load_processed_dataset
from fnirsatlas.logging import setup_logging
from fnirsatlas.config import get_classification_tasks, get_processed_datasets_dir
from fnirsatlas.parsers import get_available_datasets, get_suite_for_dataset
from fnirsatlas.utils import escape_latex

setup_logging()
logger = logging.getLogger(__name__)


def load_raw_types():
    """Load raw types data from JSON file."""
    raw_types_file = Path("scripts/paper/raw_types.json")
    if raw_types_file.exists():
        with open(raw_types_file, "r") as f:
            return json.load(f)
    else:
        logger.warning(
            "scripts/paper/raw_types.json not found. Run check_raw_types.py first."
        )
        return {}


def load_raw_hours():
    """Load raw hours data from JSON file."""
    raw_hours_file = Path(__file__).with_name("raw_hours.json")
    if raw_hours_file.exists():
        with open(raw_hours_file, "r") as f:
            return json.load(f)
    else:
        logger.warning(
            "scripts/paper/raw_hours.json not found. Run raw_hours.py first."
        )
        return {}


def load_devices():
    """Load device data from JSON file."""
    devices_file = Path("scripts/paper/devices.json")
    if devices_file.exists():
        with open(devices_file, "r") as f:
            return json.load(f)
    else:
        logger.warning("scripts/paper/devices.json not found.")
        return {}


dataset_orig_sfreq = {
    "buccino2016": 10.42,
    "bak2019": 13.33,
    "ortega2020": 12.5,
    "luke2021": 7.81,
    "cockx2023": 50.0,
    "guevara2024": 50.0,
    "luke2024b": 8.93,
    "gao2024": 8.72,
    "bnci": 10.0,
    "shin2016__ma": 10.0,
    "shin2018b": 10.0,
    "shin2019__ma": 13.33,
    "shin2016__mi": 10.0,
    "shin2019__mi": 13.33,
    "yi2025": 7.81,
    "herff2014": 25.0,
    "shin2018__nback": 10.0,
    "huang2021": 5.2,
    "nemo__empe": 50.0,
    "nemo__afim": 50.0,
    "shin2018__wg": 10.0,
    "shader2021": 3.91,
    "luke2024a": 5.21,
}


dataset_citation_mapping = {
    "YI2025": "yi2025multi",
    "BAK2019": "bak2019open",
    "BNCI": "bauernfeind2011single",
    "BUCCINO2016": "buccino2016hybrid",
    "SHIN2019__MI": "shin2018ternary",
    "SHIN2019__MA": "shin2018ternary",
    "HERFF2014": "herff2013classification",
    "GUEVARA2024": "guevara2024parkinson",
    "NEMO__AFIM": "spape2023nemo",
    "NEMO__EMPE": "spape2023nemo",
    "COCKX2023": "cockx2023fnirs",
    "SHIN2016__MA": "shin2016open",
    "SHIN2016__MI": "shin2016open",
    "SHIN2018B": "shin2018eyes",
    "SHADER2021": "shader2021use",
    "HUANG2021": "huang2021tufts",
    "ORTEGA2020": "ortega2020hygrip",
    "LUKE2021": "luke2021finger",
    "SHIN2018__NBACK": "shin2018simultaneous",
    "SHIN2018__WG": "shin2018simultaneous",
    "LUKE2024B": "yucel2024fnirs",
    "LUKE2024A": "yucel2024fnirs",
    "GAO2024": "gao2023short",
}


suite_display_mapping = {
    "motor": "Motor",
    "nback": "N-back",
}


task_display_mapping = {
    "shin2018__nback": "N-back",
    "shin2018__wg": "Word Gen.",
    "shader2021": "Aud-Vis",
    "luke2024a": "Auditory",
}


raw_type_display_mapping = {
    "HbO/HbR": "Hb",
}


def format_class_distribution(class_dist) -> str:
    mappings = {
        "afim_HANV": "HANV",
        "afim_HAPV": "HAPV",
        "afim_LANV": "LANV",
        "afim_LAPV": "LAPV",
        "Mental Arithmetic": "MA",
        "Motor Imagery": "MI",
        "Idle State": "Idle",
        "_": " ",
    }
    if len(class_dist) > 4:
        n_per_class = int(np.mean(list(class_dist.values())).round())
        return f"~{n_per_class}*{len(class_dist)}"
    out = ""
    for n, c in class_dist.items():
        for old, new in mappings.items():
            n = n.replace(old, new)
        if n.lower() == n:
            n = n.capitalize()
        out += f"{n}={c}"
    return out[:-1]


def format_hz(value: float) -> str:
    rounded = round(value, 1)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.1f}"


def format_suite_name(suite_name: str) -> str:
    return suite_display_mapping.get(suite_name, suite_name)


def format_dataset_part(part: str) -> str:
    return suite_display_mapping.get(part, part)


def get_display_suite(dataset_name: str, task_description: str) -> str:
    suite_name = get_suite_for_dataset(dataset_name) or ""
    if suite_name == "MISC" and task_description:
        return task_description
    return format_suite_name(suite_name)


def format_dataframe_latex(df: pd.DataFrame) -> str:
    """Format pandas DataFrame as LaTeX tabular with suite separators."""
    content = df.to_latex(index=False, column_format="@{}llrrrrrrlrl@{}").split("\n")
    for i, line in enumerate(content):
        if "\\begin{tabular}" in line:
            start_idx = i
        if "\\end{tabular}" in line:
            end_idx = i
            break
    content = content[start_idx : end_idx + 1]
    for idx in [24, 22, 19, 16, 12]:
        content.insert(idx, format_suite())
    content.insert(len(content) - 3, format_suite())
    return "\n".join(content)


def format_suite():
    return "\\midrule"


def get_title(n):
    parts = n.split("__")
    nf = parts[0].capitalize()
    if len(parts) > 1:
        nf += f" ({format_dataset_part(parts[1])})"
    return rf"{nf} \cite{{{dataset_citation_mapping[n.upper()]}}}"


def get_task(dataset_name, info):
    """For hardcoding, mostly if def task does not include all data."""
    mask = None
    match dataset_name.lower():
        case name if name in task_display_mapping:
            description = task_display_mapping[name]
        case "yi2025":
            description = "Upper Limb MI"
        case "luke2021":
            description = "Rest|TapFinger[L|R]"
        case "bak2019":
            description = "Tap[Foot|Finger[L|R]]"
        case "cockx2023":
            description = "Tap[Foot|FingerR]"
        case "guevara2024":
            description = "TapFinger[L|R]"
        case _:
            tasks = get_classification_tasks(
                dataset_name, cond2label=info.get("cond2label")
            )
            if tasks:
                description = escape_latex(tasks[0]["description"])
            else:
                description = ""
    return description, mask


def count_participants(subjects_by_dataset: dict) -> int:
    """Count unique participants. Tasks of one dataset (e.g. shin2016__ma, shin2016__mi) share a cohort."""
    cohorts = {}
    for dataset_name, subjects in subjects_by_dataset.items():
        cohorts.setdefault(dataset_name.split("__")[0], set()).update(subjects)
    return sum(len(subjects) for subjects in cohorts.values())


def add_totals_row(df: pd.DataFrame, n_participants: int) -> pd.DataFrame:
    totals = {column: "" for column in df.columns}
    totals["\\#Sub"] = n_participants
    totals["\\#Epo"] = int(pd.to_numeric(df["\\#Epo"], errors="coerce").sum())
    hours = pd.to_numeric(df["Hours"], errors="coerce")
    if hours.notna().any():
        totals["Hours"] = f"{hours.sum():.1f}"
    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)


def analyze_processed_dataset(
    dataset_name: str,
    raw_types: dict = None,
    raw_hours: dict = None,
    devices: dict = None,
):
    """Analyze a processed dataset. Returns the table row and the subject ids."""
    processed_path = get_processed_datasets_dir() / f"{dataset_name}_processed.npz"
    X, y, groups, info = load_processed_dataset(processed_path)
    task_description, mask = get_task(dataset_name, info)
    if mask is not None:
        (
            X,
            y,
            groups,
        ) = (
            X[mask],
            y[mask],
            groups[mask],
        )
    data = {
        "Dataset": dataset_name.upper(),
        "Suite": get_display_suite(dataset_name, task_description),
        "\\#Sub": len(np.unique(groups)),
        "\\#Epo": X.shape[0],
        "\\#Ch": X.shape[1],
        "\\#Cl": len(np.unique(y)),
        "Hz": format_hz(dataset_orig_sfreq[dataset_name])
        if dataset_name in dataset_orig_sfreq
        else "",
        "Len": round(info["epoch_duration"]),
    }

    # Add raw type if available
    if raw_types and dataset_name in raw_types:
        raw_type = raw_types[dataset_name]
        data["Raw"] = raw_type_display_mapping.get(raw_type, raw_type)
    else:
        data["Raw"] = "Unknown"

    # Add raw hours if available
    if raw_hours and dataset_name in raw_hours:
        data["Hours"] = f"{raw_hours[dataset_name]:.1f}"
    else:
        data["Hours"] = ""

    if devices and dataset_name in devices:
        data["Device"] = devices[dataset_name]
    else:
        data["Device"] = ""

    c2l = info["cond2label"]
    data["Dataset"] = get_title(dataset_name)
    return data, set(groups)


def main():
    """Main function to analyze processed datasets and create a pandas DataFrame."""
    parser = argparse.ArgumentParser(description="Analyze processed fNIRS datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to analyze (default: all available processed datasets)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (optional)",
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        help="Output LaTeX file path (optional)",
    )
    args = parser.parse_args()

    # Load raw types data
    raw_types = load_raw_types()
    raw_hours = load_raw_hours()
    devices = load_devices()

    processed_dir = get_processed_datasets_dir()
    if not processed_dir.exists():
        logger.error("No processed_datasets directory found")
        return
    available_datasets = []
    for file in processed_dir.glob("*_processed.npz"):
        dataset_name = file.stem.replace("_processed", "")
        available_datasets.append(dataset_name)
    if not available_datasets:
        logger.error("No processed datasets found")
        return

    # Determine which datasets to analyze
    if args.datasets:
        datasets_to_analyze = args.datasets
    else:
        suite_order = get_available_datasets()
        datasets_to_analyze = [ds for ds in suite_order if ds in available_datasets]

    # Collect data for all datasets
    all_data = []
    subjects_by_dataset = {}
    for dataset_name in datasets_to_analyze:
        dataset_data, subjects_by_dataset[dataset_name] = analyze_processed_dataset(
            dataset_name,
            raw_types,
            raw_hours,
            devices,
        )
        all_data.append(dataset_data)

    if not all_data:
        logger.error("No valid dataset data collected")
        return

    n_participants = count_participants(subjects_by_dataset)
    df = add_totals_row(pd.DataFrame(all_data), n_participants)
    latex_output = format_dataframe_latex(df)

    # Generate LaTeX output
    print("\n=== LaTeX Output ===")
    print(latex_output)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nDataFrame saved to: {args.output}")

    if args.latex_output:
        with open(args.latex_output, "w") as f:
            f.write(latex_output)
        print(f"LaTeX table saved to: {args.latex_output}")

    return df


if __name__ == "__main__":
    df = main()
    print(df.shape)
