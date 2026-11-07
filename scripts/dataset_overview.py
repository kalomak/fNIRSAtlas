"""
Script to load processed datasets and print basic information.
"""

import argparse
import logging
from fnirsatlas.utils import get_processed_data_path
from fnirsatlas.config import get_processed_datasets_dir
import numpy as np
from fnirsatlas.processed_dataset import load_processed_dataset
from fnirsatlas.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def analyze_processed_dataset(dataset_name: str):
    """Analyze a processed dataset."""
    processed_path = get_processed_data_path(dataset_name)

    if not processed_path.exists():
        logger.warning(f"Processed dataset not found: {processed_path}")
        return

    try:
        X, y, groups, info = load_processed_dataset(processed_path)

        print(f"=== {dataset_name.upper()} Dataset ===")
        print(f"Epochs:     {X.shape[0]}")
        print(f"Channels:   {X.shape[1]}")
        print(f"Timepoints: {X.shape[2]}")
        print(f"Subjects:   {len(np.unique(groups))}")
        print(f"Classes:    {len(np.unique(y))}")

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")

        # Get class names and find the longest for alignment
        class_names = []
        for class_id in unique:
            class_name = (
                info.get("label2cond", {}).get(class_id)
                or {v: k for k, v in info.get("cond2label", {}).items()}.get(class_id)
                or f"Class {class_id}"
            )
            class_names.append(class_name)

        max_name_length = max(len(name) for name in class_names)

        for class_id, count, class_name in zip(unique, counts, class_names):
            aligned_name = class_name.ljust(max_name_length)
            print(f"  {aligned_name}: {count}")

        print("")

    except Exception as e:
        logger.error(f"Error analyzing {dataset_name}: {e}")


def main():
    """Main function to analyze processed datasets."""
    parser = argparse.ArgumentParser(description="Analyze processed fNIRS datasets")
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        help="Specific datasets to analyze (default: all available processed datasets)",
    )

    args = parser.parse_args()

    # Find available processed datasets
    processed_dir = get_processed_datasets_dir()
    if not processed_dir.exists():
        logger.error(f"Processed datasets directory not found: {processed_dir}")
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
        datasets_to_analyze = available_datasets

    print("=== Processed fNIRS Dataset Overview ===\n")

    for dataset_name in datasets_to_analyze:
        analyze_processed_dataset(dataset_name)

    print("=== Analysis Complete ===")


if __name__ == "__main__":
    main()
