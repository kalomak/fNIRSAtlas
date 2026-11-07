#!/usr/bin/env python3
"""
Script to download datasets using the fnirsatlas.download module.
"""

import argparse
import logging
from fnirsatlas.download import download_dataset
from fnirsatlas.utils import check_data_exists
from fnirsatlas.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download a single dataset")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset to download (e.g., nemo)",
    )
    return parser.parse_args()


def main():
    """Download a single dataset specified by command line arguments."""
    args = parse_arguments()
    dataset = args.dataset.lower()
    base = dataset.split("__")[0]
    if check_data_exists(dataset)["raw_exists"]:
        logger.info(f"Raw data already exists for {dataset} (base: {base}).")
        return 0
    if download_dataset(base):
        logger.info(f"Downloaded {base} successfully.")
        return 0
    logger.error(f"Failed to download {base}.")
    return 1


if __name__ == "__main__":
    exit(main())
