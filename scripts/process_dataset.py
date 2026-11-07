"""
Processes and saves datasets.
"""

from pathlib import Path
import numpy as np
import argparse
import logging
from fnirsatlas.config import get_dataset_path, get_processed_datasets_dir
from fnirsatlas.processed_dataset import to_dataset
from fnirsatlas.load_parser import load_parser
from fnirsatlas.utils import check_data_exists
from fnirsatlas.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process and save a single dataset")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Dataset to process (e.g., shin2016__ma)",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-process and overwrite saved dataset even if it already exists.",
    )
    return parser.parse_args()


def process_single_dataset(dataset_name: str, override: bool = False):
    """Process and save a single dataset."""
    logger.info(f"{'='*60}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*60}")

    data_status = check_data_exists(dataset_name)
    if data_status["processed_exists"] and not override:
        logger.info(
            f"Processed data found for {dataset_name} (use --override to rebuild)"
        )
        return True

    if not data_status["raw_exists"]:
        logger.error(f"Raw data not found for {dataset_name}. Skipping.")
        return False

    try:
        parser = load_parser(dataset_name, download="no")
    except RuntimeError as e:
        logger.error(f"Failed to load parser for {dataset_name}: {e}")
        return False

    subject_ids = parser.get_subject_ids()
    ed = parser.get_epochs(subject_ids=subject_ids)

    X, y, groups, info = to_dataset(ed, dataset_name=dataset_name)

    save_dir = get_processed_datasets_dir()
    save_file = save_dir / f"{dataset_name}_processed.npz"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_file, X=X, y=y, groups=groups, info=info)
    logger.info(
        f"Saved to: {save_file} ({save_file.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    return True


def main():
    """Process and save a single dataset specified by command line arguments."""
    args = parse_arguments()
    ok = process_single_dataset(args.dataset.lower(), override=args.override)
    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
