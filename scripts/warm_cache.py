"""Fill the MOABB cache for one dataset so parallel jobs only read it."""

import argparse
import logging

from moabb.datasets.base import CacheConfig

from fnirsatlas.benchmark import PARADIGMS, select_datasets

logger = logging.getLogger(__name__)

CACHE_CONFIG = CacheConfig(use=True, save_epochs=True, save_array=True)


def warm_cache(paradigm_name, dataset_code):
    """Load every subject of one dataset once, writing epoch and array caches."""
    paradigm = PARADIGMS[paradigm_name]()
    datasets = select_datasets(paradigm, include=[dataset_code])
    if not datasets:
        raise ValueError(f"{dataset_code} is not a {paradigm_name} dataset")
    dataset = datasets[0]
    for subject in dataset.subject_list:
        # Epochs and arrays are cached separately, so both forms are requested.
        for return_epochs in (True, False):
            try:
                paradigm.get_data(
                    dataset,
                    subjects=[subject],
                    return_epochs=return_epochs,
                    cache_config=CACHE_CONFIG,
                )
            # Every run of the subject was dropped, so one-subject loading has
            # nothing to concatenate. The evaluation loads all subjects at once
            # and skips this one.
            except ValueError as error:
                if "No objects to concatenate" not in str(error):
                    raise
                logger.warning(
                    "Skipping %s subject %s: no usable runs", dataset_code, subject
                )


def parse_args(argv=None):
    """Parse cache-warming command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: uv run python scripts/warm_cache.py "
            "--paradigm MentalArithmetic --dataset Bnci2014FnirsMA"
        ),
    )
    parser.add_argument("--paradigm", required=True, choices=PARADIGMS)
    parser.add_argument("--dataset", required=True, help="Dataset code")
    return parser.parse_args(argv)


def main(argv=None):
    """Warm the cache for the selected paradigm and dataset."""
    args = parse_args(argv)
    warm_cache(args.paradigm, args.dataset)


if __name__ == "__main__":
    main()
