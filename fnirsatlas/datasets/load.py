from __future__ import annotations
from typing import Tuple, Dict, Any
import logging
from pathlib import Path
import numpy as np
from fnirsatlas.io.processed_dataset import load_processed_dataset
from fnirsatlas.config import get_processed_datasets_dir
from fnirsatlas.utils import get_processed_data_path

logger = logging.getLogger(__name__)


def load_dataset(
    name_or_path: str,
    download: str = "ask",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Scikit-learn compatible loader that ensures data is available.

    Behavior:
    - If `name_or_path` points to an existing file, load it as a processed dataset.
    - Otherwise, treat it as a dataset name. If processed data exists, load it.
    - Otherwise, download raw data if missing, process it, and then load.

    Parameters
    ----------
    name_or_path : str
        Dataset name (e.g. "bnci") or path to a processed ``.npz`` file.
    download : str
        Raw data download behavior when raw data is missing: "ask" prompts
        before downloading, "yes" downloads without prompting, "no" never
        downloads.
    """
    p = Path(name_or_path)
    if p.exists():
        logger.info(f"Loading processed dataset from file: {p}")
        return load_processed_dataset(p)

    dataset_name = p.name.lower()
    # Resolve a base name to its default experiment, e.g. shin2016 -> shin2016__ma.
    # Skipped if processed data exists under the given name, since resolving imports the parsers.
    if not get_processed_data_path(dataset_name).exists():
        dataset_name = resolve_dataset_id(dataset_name)

    processed_file = get_processed_data_path(dataset_name)
    if processed_file.exists():
        logger.info(f"Loading processed dataset: {dataset_name}")
        return load_processed_dataset(processed_file)

    # Building from raw needs the `process` extra (mne). Reading processed data does not.
    from fnirsatlas.load_parser import load_parser
    from fnirsatlas.processed_dataset import to_dataset

    logger.info(f"Processed dataset not found for {dataset_name}. Building from raw...")
    parser = load_parser(dataset_name, download=download)
    subject_ids = parser.get_subject_ids()
    epochs_dict = parser.get_epochs(subject_ids=subject_ids)
    X, y, groups, info = to_dataset(epochs_dict, dataset_name=dataset_name)

    out_dir = get_processed_datasets_dir()
    out_path = out_dir / f"{dataset_name}_processed.npz"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y, groups=groups, info=info)
    logger.info(
        f"Saved processed dataset to: {out_path} "
        f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)"
    )

    return X, y, groups, info


def resolve_dataset_id(base_name: str) -> str:
    """Resolves a dataset name without an experiment to a canonical id with a default experiment."""
    if "__" in base_name:
        return base_name

    try:
        from fnirsatlas.parsers import get_parser_class

        parser_cls = get_parser_class(base_name)
        experiments = getattr(parser_cls, "EXPERIMENTS", []) or []
        if len(experiments) == 0:
            return base_name
        else:
            primary = str(experiments[0]).lower()
            choice = f"{base_name}__{primary}"
            logger.info(
                f"Interpreting {base_name} as {choice}. Experiments: {experiments}"
            )
            return choice
    except Exception:
        pass

    return base_name
