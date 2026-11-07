from __future__ import annotations
from typing import Tuple, Dict, Any
import logging
from pathlib import Path
import numpy as np
from fnirsatlas.io.processed_dataset import load_processed_dataset
from fnirsatlas.processed_dataset import to_dataset
from fnirsatlas.load_parser import load_parser
from fnirsatlas.config import get_processed_datasets_dir
from fnirsatlas.utils import check_data_exists
from fnirsatlas.parsers import get_parser_class

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

    # Resolve base dataset name into a canonical id with default experiment if needed
    dataset_name = resolve_dataset_id(p.name.lower())
    state = check_data_exists(dataset_name)

    if state.get("processed_exists", False):
        logger.info(f"Loading processed dataset: {dataset_name}")
        processed_file = get_processed_datasets_dir() / f"{dataset_name}_processed.npz"
        return load_processed_dataset(processed_file)

    # Process from raw
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
