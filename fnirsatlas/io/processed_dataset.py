from pathlib import Path
from typing import Union
import numpy as np

from fnirsatlas.dataset_catalog import get_available_datasets


def _get_dataset_name(dataset_file_path: Path) -> str:
    return dataset_file_path.name.rsplit("_processed.npz", 1)[0]


def _find_similar_dataset_file(dataset_file_path: Path) -> list[str]:
    processed_dir = dataset_file_path.parent
    dataset_name = _get_dataset_name(dataset_file_path)
    suggestions = list(processed_dir.glob(f"{dataset_name}*"))
    return [
        _get_dataset_name(s)
        for s in suggestions
        if s.is_file() and s.name.endswith("_processed.npz")
    ]


def _resolve_dataset_path(dataset_file: Union[Path, str]) -> Path:
    from fnirsatlas.config import get_processed_datasets_dir

    dataset_path = Path(dataset_file)
    if dataset_path.exists():
        return dataset_path

    processed_dir = get_processed_datasets_dir()
    candidate = processed_dir / f"{dataset_path.name}_processed.npz"
    if candidate.exists():
        return candidate

    suggestions = _find_similar_dataset_file(candidate)
    msg = f"Dataset file not found: {candidate}"
    if suggestions:
        msg += f"\nDid you mean one of these?: {suggestions}"
    raise FileNotFoundError(msg)


def load_processed_dataset(dataset_file: Union[Path, str]):
    dataset_path = _resolve_dataset_path(dataset_file)
    loaded = np.load(dataset_path, allow_pickle=True)
    X = loaded["X"]
    y = loaded["y"]
    groups = loaded["groups"]
    info = loaded["info"].item() if hasattr(loaded["info"], "item") else loaded["info"]
    return X, y, groups, info


def is_processed_dataset(dataset_name: str) -> bool:
    from fnirsatlas.config import get_processed_datasets_dir

    processed_dir = get_processed_datasets_dir()
    return (processed_dir / f"{dataset_name}_processed.npz").exists()


def get_available_processed_datasets() -> list[str]:
    from fnirsatlas.config import get_processed_datasets_dir

    processed_dir = get_processed_datasets_dir()
    all_processed = [
        f.stem.replace("_processed", "") for f in processed_dir.glob("*_processed.npz")
    ]
    all_processed_set = set(all_processed)
    known_order = get_available_datasets()
    known = [dataset for dataset in known_order if dataset in all_processed_set]
    extras = sorted(dataset for dataset in all_processed if dataset not in known_order)
    return known + extras
