"""
Download functionality for datasets.

This module provides functions to download datasets automatically based on
configuration in downloads.yaml.
"""

import zipfile
import yaml
import requests
from tqdm import tqdm
from pathlib import Path
import tarfile
import re
import shutil
import logging
import sys
from typing import Dict, Any, Optional
from fnirsatlas import __version__
from fnirsatlas.config import load_config
from fnirsatlas.utils import get_tqdm_env_kwargs
from fnirsatlas.utils import CONFIG_DIR, resolve_path
from fnirsatlas.logging import set_log_level

logger = logging.getLogger(__name__)


def parse_size(size_str: str) -> Optional[int]:
    """Parse size string like '1.5GB' into bytes."""
    size_str = size_str.strip().upper()
    match = re.match(r"([\d.]+)\s*([KMG]?B)", size_str)
    if not match:
        return None
    num, unit = match.groups()
    num = float(num)
    if unit == "B":
        return int(num)
    elif unit == "KB":
        return int(num * 1_000)
    elif unit == "MB":
        return int(num * 1_000_000)
    elif unit == "GB":
        return int(num * 1_000_000_000)
    return None


def resolve_download_destination(destination: str) -> Path:
    """Resolve download destinations against the active raw-data root."""
    dest_path = Path(destination)
    if dest_path.is_absolute():
        return dest_path
    if dest_path.parts and dest_path.parts[0] == "datasets":
        raw_root = Path(load_config()["paths"]["raw_datasets"]).expanduser()
        return raw_root.joinpath(*dest_path.parts[1:])
    return resolve_path(f"./{destination}")


def parse_filepath(entry: Dict[str, Any]) -> Path:
    destination = resolve_download_destination(entry["destination"])
    if "filename" in entry:
        return destination / entry["filename"]
    return destination


def download_file(entry: Dict[str, Any]) -> bool:
    """Download a file over HTTP(S) based on entry configuration."""
    url = entry["url"]
    dest_path = parse_filepath(entry)
    if dest_path.exists():
        logger.info(f"File already exists, skipping download: {dest_path}")
        return True
    try:
        if _is_datalad_entry(entry):
            return download_with_datalad(entry)
        else:
            return http_download(entry)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def _is_datalad_entry(entry: Dict[str, Any]) -> bool:
    """Heuristic to detect DataLad/git-annex style datasets."""
    url = str(entry.get("url", ""))
    return url.endswith(".git")


def http_download(
    entry: Dict[str, Any],
) -> bool:
    url = entry["url"]
    dest_path = parse_filepath(entry)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = entry.get("headers", {})
    size = entry.get("size", None)
    if isinstance(size, str):
        size = parse_size(size)
    if "User-Agent" not in headers:
        headers["User-Agent"] = (
            f"fnirsatlas/{__version__} (https://github.com/kalomak/fNIRSAtlas; fNIRS classification benchmark)"
        )
    with requests.get(url, stream=True, headers=headers) as r:
        r.raise_for_status()
        total = (
            size if size is not None else int(r.headers.get("content-length", 0))
        )
        desc = f"{'Downloading':11} {entry['dataset_name'][:12]:<12}"
        with open(dest_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=desc,
            **get_tqdm_env_kwargs(),
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    return True


def download_with_datalad(entry: Dict[str, Any]) -> bool:
    """Download a dataset using DataLad (git-annex-backed)."""
    try:
        from datalad import api as dl
        from datalad.distribution.dataset import Dataset  # type: ignore[import]
    except Exception as e:  # pragma: no cover - import error path
        raise RuntimeError(
            "DataLad is required to download this dataset. "
            "Please install DataLad in the active environment."
        ) from e
    dest_root = resolve_download_destination(entry["destination"])
    url = entry["url"]
    # This silences everything, not just info logs
    # Ideally we would keep progress bar but not sure how to do that
    # with set_log_level(logging.WARNING, logging.getLogger("datalad")):
    ds = dl.clone(source=url, path=str(dest_root))
    try:
        dl.get(dataset=ds)
    except Exception as e:
        raise RuntimeError(
            f"Failed to retrieve annexed content for dataset '{entry['dataset_name']}': {e}"
        ) from e
    return True

def extract_archive_simple(archive_path: Path, extract_to: Path) -> bool:
    """Extract a single archive to the specified destination."""
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as arc_ref:
                arc_ref.extractall(extract_to)
        elif (
            archive_path.suffixes[-2:] == [".tar", ".gz"]
            or archive_path.suffix == ".tgz"
        ):
            with tarfile.open(archive_path, "r:gz") as arc_ref:
                arc_ref.extractall(extract_to)
        else:
            logger.warning(f"Unsupported archive type: {archive_path}")
            return False
        logger.info(f"Successfully extracted: {archive_path}")

        # Delete the archive file after successful extraction
        try:
            archive_path.unlink()
            logger.info(f"Deleted archive file: {archive_path}")
        except Exception as e:
            logger.warning(f"Failed to delete archive file {archive_path}: {e}")

        return True
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        try:
            archive_path.unlink()
            logger.info(f"Deleted corrupted archive file: {archive_path}")
        except Exception as del_e:
            logger.warning(f"Failed to delete archive file {archive_path}: {del_e}")
        return False


def replace_part(obj: Any, part: str) -> Any:
    """Replace {part} with the given part."""
    if isinstance(obj, str):
        return obj.replace("{part}", part)
    elif isinstance(obj, list):
        return [replace_part(x, part) for x in obj]
    elif isinstance(obj, dict):
        return {k: replace_part(v, part) for k, v in obj.items()}
    else:
        return obj


def handle_entry(entry: Dict[str, Any]) -> bool:
    """Handle a single download entry.

    Handles download, optional extractions, and renames.
    """
    if not download_file(entry):
        return False

    dest_dir = resolve_download_destination(entry["destination"])

    if "extractions" in entry:
        for extraction in entry["extractions"]:
            archive_path = dest_dir / extraction["archive"]
            extract_to = dest_dir / extraction["destination"]
            extract_to.mkdir(parents=True, exist_ok=True)

            # If extraction fails, try to redownload and extract again
            if not extract_archive_simple(archive_path, extract_to):
                logger.warning(
                    f"Extraction failed, attempting to redownload: {archive_path}"
                )
                if not download_file(entry):
                    return False
                if not extract_archive_simple(archive_path, extract_to):
                    return False

    # Handle renames
    if "renames" in entry:
        for r in entry["renames"]:
            src = dest_dir / r["from"]
            dst = dest_dir / r["to"]
            try:
                shutil.move(str(src), str(dst))
                logger.info(f"Renamed: {src} -> {dst}")
            except Exception as e:
                logger.error(f"Failed to rename {src} to {dst}: {e}")
                return False

    return True

def load_download_config() -> Dict[str, Any]:
    """Load the download configuration from downloads.yaml."""
    with open(CONFIG_DIR / "downloads.yaml", "r") as f:
        return yaml.safe_load(f)


def download_dataset(dataset_name: str) -> bool:
    """
    Download a dataset automatically.

    Args:
        dataset_name: Name of the dataset to download

    Returns:
        True if download was successful, False otherwise
    """
    logger.info(f"Downloading dataset: {dataset_name}")

    if len(dataset_name.split("__")) > 1:
        dataset_name = dataset_name.split("__")[0]

    config = load_download_config()
    if dataset_name not in config:
        raise RuntimeError(f"Dataset '{dataset_name}' not found in downloads.yaml")

    entry = config[dataset_name]
    entry["dataset_name"] = dataset_name
    download_type = entry.get("type", "automatic")

    # Handle different download types
    if download_type == "manual" or download_type == "request":
        msg = f"Manual download required for dataset '{dataset_name}'. "
        msg += f"Please download from: [ {entry['url']} ]. "
        msg += f"Then place the downloaded data in: [ {entry['destination']} ], "
        msg += f"or set the path in your `FNIRSATLAS_CONFIG` file."
        raise RuntimeError(msg)
    elif download_type == "automatic":
        if "parts" in entry:
            parts = entry.pop("parts")
            nparts = len(parts)
            for part_ix, part in enumerate(parts):
                part_entry = replace_part(entry, part)
                part_entry["dataset_name"] = f"({part_ix+1}/{nparts}) {dataset_name}"
                if not handle_entry(part_entry):
                    raise RuntimeError(
                        f"Failed to download part '{part}' for dataset '{dataset_name}'"
                    )
        else:
            if not handle_entry(entry):
                raise RuntimeError(f"Failed to download dataset '{dataset_name}'")

        logger.info(f"Successfully downloaded dataset: {dataset_name}")
        return True
    else:
        logger.error(
            f"Unknown download type '{download_type}' for dataset '{dataset_name}'"
        )
        return False


def is_in_dataset_yaml(dataset_name: str) -> bool:
    """Check if a dataset is available in downloads.yaml."""
    config = load_download_config()
    return dataset_name in config
