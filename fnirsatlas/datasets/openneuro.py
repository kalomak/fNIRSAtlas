"""OpenNeuro source helpers."""

import logging
from collections.abc import Iterable
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .bids_base import get_bids_cache_root

log = logging.getLogger(__name__)


def _requests_session() -> requests.Session:
    """Create an HTTP session with retries for transient S3 failures."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def download_openneuro_dataset_files(
    dataset_id: str,
    dataset_code: str,
    path: str | Path | None,
    files: Iterable[str],
    force_update: bool = False,
) -> Path:
    """Download files into a MOABB-managed OpenNeuro dataset cache."""
    root = get_bids_cache_root(dataset_code, path)
    root.mkdir(parents=True, exist_ok=True)
    base_url = f"https://s3.amazonaws.com/openneuro.org/{dataset_id}"
    with _requests_session() as session:
        for relative_path in files:
            local_path = root / relative_path
            if local_path.is_file() and not force_update:
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading %s ...", relative_path)
            with session.get(
                f"{base_url}/{relative_path}", stream=True, timeout=120
            ) as response:
                response.raise_for_status()
                partial_path = local_path.with_suffix(f"{local_path.suffix}.part")
                with partial_path.open("wb") as output:
                    for chunk in response.iter_content(chunk_size=8192):
                        output.write(chunk)
                partial_path.replace(local_path)
    return root
