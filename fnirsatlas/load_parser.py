from fnirsatlas.utils import check_data_exists
from fnirsatlas.parsers import PARSER_MAP
from fnirsatlas.download import download_dataset, is_in_dataset_yaml
import logging

logger = logging.getLogger(__name__)


def handle_dataset_download(dataset_id: str, download: str = "ask") -> bool:
    """
    Handle dataset download based on download parameter.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to check/download
    download : str
        Download behavior: "ask", "yes", "no"

    Returns
    -------
    bool
        True if data is available (either existed or was downloaded), False otherwise
    """
    data_status = check_data_exists(dataset_id)
    if data_status["raw_exists"]:
        return True

    if download == "ask":
        # Ask user for confirmation
        response = (
            input(f"Raw data not found for {dataset_id}. Download? (y/n): ")
            .lower()
            .strip()
        )
        if response not in ["y", "yes"]:
            logger.info(f"Skipping download for {dataset_id}")
            raise RuntimeError(f"Download cancelled for dataset {dataset_id}")

    if not is_in_dataset_yaml(dataset_id):
        raise RuntimeError(f"Dataset {dataset_id} not found in fnirsatlas/configs/downloads.yaml")

    logger.info(f"Raw data not found for {dataset_id}. Attempting to download...")
    if download_dataset(dataset_id):
        logger.info(f"Successfully downloaded {dataset_id}")
        return True
    else:
        raise RuntimeError(f"Failed to download dataset {dataset_id}")


def load_parser(dataset_name: str, **kwargs):
    """
    Load a parser for a given dataset name with automatic download handling.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'nemo', 'nemo__afim')
    download : str, optional
        Download behavior: "ask", "yes", "no". Default is "ask".
        If "ask" or "yes" and data doesn't exist, will attempt to download.
        "ask" will prompt for user confirmation before downloading.

    Returns
    -------
    parser : BaseFNIRSParser
    """
    download = kwargs.pop("download", "ask")
    dataset_name = dataset_name.lower()
    parts = dataset_name.split("__")
    dataset_id = parts[0]
    if download in ["ask", "yes"]:
        handle_dataset_download(dataset_id, download)
    if len(parts) > 1:
        experiment = parts[1]
        parser = PARSER_MAP[dataset_id](experiment=experiment, **kwargs)
    else:
        parser = PARSER_MAP[dataset_id](**kwargs)
    return parser
