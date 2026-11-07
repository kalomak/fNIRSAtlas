import logging
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from fnirsatlas.utils import CONFIG_DIR, map_values_to_sequential, deep_merge, resolve_path

logger = logging.getLogger(__name__)
config = None


def _resolve_override_file(config_path: Optional[str] = None) -> Optional[Path]:
    """Return the override file from `config_path` or `FNIRSATLAS_CONFIG`, if any."""
    requested = config_path or os.environ.get("FNIRSATLAS_CONFIG")
    if not requested:
        return None
    override_file = Path(requested).expanduser()
    if not override_file.exists():
        raise FileNotFoundError(f"Config file not found: {override_file}")
    return override_file


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML files with override support."""
    base_config_file = CONFIG_DIR / "config.yaml"
    override_file = _resolve_override_file(config_path)

    # Load base configuration
    if not base_config_file.exists():
        raise FileNotFoundError(
            f"Base configuration file not found: {base_config_file}"
        )

    with open(base_config_file, "r") as f:
        base_config = yaml.safe_load(f)

    if override_file is None:
        return base_config

    with open(override_file, "r") as f:
        return deep_merge(base_config, yaml.safe_load(f))


def get_config():
    if config is None:
        load_config()
    return config


def get_dataset_path(dataset_name: str, config_path: Optional[str] = None) -> Path:
    """Get the path for a specific dataset."""
    config = load_config(config_path)
    dataset_config = config["datasets"].get(dataset_name.lower())

    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")

    if "path" in dataset_config:
        return resolve_path(dataset_config["path"])

    raw_datasets_dir = config["paths"]["raw_datasets"]
    dataset_path = Path(raw_datasets_dir) / dataset_config["subpath"]
    return resolve_path(str(dataset_path))


def get_dataset_config(
    dataset_name: str, config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Get the configuration for a specific dataset."""
    config = load_config(config_path)

    parts = dataset_name.split("__")
    dataset_id = parts[0]
    dataset_config = config["datasets"].get(dataset_id.lower())

    # override with experiment-specific config
    if len(parts) == 2:
        experiment_config = dataset_config.pop(parts[1], {})
        dataset_config = deep_merge(dataset_config, experiment_config)

    if not dataset_config:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration")

    return dataset_config


def get_preprocessing_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get preprocessing configuration."""
    config = load_config(config_path)
    return config.get("preprocessing", {})


def get_parser_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get parser configuration."""
    config = load_config(config_path)
    return config.get("parser", {})


def get_cv_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get cross-validation configuration."""
    config = load_config(config_path)
    return config.get("cv", {})


def get_paper_models(
    key: str = "comparison_models", config_path: Optional[str] = None
) -> list[str]:
    """Get paper model lists from config."""
    config = load_config(config_path)
    return config.get("paper", {}).get(key, [])


def get_dataloader_config(
    device: str = "cpu", config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Get DataLoader configuration for the given device type."""
    config = load_config(config_path)
    profile = "gpu" if device == "cuda" else "cpu"
    return config.get("dataloader", {}).get(profile, {})


def get_processed_datasets_dir(config_path: Optional[str] = None) -> Path:
    """Get the processed datasets directory path."""
    config = load_config(config_path)
    processed_dir = config["paths"]["processed_datasets"]
    return resolve_path(processed_dir)


def get_results_cache_dir(config_path: Optional[str] = None) -> Path:
    """Get the computational results cache directory path."""
    config = load_config(config_path)
    return resolve_path(config["paths"]["results_cache"])


def get_results_dir(config_path: Optional[str] = None) -> Path:
    """Get the generated plots and tables directory path."""
    config = load_config(config_path)
    results_dir = config["paths"]["results"]
    return resolve_path(results_dir)


def load_classification_tasks_from_config(
    dataset_name: str, conditions: list = None
) -> list:
    """
    Load and resolve classification tasks for a given dataset from classification_tasks.yaml.
    """
    config_path = CONFIG_DIR / "classification_tasks.yaml"
    if not config_path.exists():
        return []
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    parts = dataset_name.split("__")
    dataset_id = parts[0]
    experiment = parts[1] if len(parts) > 1 else None

    dataset_cfg = config.get(dataset_id)
    tasks = dataset_cfg.get("classification_tasks") if dataset_cfg else None
    if isinstance(tasks, dict):
        tasks = tasks.get(experiment) if experiment else None
    if not tasks:
        return []
    result = []
    for t in tasks:
        if t.get("include", True) is False:
            continue
        mapping = map_values_to_sequential(t["mapping"])
        result.append({"description": t["description"], "mapping": mapping})
    return result


def get_classification_tasks(dataset_name: str, cond2label: dict = None) -> list:
    """
    Return a list of available classification tasks for a dataset.

    This function handles both config-based tasks and fallback logic.
    """
    conditions = list(cond2label.keys()) if cond2label else []
    config_tasks = load_classification_tasks_from_config(
        dataset_name,
        conditions=conditions,
    )
    if config_tasks:
        return config_tasks
    if cond2label:
        return [
            {
                "description": "|".join(sorted(cond2label.keys())),
                "mapping": map_values_to_sequential(cond2label),
            }
        ]
    return []


def get_param_grids() -> Dict[str, Any]:
    """Get parameter grids for baseline models."""
    grid_file = CONFIG_DIR / "param_grids.yaml"

    if not grid_file.exists():
        logger.warning(f"Parameter grid file not found: {grid_file}")
        return {}

    with open(grid_file, "r") as f:
        grids = yaml.safe_load(f)

    return grids


def get_param_grid(model_name: str) -> Optional[Dict[str, Any]]:
    """Get parameter grid for a specific baseline model."""
    grids = get_param_grids()
    return grids.get(model_name)


def get_model_params(
    model_name: str, config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Get default parameters for a specific model."""
    config = load_config(config_path)
    model_params = config.get("model_params", {})
    return model_params.get(model_name, {})
