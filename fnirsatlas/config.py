from importlib.resources import files
from pathlib import Path

import yaml

from .utils import deep_merge_dicts


config = None


def _load_yaml(path):
    with path.open() as file:
        return yaml.safe_load(file) or {}


def _default_config_path():
    return files(__package__) / "config" / "config.yaml"


def load_config():
    """Load packaged defaults and optional project-local overrides."""
    global config
    global_config = _load_yaml(_default_config_path())
    local_path = Path.cwd() / "config" / "config_local.yaml"
    try:
        local_config = _load_yaml(local_path)
    except FileNotFoundError:
        local_config = {}
    config = deep_merge_dicts(global_config, local_config)
    return config


def get_config():
    if config is None:
        load_config()
    return config


def get_path_from_config(path_key):
    """Get a path from config with relative path handling.

    Paths starting with ``./`` are relative to the working directory.
    """
    path_str = get_config().get(path_key)
    if path_str is None:
        return None
    if path_str.startswith("./"):
        return Path.cwd() / path_str[2:]
    path_str = str(Path(path_str).expanduser())
    return Path(path_str)
