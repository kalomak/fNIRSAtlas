import logging
import sys
from pathlib import Path

from .utils import deep_merge_dicts
from .config import get_config


def setup_logging(config_dict=None):
    """Setup project-wide logging with compact format"""
    if config_dict is not None:
        config_dict = deep_merge_dicts(get_config(), config_dict)
    else:
        config_dict = get_config()

    # Get logging config
    log_config = config_dict.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(levelname)s: %(message)s")
    log_file = log_config.get("file")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = ColoredFormatter(format_str)

    # Add stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(level)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Add file handler if specified
    if log_file and log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        # Use regular formatter for file (no colors in log files)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and single letter level names"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Get first letter of level name
        level_letter = record.levelname[0]

        # Add color if available
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        # Format: I: message (with color)
        record.levelname = f"{color}{level_letter}{reset}"

        return super().format(record)
