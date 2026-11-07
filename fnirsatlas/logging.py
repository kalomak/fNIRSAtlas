"""
Centralized logging configuration for fnirsatlas package.
"""

import logging
import os
from pathlib import Path
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    """Custom formatter with ANSI color codes."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)


def setup_logging(level=None, log_file=None):
    """Setup logging for fnirsatlas package.

    Parameters
    ----------
    level : str, optional
        Logging level. If None, uses FNIRSATLAS_LOG_LEVEL environment variable
        or defaults to 'WARNING'.
    log_file : str, optional
        Path to log file. If provided, logs will also be written to this file.
    """
    if level is None:
        level = os.getenv("FNIRSATLAS_LOG_LEVEL", "WARNING").upper()

    # Setup handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=handlers,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Apply colored formatter to console handler
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(ColoredFormatter("%(levelname)s:%(name)s:%(message)s"))

    # Suppress noisy third-party loggers
    for logger_name in ["mne", "sklearn", "numpy", "matplotlib", "tqdm", "datalad"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


@contextmanager
def set_log_level(level, logger=None):
    logger = logger or logging.getLogger()
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
