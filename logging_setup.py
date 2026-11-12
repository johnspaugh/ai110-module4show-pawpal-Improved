"""Logging for PawPal+: every agent step, tool call and retry goes to pawpal.log and the console."""

import logging
import os
from pathlib import Path
from typing import Optional

LOG_FILE = Path(__file__).resolve().parent / "pawpal.log"


def setup_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure the "pawpal" logger once and return it. Safe to call on every Streamlit rerun.

    The file is log_file if given, else $PAWPAL_LOG_FILE (the tests point this at a temp file so
    they never write into the real pawpal.log), else pawpal.log next to this file.
    """
    log_file = log_file or Path(os.getenv("PAWPAL_LOG_FILE") or LOG_FILE)
    logger = logging.getLogger("pawpal")
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False  # keep Streamlit's root logger from printing everything twice
    return logger
