"""Logging for PawPal+: every agent step, tool call and retry goes to pawpal.log and the console."""

import logging
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent / "pawpal.log"


def setup_logging(level: int = logging.INFO, log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure the "pawpal" logger once and return it. Safe to call on every Streamlit rerun."""
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
