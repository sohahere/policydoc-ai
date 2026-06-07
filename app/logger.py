import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger.

    This logger prints clean timestamped logs to the terminal.
    It helps track pipeline steps such as loading, chunking,
    embedding, retrieval, and LLM generation.
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers when modules are reloaded
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger