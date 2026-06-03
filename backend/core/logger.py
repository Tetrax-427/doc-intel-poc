import logging
import sys
from datetime import datetime, timezone


class StructuredLogger:
    """
    Thin wrapper around Python's stdlib logging that appends
    key=value pairs to every log line for easy grep/parsing.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _format_kwargs(self, kwargs: dict) -> str:
        if not kwargs:
            return ""
        return " | " + " ".join(f"{k}={v}" for k, v in kwargs.items())

    def info(self, message: str, **kwargs):
        self._logger.info(message + self._format_kwargs(kwargs))

    def warning(self, message: str, **kwargs):
        self._logger.warning(message + self._format_kwargs(kwargs))

    def error(self, message: str, **kwargs):
        self._logger.error(message + self._format_kwargs(kwargs))

    def debug(self, message: str, **kwargs):
        self._logger.debug(message + self._format_kwargs(kwargs))


def _configure_root_logger():
    """
    Configure the root logger once at import time.
    Uses a clean single-line format: timestamp | level | name | message
    """
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. uvicorn already set up logging)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


_configure_root_logger()


def get_logger(name: str) -> StructuredLogger:
    """
    Get a named StructuredLogger.

    Convention: use dotted module path as name.
        get_logger("routers.documents")
        get_logger("routers.query")
        get_logger("retrieval.classify")
    """
    return StructuredLogger(name)