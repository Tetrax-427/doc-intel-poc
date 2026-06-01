import logging
import json
from datetime import datetime, timezone


class StructuredLogger:
    """
    Every log entry is a JSON object with consistent fields.
    Makes logs searchable and parseable by any log aggregator.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

        # Avoid adding duplicate handlers if logger already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.DEBUG)
        # Prevent propagation to root logger to avoid duplicate output
        self.logger.propagate = False

    def _log(self, level: str, message: str, **kwargs):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs,
        }
        getattr(self.logger, level.lower())(json.dumps(entry))

    def info(self, message: str, **kwargs):
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("error", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("debug", message, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a named structured logger. Use the module or component name.

    Usage:
        logger = get_logger("ingestion")
        logger.info("Parsing file", doc_id="abc", file="invoice.pdf", parser="llamaparse")
        logger.error("Parse failed", doc_id="abc", error_code="PARSE_001", retryable=True)
    """
    return StructuredLogger(name)