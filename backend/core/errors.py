"""
DocIntel error taxonomy.

Error code namespaces:
    PARSE_0xx   — file parsing and ingestion
    CLASS_0xx   — document classification
    EMBED_0xx   — embedding generation
    RETRIEV_0xx — search and retrieval
    EXTRACT_0xx — field extraction
    VALID_0xx   — validation
    VISION_0xx  — vision model calls
    CONFIG_0xx  — configuration
"""


class DocIntelError(Exception):
    """Base exception for all DocIntel errors."""

    def __init__(
        self,
        message: str,
        code: str,
        severity: str = "ERROR",
        retryable: bool = False,
        context: dict = None,
    ):
        super().__init__(message)
        self.code = code
        self.severity = severity    # "ERROR", "WARNING", "INFO"
        self.retryable = retryable
        self.context = context or {}

    def to_dict(self) -> dict:
        return {
            "error": str(self),
            "code": self.code,
            "severity": self.severity,
            "retryable": self.retryable,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Parsing errors (PARSE_0xx)
# ---------------------------------------------------------------------------

class ParseError(DocIntelError):
    """Raised when a parser fails to process a file."""

    def __init__(self, message: str, file_name: str = "", retryable: bool = False):
        super().__init__(
            message,
            code="PARSE_001",
            retryable=retryable,
            context={"file_name": file_name},
        )


class UnsupportedFileTypeError(DocIntelError):
    """Raised when no parser exists for a given file extension."""

    def __init__(self, file_name: str, extension: str):
        super().__init__(
            f"No parser available for file type '{extension}'",
            code="PARSE_002",
            retryable=False,
            context={"file_name": file_name, "extension": extension},
        )


class EmptyDocumentError(DocIntelError):
    """Raised when a file is parsed but yields no extractable content."""

    def __init__(self, file_name: str):
        super().__init__(
            f"File '{file_name}' produced no extractable text",
            code="PARSE_003",
            retryable=False,
            context={"file_name": file_name},
        )


# ---------------------------------------------------------------------------
# Classification errors (CLASS_0xx)
# ---------------------------------------------------------------------------

class ClassificationError(DocIntelError):
    """Raised when document classification fails."""

    def __init__(self, message: str, doc_id: str = ""):
        super().__init__(
            message,
            code="CLASS_001",
            retryable=True,
            context={"doc_id": doc_id},
        )


# ---------------------------------------------------------------------------
# Embedding errors (EMBED_0xx)
# ---------------------------------------------------------------------------

class EmbeddingError(DocIntelError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="EMBED_001", retryable=True)


# ---------------------------------------------------------------------------
# Retrieval errors (RETRIEV_0xx)
# ---------------------------------------------------------------------------

class RetrievalError(DocIntelError):
    """Raised when vector search or reranking fails."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message, code="RETRIEV_001", retryable=retryable)


# ---------------------------------------------------------------------------
# Extraction errors (EXTRACT_0xx)
# ---------------------------------------------------------------------------

class ExtractionError(DocIntelError):
    """Raised when structured field extraction fails."""

    def __init__(self, message: str, field: str = ""):
        super().__init__(
            message,
            code="EXTRACT_001",
            retryable=True,
            context={"field": field},
        )


# ---------------------------------------------------------------------------
# Validation errors (VALID_0xx)
# ---------------------------------------------------------------------------

class ValidationError(DocIntelError):
    """Raised when extracted data fails validation rules."""

    def __init__(self, message: str, field: str = "", value: str = ""):
        super().__init__(
            message,
            code="VALID_001",
            severity="WARNING",
            retryable=False,
            context={"field": field, "value": value},
        )


# ---------------------------------------------------------------------------
# Vision errors (VISION_0xx)
# ---------------------------------------------------------------------------

class VisionError(DocIntelError):
    """Raised when a vision model call fails."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message, code="VISION_001", retryable=retryable)


# ---------------------------------------------------------------------------
# Config errors (CONFIG_0xx)
# ---------------------------------------------------------------------------

class ConfigError(DocIntelError):
    """Raised when required configuration is missing or invalid."""

    def __init__(self, message: str):
        super().__init__(
            message,
            code="CONFIG_001",
            severity="ERROR",
            retryable=False,
        )