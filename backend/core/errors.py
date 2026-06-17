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
    LLM_0xx     — LLM engine, fallback chain, structured outputs
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


# ---------------------------------------------------------------------------
# LLM engine errors (LLM_0xx)
# ---------------------------------------------------------------------------

class LLMError(DocIntelError):
    """
    Raised when a single LLM provider call fails.
    Carries the provider/model that failed for logging in the fallback loop.
    """

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        retryable: bool = True,
    ):
        super().__init__(
            message,
            code="LLM_001",
            severity="ERROR",
            retryable=retryable,
            context={"provider": provider, "model": model},
        )


class LLMFallbackExhaustedError(DocIntelError):
    """
    Raised when every provider in the fallback chain has failed.
    The `providers_tried` context field lists each (provider, model) pair
    that was attempted, in order, for post-mortem logging.
    """

    def __init__(self, providers_tried: list[tuple[str, str]]):
        tried_str = ", ".join(f"{p}:{m}" for p, m in providers_tried)
        super().__init__(
            f"All LLM providers in fallback chain failed. Tried: {tried_str}",
            code="LLM_002",
            severity="ERROR",
            retryable=False,
            context={"providers_tried": providers_tried},
        )


class LLMProviderOverrideError(DocIntelError):
    """
    Raised when a per-call provider/model override fails and no fallback
    is attempted (override calls are single-shot by design — C3).
    """

    def __init__(self, provider: str, model: str, reason: str = ""):
        super().__init__(
            f"Provider override failed for {provider}:{model}. {reason}".strip(),
            code="LLM_003",
            severity="ERROR",
            retryable=False,
            context={"provider": provider, "model": model, "reason": reason},
        )


class StructuredOutputError(DocIntelError):
    """
    Raised when Instructor fails to coerce an LLM response into the
    requested Pydantic model after exhausting its internal retries.
    Carries the response_model name and the provider/model involved.
    """

    def __init__(
        self,
        message: str,
        response_model_name: str = "",
        provider: str = "",
        model: str = "",
    ):
        super().__init__(
            message,
            code="LLM_004",
            severity="ERROR",
            retryable=True,   # outer fallback chain may succeed on next provider
            context={
                "response_model": response_model_name,
                "provider": provider,
                "model": model,
            },
        )


class LLMConfigError(DocIntelError):
    """
    Raised for LLM-specific configuration problems:
    - unsupported provider name in fallback chain
    - missing API key for a provider that appears in the chain
    - Instructor adapter not available for a provider
    """

    def __init__(self, message: str, provider: str = ""):
        super().__init__(
            message,
            code="LLM_005",
            severity="ERROR",
            retryable=False,
            context={"provider": provider},
        )