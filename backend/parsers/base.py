"""
Abstract base class for all parsers.
The rest of the system only ever talks to BaseParser — never to specific parsers.
"""

from abc import ABC, abstractmethod
from core.document import Document
from core.config import Config


class BaseParser(ABC):
    """
    Every parser implements this interface.

    Rules:
    - parse() must return a fully populated Document
    - parse() must NEVER return None — raise ParseError on failure
    - parse() must log timing and outcome
    - parse() must fill metadata with parser_used, duration_ms, page_count
    """

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Return True if this parser can handle the given file extension."""
        ...

    @abstractmethod
    def parse(self, file_path: str, config: Config) -> Document:
        """
        Parse the file and return a fully populated Document.
        Never return None — raise ParseError if parsing fails.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a short parser identifier e.g. 'llamaparse', 'pypdf'."""
        ...

    def is_available(self, config: Config) -> bool:
        """
        Return True if this parser has all required API keys / dependencies.
        Default: True. Override for parsers that need external services.
        """
        return True