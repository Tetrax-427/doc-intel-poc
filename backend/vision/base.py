# backend/vision/base.py

from abc import ABC, abstractmethod
from core.config import Config


class BaseVisionModel(ABC):
    """
    Abstract base for all vision model implementations.

    Contract:
    - describe() NEVER raises — returns "" on any failure.
    - Vision is always optional. The system must work identically
      with NoVisionModel as with any real model.
    """

    @abstractmethod
    def describe(self, image_path: str, prompt: str) -> str:
        """
        Generate a natural language description of an image.

        Args:
            image_path: Absolute path to the image file.
            prompt:     Classification-aware prompt string.

        Returns:
            Description string, or "" on any failure.
            NEVER raises.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the provider name string (e.g. 'openai', 'anthropic', 'none')."""
        ...

    @abstractmethod
    def is_available(self, config: Config) -> bool:
        """
        Return True if this model can be used with the given config.
        Checks that the required API key is present and the provider matches.
        """
        ...