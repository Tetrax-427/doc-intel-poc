from vision.base import BaseVisionModel
from core.config import Config


class NoVisionModel(BaseVisionModel):
    """
    Always-available fallback vision model.

    Returns empty string for every describe() call.
    Used when VISION_PROVIDER is not set in .env, or when the
    configured provider's API key is missing.

    Guarantees the system behaves identically whether or not a real
    vision model is configured — callers never need to branch on this.
    """

    def describe(self, image_path: str, prompt: str) -> str:
        return ""

    def get_name(self) -> str:
        return "none"

    def is_available(self, config: Config) -> bool:
        # Always available — it's the safe fallback.
        return True