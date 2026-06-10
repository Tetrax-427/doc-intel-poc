import os
import base64
import time

from vision.base import BaseVisionModel
from core.config import Config
from core.logger import get_logger
import anthropic
from llm.usage import log_usage     
from core.config import config

logger = get_logger("vision.anthropic")

# MIME types supported by the Anthropic vision API
_MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}


class AnthropicVisionModel(BaseVisionModel):
    """
    Claude Vision implementation (claude-3-5-sonnet, claude-opus-4 etc.)

    Requires:
        VISION_PROVIDER=anthropic
        ANTHROPIC_API_KEY=<key>
        VISION_MODEL=claude-3-5-sonnet-20241022  (or any claude model with vision)
    """

    def get_name(self) -> str:
        return "anthropic"

    def is_available(self, config: Config) -> bool:
        return bool(config.anthropic_api_key) and config.vision_provider == "anthropic"

    def describe(self, image_path: str, prompt: str) -> str:
        try:
            
            if not os.path.exists(image_path):
                logger.warning("Vision image file not found", path=image_path)
                return ""

            ext = os.path.splitext(image_path)[1].lower()
            mime_type = _MIME_MAP.get(ext, "image/jpeg")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            client = anthropic.Anthropic(api_key=config.anthropic_api_key)

            start = time.time()
            response = client.messages.create(
                model=config.vision_model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }]
            )
            duration_ms = int((time.time() - start) * 1000)
            description = response.content[0].text if response.content else ""

            # Log token usage if usage tracking is available
            try:
                log_usage(
                    "vision", config.vision_model,
                    len(prompt), len(description),
                    duration_ms / 1000
                )
            except Exception:
                pass  # usage logging is non-critical

            logger.info(
                "Vision description generated",
                provider="anthropic",
                model=config.vision_model,
                duration_ms=duration_ms,
                chars=len(description)
            )
            return description

        except Exception as e:
            logger.error("Vision call failed", provider="anthropic", error=str(e))
            return ""  # NEVER raise — vision is always optional