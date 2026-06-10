import os
import base64
import time

from vision.base import BaseVisionModel
from core.config import Config
from core.logger import get_logger
from openai import OpenAI
from core.config import config
from llm.usage import log_usage

logger = get_logger("vision.openai")

# MIME types supported by the OpenAI vision API
_MIME_MAP = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
}


class OpenAIVisionModel(BaseVisionModel):
    """
    GPT-4o Vision implementation.

    Requires:
        VISION_PROVIDER=openai
        OPENAI_API_KEY=<key>
        VISION_MODEL=gpt-4o   (or any other OpenAI vision-capable model)
    """

    def get_name(self) -> str:
        return "openai"

    def is_available(self, config: Config) -> bool:
        return bool(config.openai_api_key) and config.vision_provider == "openai"

    def describe(self, image_path: str, prompt: str) -> str:
        try:
            
            if not os.path.exists(image_path):
                logger.warning("Vision image file not found", path=image_path)
                return ""

            ext = os.path.splitext(image_path)[1].lower()
            mime_type = _MIME_MAP.get(ext, "image/jpeg")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            client = OpenAI(api_key=config.openai_api_key)

            start = time.time()
            response = client.chat.completions.create(
                model=config.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}"
                            }
                        }
                    ]
                }],
                max_tokens=500
            )
            duration_ms = int((time.time() - start) * 1000)
            description = response.choices[0].message.content or ""

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
                provider="openai",
                model=config.vision_model,
                duration_ms=duration_ms,
                chars=len(description)
            )
            return description

        except Exception as e:
            logger.error("Vision call failed", provider="openai", error=str(e))
            return ""  # NEVER raise — vision is always optional