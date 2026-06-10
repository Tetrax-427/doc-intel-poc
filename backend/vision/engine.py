import os
import tempfile
import fitz  
from core.config import Config
from core.logger import get_logger
from vision.base import BaseVisionModel
from vision.null import NoVisionModel
from core.cache import get_vision_description, set_vision_description
from core.config import config as _config
from vision.openai import OpenAIVisionModel
from vision.anthropic import AnthropicVisionModel
from core.config import config
from schemas.templates import get_vision_prompt
 
logger = get_logger("vision.engine")

# Module-level singleton — initialised once, reused for all requests.
# Reset to None in tests to allow different config scenarios.
_vision_model: BaseVisionModel | None = None


def get_vision_model(config: Config = None) -> BaseVisionModel:
    """
    Return the configured vision model, initialising it on first call.

    Selection logic:
      1. If VISION_PROVIDER is not set → NoVisionModel
      2. If provider is 'openai' → OpenAIVisionModel (if API key present)
      3. If provider is 'anthropic' → AnthropicVisionModel (if API key present)
      4. If API key missing or provider unknown → NoVisionModel (with warning)

    The singleton is safe to call from multiple threads — worst case is
    double-initialisation on first call, both paths produce the same result.
    """
    global _vision_model
    if _vision_model is not None:
        return _vision_model

    if config is None:
        config = _config

    if not config.vision_provider:
        logger.info("No VISION_PROVIDER set — vision disabled")
        _vision_model = NoVisionModel()
        return _vision_model

    provider = config.vision_provider.lower()

    if provider == "openai":
        model: BaseVisionModel = OpenAIVisionModel()
    elif provider == "anthropic":
        model = AnthropicVisionModel()
    else:
        logger.warning("Unknown VISION_PROVIDER — falling back to NoVision",
                       provider=config.vision_provider)
        _vision_model = NoVisionModel()
        return _vision_model

    if model.is_available(config):
        _vision_model = model
        logger.info("Vision model initialised",
                    provider=provider,
                    model=config.vision_model)
    else:
        logger.warning(
            "Vision model unavailable (missing API key?) — falling back to NoVision",
            provider=provider
        )
        _vision_model = NoVisionModel()

    return _vision_model


def describe_image(image_path: str, doc_type: str = "general") -> str:
    """
    Describe an image file using the configured vision model.

    Checks the cache first; calls the model on cache miss and writes result back.
    Returns "" if:
      - No vision model is configured
      - The file does not exist
      - The model call fails for any reason

    Never raises.
    """
    try:
        
        # Cache check — images don't change, 7-day TTL
        cached = get_vision_description(image_path, doc_type)
        if cached is not None:
            logger.debug("Vision cache hit", path=os.path.basename(image_path))
            return cached

        
        prompt = get_vision_prompt(doc_type)
        model = get_vision_model(config)
        description = model.describe(image_path, prompt)

        if description:
            set_vision_description(image_path, doc_type, description)

        return description

    except Exception as e:
        logger.error("describe_image failed unexpectedly", error=str(e))
        return ""


def describe_pdf_page(file_path: str, page_num: int, doc_type: str = "general") -> str:
    """
    Render a single PDF page to a temporary PNG and describe it.

    Uses PyMuPDF (fitz) at 150 DPI — sufficient for text/diagram recognition,
    cheap enough not to blow up memory on multi-page PDFs.

    The temporary PNG is deleted immediately after the vision call.
    Returns "" if rendering or the vision call fails.
    Never raises.
    """
    try:
        doc = fitz.open(file_path)

        if page_num >= len(doc):
            logger.warning(
                "Page number out of range",
                file=os.path.basename(file_path),
                page=page_num,
                total_pages=len(doc)
            )
            doc.close()
            return ""

        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        doc.close()

        # Write to a temp file — vision models need a file path
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"docintel_page_{os.getpid()}_{page_num}.png"
        )
        pix.save(temp_path)

        description = describe_image(temp_path, doc_type)

        # Clean up temp file — don't leave PNG files accumulating
        try:
            os.remove(temp_path)
        except Exception:
            pass  # non-critical

        return description

    except ImportError:
        logger.warning("PyMuPDF not installed — cannot render PDF pages for vision")
        return ""
    except Exception as e:
        logger.error(
            "PDF page render failed",
            file=os.path.basename(file_path),
            page=page_num,
            error=str(e)
        )
        return ""


def reset_vision_model():
    """
    Reset the singleton — used in tests to allow re-initialisation
    with different config values without restarting the process.
    """
    global _vision_model
    _vision_model = None