# backend/ingestion.py

import os
import threading

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaIndexDocument

from db import insert_document, insert_chunks
from core.config import config
from core.logger import get_logger
from core.errors import ParseError, UnsupportedFileTypeError
from core.document import ImageElement
from core.cache import get_embedding, set_embedding
from vision.triggers import should_use_vision
from vision.engine import describe_image, describe_pdf_page
from parsers.router import AutoRouter

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
             
load_dotenv()

logger = get_logger("ingestion")

embed_model = None
_model_lock = threading.Lock()

SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".csv", ".xlsx",
    ".rtf", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tiff"
]

# Image file extensions — these trigger whole-file vision (not page rendering)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff"}


def get_embed_model():
    global embed_model
    if embed_model is None:
        with _model_lock:
            if embed_model is None:
                logger.info("Loading embedding model (first time only)")
                embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    device="cpu"
                )
    return embed_model


splitter = SentenceSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
)

# Single router instance — reused for all requests
_router = AutoRouter(config)


# ---------------------------------------------------------------------------
# Embedding helper — cache-aware
# ---------------------------------------------------------------------------

def _get_embedding(text: str, model) -> list[float]:
    """
    Return embedding for text, checking cache first.
    Falls back to direct model call if cache is unavailable.
    """
    try:
        cached = get_embedding(text)
        if cached is not None:
            return cached
        embedding = model.get_text_embedding(text)
        set_embedding(text, embedding)
        return embedding
    except ImportError:
        # cache.py not yet available — degrade gracefully
        return model.get_text_embedding(text)
    except Exception as e:
        logger.warning("Cache lookup failed — calling model directly", error=str(e))
        return model.get_text_embedding(text)


# ---------------------------------------------------------------------------
# Vision helper — called per page during ingestion
# ---------------------------------------------------------------------------

def _run_vision_for_page(
    file_path: str,
    page,
    doc_type: str,
    is_scanned: bool,
) -> list[ImageElement]:
    """
    Run the vision engine for a single page and return a list of ImageElements
    with descriptions populated.

    Two cases:
      A) Page already has ImageElements (from parser) — describe each one.
      B) Page has no images but vision should still run (scanned PDF / low text) —
         synthesize one ImageElement by rendering the whole page.

    Returns the same list that was on page.images if no vision ran.
    Never raises.
    """
    try:

        if not should_use_vision(file_path, page.text, is_scanned, doc_type, config):
            return page.images  # vision not needed — return existing (empty) list

        ext = os.path.splitext(file_path)[1].lower()
        is_image_file = ext in _IMAGE_EXTENSIONS
        updated_images: list[ImageElement] = []

        if page.images:
            # Case A — parser found embedded images; describe each
            for img in page.images:
                if img.description:
                    # Already has a description.
                    updated_images.append(img)
                    continue

                if is_image_file:
                    description = describe_image(file_path, doc_type)
                else:
                    description = describe_pdf_page(
                        file_path, page.page_num - 1, doc_type
                    )

                updated_images.append(ImageElement(
                    page_num=img.page_num,
                    image_ref=img.image_ref or f"page_{page.page_num}",
                    ocr_text=img.ocr_text,
                    description=description,
                    chunk_type="description" if description else img.chunk_type,
                    vision_prompt_used=doc_type,
                ))

        else:
            # Case B — no embedded images, but vision triggered (scanned / low word count)
            # Synthesize one ImageElement by rendering the whole page
            if is_image_file:
                description = describe_image(file_path, doc_type)
            else:
                description = describe_pdf_page(
                    file_path, page.page_num - 1, doc_type
                )

            if description:
                updated_images.append(ImageElement(
                    page_num=page.page_num,
                    image_ref=f"page_{page.page_num}",
                    ocr_text="",
                    description=description,
                    chunk_type="description",
                    vision_prompt_used=doc_type,
                ))

        return updated_images

    except Exception as e:
        logger.error(
            "Vision page processing failed — skipping vision for this page",
            page=page.page_num,
            error=str(e)
        )
        return page.images  # safe fallback — never block ingestion


# ---------------------------------------------------------------------------
# Main file ingestion
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    use_llamaparse: bool = True,
    doc_type: str = "general",
) -> dict:
    """
    Ingest a file into the vector store.

    Args:
        file_path:      Absolute path to the uploaded file.
        use_llamaparse: Use LlamaParse for PDF parsing (falls back to pypdf).
        doc_type:       Pre-classified document type — used to select the right
                        vision prompt. Pass "general" if classification hasn't
                        run yet.

    Returns:
        Dict with document_id, chunk counts, parser used, vision_used flag.
        Returns {"error": "..."} on failure.
    """
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    model = get_embed_model()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file type", file=file_name, ext=ext)
        return {"error": f"Unsupported file type: {ext}"}

    # --- Parse ---
    try:
        logger.info("Routing file to parser", file=file_name, doc_type=doc_type)
        document = _router.parse(file_path)
        logger.info(
            "Parse complete",
            file=file_name,
            parser=document.parser_used,
            pages=document.page_count,
            tables=len(document.tables),
            is_scanned=document.is_scanned,
        )
    except UnsupportedFileTypeError as e:
        logger.error("No parser available", file=file_name, error=str(e))
        return {"error": str(e), "code": e.code}
    except ParseError as e:
        logger.error("Parse failed", file=file_name, error=str(e),
                     code=e.code, retryable=e.retryable)
        return {"error": str(e), "code": e.code}
    except Exception as e:
        logger.error("Unexpected parse error", file=file_name, error=str(e))
        return {"error": f"Could not process file: {str(e)}"}

    if not document.full_text.strip():
        # Image-only files may have no text — still allow vision to proceed
        ext_lower = os.path.splitext(file_path)[1].lower()
        if ext_lower not in _IMAGE_EXTENSIONS:
            return {"error": "Could not extract text. File may be empty or image-only."}

    is_scanned = document.is_scanned
    doc_id = insert_document(file_name)
    chunk_rows = []
    vision_used = False

    for page in document.pages:

        # ----------------------------------------------------------------
        # 1. Text chunks
        # ----------------------------------------------------------------
        llama_doc = LlamaIndexDocument(text=page.text)
        nodes = splitter.get_nodes_from_documents([llama_doc])

        for node in nodes:
            clean_text = node.text.replace("\x00", " ").strip()
            if not clean_text:
                continue
            embedding = _get_embedding(clean_text, model)
            chunk_rows.append({
                "document_id": doc_id,
                "content": clean_text,
                "embedding": embedding,
                "metadata": {
                    "page": str(page.page_num),
                    "file": file_name,
                    "chunk_type": "text",
                    "image_ref": None,
                }
            })

        # ----------------------------------------------------------------
        # 2. Table chunks
        # ----------------------------------------------------------------
        for table in page.tables:
            if not table.raw_text.strip():
                continue
            table_text = f"[Table: {table.title or 'untitled'}]\n{table.raw_text}"
            clean_table = table_text.replace("\x00", " ").strip()
            embedding = _get_embedding(clean_table, model)
            chunk_rows.append({
                "document_id": doc_id,
                "content": clean_table,
                "embedding": embedding,
                "metadata": {
                    "page": str(page.page_num),
                    "file": file_name,
                    "chunk_type": "table",
                    "image_ref": None,
                }
            })

        # ----------------------------------------------------------------
        # 3. Vision description chunks
        #    Run vision live here — _run_vision_for_page() calls the
        #    vision engine and returns ImageElements with descriptions.
        # ----------------------------------------------------------------
        images_with_descriptions = _run_vision_for_page(
            file_path, page, doc_type, is_scanned
        )

        for img in images_with_descriptions:
            if not img.description:
                continue
            clean_desc = img.description.replace("\x00", " ").strip()
            if not clean_desc:
                continue

            content = f"[Visual Description — Page {page.page_num}]: {clean_desc}"
            embedding = _get_embedding(content, model)
            chunk_rows.append({
                "document_id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": {
                    "page": str(page.page_num),
                    "file": file_name,
                    "chunk_type": "description",       
                    "image_ref": img.image_ref,
                    "vision_prompt_used": img.vision_prompt_used,
                }
            })
            vision_used = True

    if not chunk_rows:
        return {"error": "No content could be extracted for indexing."}

    insert_chunks(chunk_rows)

    text_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "text")
    table_chunks = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "table")
    desc_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "description")

    logger.info(
        "Ingestion complete",
        file=file_name,
        doc_id=doc_id,
        chunks=len(chunk_rows),
        text_chunks=text_chunks,
        table_chunks=table_chunks,
        desc_chunks=desc_chunks,
        parser=document.parser_used,
        vision_used=vision_used,
        doc_type=doc_type,
    )

    return {
        "document_id": doc_id,
        "file": file_name,
        "chunks_stored": len(chunk_rows),
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "description_chunks": desc_chunks,
        "parser": document.parser_used,
        "page_count": document.page_count,
        "table_count": len(document.tables),
        "vision_used": vision_used,
    }


# ---------------------------------------------------------------------------
# URL ingestion —
# ---------------------------------------------------------------------------

def ingest_url(url: str) -> dict:
    model = get_embed_model()

    logger.info("Starting URL ingestion", url=url)

    try:
        document = _router.parse(url)
    except ParseError as e:
        logger.error("URL parse failed", url=url, error=str(e))
        return {"error": str(e), "code": e.code}
    except Exception as e:
        logger.error("Unexpected URL parse error", url=url, error=str(e))
        return {"error": f"Could not fetch URL: {e}"}

    if not document.full_text.strip():
        return {"error": "Could not extract text from this URL."}

    doc_id = insert_document(document.file_name)
    chunk_rows = []

    for page in document.pages:
        llama_doc = LlamaIndexDocument(text=page.text)
        nodes = splitter.get_nodes_from_documents([llama_doc])

        for node in nodes:
            clean = node.text.replace("\x00", " ").strip()
            if not clean:
                continue
            embedding = _get_embedding(clean, model)
            chunk_rows.append({
                "document_id": doc_id,
                "content": clean,
                "embedding": embedding,
                "metadata": {
                    "page": str(page.page_num),
                    "file": document.file_name,
                    "chunk_type": "text",
                    "image_ref": None,
                    "source_url": url,
                }
            })

    if not chunk_rows:
        return {"error": "No content could be extracted from this URL."}

    insert_chunks(chunk_rows)

    logger.info(
        "URL ingestion complete",
        url=url,
        doc_id=doc_id,
        chunks=len(chunk_rows),
    )

    return {
        "document_id": doc_id,
        "file": document.file_name,
        "title": document.metadata.get("title", ""),
        "chunks_stored": len(chunk_rows),
        "page_count": document.page_count,
        "parser": "url",
    }