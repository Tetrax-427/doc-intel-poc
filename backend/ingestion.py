import os
import threading

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaIndexDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from db import insert_document, insert_chunks
from core.config import config, uses_hierarchical_chunking
from core.logger import get_logger
from core.errors import ParseError, UnsupportedFileTypeError
from core.document import ImageElement
from parsers.router import AutoRouter
from core.cache import get_embedding, set_embedding
from vision.triggers import should_use_vision
from vision.engine import describe_image, describe_pdf_page
from hierarchical import build_hierarchical_chunks, make_flat_chunk_metadata

load_dotenv()

logger = get_logger("ingestion")

embed_model = None
_model_lock = threading.Lock()

SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".csv", ".xlsx",
    ".rtf", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tiff"
]

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

_router = AutoRouter(config)


# ---------------------------------------------------------------------------
# Embedding helper — cache-aware
# ---------------------------------------------------------------------------

def _get_embedding(text: str, model) -> list[float]:
    try:
        cached = get_embedding(text)
        if cached is not None:
            return cached
        embedding = model.get_text_embedding(text)
        set_embedding(text, embedding)
        return embedding
    except ImportError:
        return model.get_text_embedding(text)
    except Exception as e:
        logger.warning("Cache lookup failed — calling model directly", error=str(e))
        return model.get_text_embedding(text)


# ---------------------------------------------------------------------------
# Vision helper
# ---------------------------------------------------------------------------

def _run_vision_for_page(
    file_path: str,
    page,
    doc_type: str,
    is_scanned: bool,
) -> list[ImageElement]:
    try:
        if not should_use_vision(file_path, page.text, is_scanned, doc_type, config):
            return page.images

        ext          = os.path.splitext(file_path)[1].lower()
        is_image_file = ext in _IMAGE_EXTENSIONS
        updated_images: list[ImageElement] = []

        if page.images:
            for img in page.images:
                if img.description:
                    updated_images.append(img)
                    continue
                description = (
                    describe_image(file_path, doc_type)
                    if is_image_file
                    else describe_pdf_page(file_path, page.page_num - 1, doc_type)
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
            description = (
                describe_image(file_path, doc_type)
                if is_image_file
                else describe_pdf_page(file_path, page.page_num - 1, doc_type)
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
            page=page.page_num, error=str(e),
        )
        return page.images


# ---------------------------------------------------------------------------
# Main file ingestion —
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    use_llamaparse: bool = True,
    doc_type: str = "general",
    user_id: str = "anonymous",
) -> dict:
    """
    Ingest a file into the vector store.

    D1 change: classifies document from raw text BEFORE chunking so the
    correct chunking mode (hierarchical vs flat) can be selected per page.
    The classification result is returned in the response dict so
    documents.py can persist it without running classification a second time.

    Args:
        file_path:      Absolute path to the uploaded file.
        use_llamaparse: Use LlamaParse for PDF parsing (falls back to pypdf).
        doc_type:       Hint from the upload form — used only as a fallback
                        if our internal classification fails or returns "general".
        user_id:        Scopes the document to this user.

    Returns:
        Dict with document_id, chunk counts, parser used, vision_used flag,
        and classification result. Returns {"error": "..."} on failure.
    """
    file_name = os.path.basename(file_path)
    ext       = os.path.splitext(file_path)[1].lower()
    model     = get_embed_model()

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
        return {"error": str(e), "code": e.code}
    except ParseError as e:
        return {"error": str(e), "code": e.code}
    except Exception as e:
        return {"error": f"Could not process file: {str(e)}"}

    if not document.full_text.strip():
        if ext not in _IMAGE_EXTENSIONS:
            return {"error": "Could not extract text. File may be empty or image-only."}

    # ----------------------------------------------------------------
    # Classify from raw text BEFORE chunking
    # This determines whether to use hierarchical or flat chunking.
    # ----------------------------------------------------------------
    detected_doc_type = doc_type   # start with caller's hint
    classification_result = None

    try:
        from retrieval import classify_document_from_text
        raw_text_sample = document.full_text[:3000]
        classification_result = classify_document_from_text(
            raw_text_sample, document_id=""
        )
        detected = classification_result.get("doc_type", "general")
        if detected and detected != "general":
            detected_doc_type = detected
            logger.info(
                "Pre-ingestion classification complete",
                file=file_name,
                doc_type=detected_doc_type,
                confidence=classification_result.get("confidence", 0.0),
            )
        else:
            logger.info(
                "Pre-ingestion classification returned general — keeping caller hint",
                file=file_name,
                hint=doc_type,
            )
    except Exception as exc:
        logger.warning(
            "Pre-ingestion classification failed — using caller hint",
            file=file_name,
            error=str(exc),
            fallback_doc_type=doc_type,
        )

    use_hierarchical = uses_hierarchical_chunking(detected_doc_type)
    logger.info(
        "Chunking mode selected",
        file=file_name,
        doc_type=detected_doc_type,
        mode="hierarchical" if use_hierarchical else "flat",
    )

    is_scanned = document.is_scanned
    doc_id     = insert_document(file_name, user_id=user_id)
    chunk_rows = []
    vision_used = False

    def _embed(text: str) -> list[float]:
        return _get_embedding(text, model)

    for page in document.pages:

        # ----------------------------------------------------------------
        # 1. Text chunks — hierarchical or flat depending on doc type
        # ----------------------------------------------------------------
        if use_hierarchical:
            hier_rows = build_hierarchical_chunks(
                page_text=page.text,
                page_num=page.page_num,
                file_name=file_name,
                document_id=doc_id,
                embed_fn=_embed,
            )
            chunk_rows.extend(hier_rows)
        else:
            # Flat chunking — unchanged from original ingestion.py
            llama_doc = LlamaIndexDocument(text=page.text)
            nodes     = splitter.get_nodes_from_documents([llama_doc])

            for node in nodes:
                clean_text = node.text.replace("\x00", " ").strip()
                if not clean_text:
                    continue
                embedding = _embed(clean_text)
                chunk_rows.append({
                    "document_id": doc_id,
                    "content":     clean_text,
                    "embedding":   embedding,
                    "metadata":    make_flat_chunk_metadata(
                        page.page_num, file_name, chunk_type="text"
                    ),
                })

        # ----------------------------------------------------------------
        # 2. Table chunks — always flat regardless of chunking mode
        # ----------------------------------------------------------------
        for table in page.tables:
            if not table.raw_text.strip():
                continue
            table_text  = f"[Table: {table.title or 'untitled'}]\n{table.raw_text}"
            clean_table = table_text.replace("\x00", " ").strip()
            embedding   = _embed(clean_table)
            chunk_rows.append({
                "document_id": doc_id,
                "content":     clean_table,
                "embedding":   embedding,
                "metadata":    make_flat_chunk_metadata(
                    page.page_num, file_name, chunk_type="table"
                ),
            })

        # ----------------------------------------------------------------
        # 3. Vision description chunks — always flat
        # ----------------------------------------------------------------
        images_with_descriptions = _run_vision_for_page(
            file_path, page, detected_doc_type, is_scanned
        )

        for img in images_with_descriptions:
            if not img.description:
                continue
            clean_desc = img.description.replace("\x00", " ").strip()
            if not clean_desc:
                continue

            content   = f"[Visual Description — Page {page.page_num}]: {clean_desc}"
            embedding = _embed(content)
            chunk_rows.append({
                "document_id": doc_id,
                "content":     content,
                "embedding":   embedding,
                "metadata":    make_flat_chunk_metadata(
                    page.page_num, file_name,
                    chunk_type="description",
                    image_ref=img.image_ref,
                    extra={"vision_prompt_used": img.vision_prompt_used},
                ),
            })
            vision_used = True

    if not chunk_rows:
        return {"error": "No content could be extracted for indexing."}

    insert_chunks(chunk_rows)

    text_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "text"
                       and c["metadata"].get("chunk_level") != "parent")
    table_chunks = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "table")
    desc_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "description")
    parent_chunks = sum(1 for c in chunk_rows if c["metadata"].get("chunk_level") == "parent")

    logger.info(
        "Ingestion complete",
        file=file_name,
        doc_id=doc_id,
        chunks=len(chunk_rows),
        text_chunks=text_chunks,
        table_chunks=table_chunks,
        desc_chunks=desc_chunks,
        parent_chunks=parent_chunks,
        parser=document.parser_used,
        vision_used=vision_used,
        doc_type=detected_doc_type,
        chunking_mode="hierarchical" if use_hierarchical else "flat",
    )

    return {
        "document_id":      doc_id,
        "file":             file_name,
        "chunks_stored":    len(chunk_rows),
        "text_chunks":      text_chunks,
        "table_chunks":     table_chunks,
        "description_chunks": desc_chunks,
        "parent_chunks":    parent_chunks,
        "parser":           document.parser_used,
        "page_count":       document.page_count,
        "table_count":      len(document.tables),
        "vision_used":      vision_used,
        "doc_type":         detected_doc_type,
        "chunking_mode":    "hierarchical" if use_hierarchical else "flat",
        # Pass classification result so documents.py can persist it
        # without running a second classify_document() call.
        "_classification":  classification_result,
    }


# ---------------------------------------------------------------------------
# URL ingestion — flat chunking only (URLs are always flat)
# ---------------------------------------------------------------------------

def ingest_url(url: str, user_id: str = "anonymous") -> dict:
    model = get_embed_model()
    logger.info("Starting URL ingestion", url=url)

    try:
        document = _router.parse(url)
    except ParseError as e:
        return {"error": str(e), "code": e.code}
    except Exception as e:
        return {"error": f"Could not fetch URL: {e}"}

    if not document.full_text.strip():
        return {"error": "Could not extract text from this URL."}

    doc_id     = insert_document(document.file_name, user_id=user_id)
    chunk_rows = []

    def _embed(text: str) -> list[float]:
        return _get_embedding(text, model)

    for page in document.pages:
        llama_doc = LlamaIndexDocument(text=page.text)
        nodes     = splitter.get_nodes_from_documents([llama_doc])

        for node in nodes:
            clean = node.text.replace("\x00", " ").strip()
            if not clean:
                continue
            chunk_rows.append({
                "document_id": doc_id,
                "content":     clean,
                "embedding":   _embed(clean),
                "metadata":    make_flat_chunk_metadata(
                    page.page_num, document.file_name,
                    chunk_type="text",
                    extra={"source_url": url},
                ),
            })

    if not chunk_rows:
        return {"error": "No content could be extracted from this URL."}

    insert_chunks(chunk_rows)

    logger.info("URL ingestion complete", url=url, doc_id=doc_id, chunks=len(chunk_rows))

    return {
        "document_id":   doc_id,
        "file":          document.file_name,
        "title":         document.metadata.get("title", ""),
        "chunks_stored": len(chunk_rows),
        "page_count":    document.page_count,
        "parser":        "url",
    }