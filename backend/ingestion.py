import os
import threading

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaIndexDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from db import insert_document, insert_chunks
from core.config import config
from core.logger import get_logger
from core.errors import ParseError, UnsupportedFileTypeError
from core.document import Document, ImageElement
from parsers.router import AutoRouter
from core.cache import get_embedding, set_embedding
from vision.triggers import should_use_vision
from vision.engine import describe_image, describe_pdf_page

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
# Embedding helper
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
    """
    Run the vision engine for a single page. Returns ImageElements with
    descriptions populated.

    For figure-type ImageElements (chunk_type='figure'), vision always runs
    if a vision model is configured — figures are the primary signal.
    For other elements, should_use_vision() gates the call as before.

    Never raises.
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        is_image_file = ext in _IMAGE_EXTENSIONS

        if not page.images:
            # No images at all — check whether we should synthesize one
            if not should_use_vision(file_path, page.text, is_scanned, doc_type, config):
                return []
            description = (
                describe_image(file_path, doc_type)
                if is_image_file
                else describe_pdf_page(file_path, page.page_num - 1, doc_type)
            )
            if description:
                return [ImageElement(
                    page_num=page.page_num,
                    image_ref=f"page_{page.page_num}",
                    ocr_text="",
                    description=description,
                    chunk_type="description",
                    vision_prompt_used=doc_type,
                )]
            return []

        updated_images: list[ImageElement] = []

        for img in page.images:
            # Always describe figures if vision model is configured;
            # for other chunk types, gate on should_use_vision()
            is_figure = img.chunk_type == "figure"

            if img.description:
                updated_images.append(img)
                continue

            if not is_figure and not should_use_vision(
                file_path, page.text, is_scanned, doc_type, config
            ):
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
                chunk_type=img.chunk_type,          # preserves "figure"
                vision_prompt_used=doc_type,
                bbox=img.bbox,                      # B1 — preserve bbox
                caption=img.caption,                # B3 — preserve caption
                element_type=img.element_type,      # B3 — preserve element_type
            ))

        return updated_images

    except Exception as e:
        logger.error("Vision page processing failed", page=page.page_num, error=str(e))
        return page.images


# ---------------------------------------------------------------------------
# Reusable chunk builder (extracted in Group A, extended in Group B)
# ---------------------------------------------------------------------------

def build_chunks_for_document(
    document: Document,
    doc_id: str,
    file_path: str,
    doc_type: str = "general",
    model=None,
) -> tuple[list[dict], bool]:
    """
    Build all chunk rows for a Document.

    Chunk types:
      "text"        — text segments (reading_order if Docling, else raw page text)
      "table"       — one chunk per table
      "description" — vision description for scanned/image pages
      "figure"      — B3: figure/chart element with caption + bbox in metadata

    Returns:
        (chunk_rows, vision_used)
    """
    if model is None:
        model = get_embed_model()

    file_name = document.file_name
    is_scanned = document.is_scanned
    chunk_rows = []
    vision_used = False

    for page in document.pages:

        # ----------------------------------------------------------------
        # 1. Text chunks — reading_order (Docling) or raw page text
        # ----------------------------------------------------------------
        if page.reading_order:
            for segment_idx, segment in enumerate(page.reading_order):
                if len(segment.strip()) < 20:
                    continue

                doc_obj = LlamaIndexDocument(text=segment)
                nodes = splitter.get_nodes_from_documents([doc_obj])

                # B1 — attach bbox if available for this segment
                segment_bbox = None
                if (page.reading_order_bboxes and
                        segment_idx < len(page.reading_order_bboxes)):
                    b = page.reading_order_bboxes[segment_idx]
                    segment_bbox = b.to_dict() if b else None

                for node in nodes:
                    clean_text = node.text.replace("\x00", " ").strip()
                    if not clean_text:
                        continue
                    embedding = _get_embedding(clean_text, model)
                    chunk_rows.append({
                        "document_id": doc_id,
                        "content":     clean_text,
                        "embedding":   embedding,
                        "metadata": {
                            "page":                str(page.page_num),
                            "file":                file_name,
                            "chunk_type":          "text",
                            "image_ref":           None,
                            "reading_order_index": segment_idx,
                            "bbox":                segment_bbox,   # B1
                        }
                    })
        else:
            llama_doc = LlamaIndexDocument(text=page.text)
            nodes = splitter.get_nodes_from_documents([llama_doc])

            for node in nodes:
                clean_text = node.text.replace("\x00", " ").strip()
                if not clean_text:
                    continue
                embedding = _get_embedding(clean_text, model)
                chunk_rows.append({
                    "document_id": doc_id,
                    "content":     clean_text,
                    "embedding":   embedding,
                    "metadata": {
                        "page":                str(page.page_num),
                        "file":                file_name,
                        "chunk_type":          "text",
                        "image_ref":           None,
                        "reading_order_index": None,
                        "bbox":                None,
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
                "content":     clean_table,
                "embedding":   embedding,
                "metadata": {
                    "page":       str(page.page_num),
                    "file":       file_name,
                    "chunk_type": "table",
                    "image_ref":  None,
                    "bbox":       None,
                }
            })

        # ----------------------------------------------------------------
        # 3. Vision + figure chunks (B3)
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

            # B3 — figure chunks get their caption prepended if present
            if img.chunk_type == "figure" and img.caption:
                content = (
                    f"[Figure — Page {page.page_num}]: "
                    f"Caption: {img.caption}. {clean_desc}"
                )
            elif img.chunk_type == "figure":
                content = f"[Figure — Page {page.page_num}]: {clean_desc}"
            else:
                content = f"[Visual Description — Page {page.page_num}]: {clean_desc}"

            embedding = _get_embedding(content, model)
            chunk_rows.append({
                "document_id": doc_id,
                "content":     content,
                "embedding":   embedding,
                "metadata": {
                    "page":               str(page.page_num),
                    "file":               file_name,
                    "chunk_type":         img.chunk_type,   # "figure" or "description"
                    "image_ref":          img.image_ref,
                    "vision_prompt_used": img.vision_prompt_used,
                    "caption":            img.caption,      # B3
                    "element_type":       img.element_type, # B3
                    "bbox":               img.bbox.to_dict() if img.bbox else None,  # B1
                }
            })
            vision_used = True

    return chunk_rows, vision_used


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str,
    use_llamaparse: bool = True,
    doc_type: str = "general",
    user_id: str = "anonymous",
) -> dict:
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    model = get_embed_model()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file type", file=file_name, ext=ext)
        return {"error": f"Unsupported file type: {ext}"}

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
        if os.path.splitext(file_path)[1].lower() not in _IMAGE_EXTENSIONS:
            return {"error": "Could not extract text. File may be empty or image-only."}

    doc_id = insert_document(file_name, user_id=user_id)

    chunk_rows, vision_used = build_chunks_for_document(
        document=document,
        doc_id=doc_id,
        file_path=file_path,
        doc_type=doc_type,
        model=model,
    )

    if not chunk_rows:
        return {"error": "No content could be extracted for indexing."}

    insert_chunks(chunk_rows)

    text_chunks   = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "text")
    table_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "table")
    desc_chunks   = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "description")
    figure_chunks = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "figure")

    logger.info(
        "Ingestion complete",
        file=file_name,
        doc_id=doc_id,
        chunks=len(chunk_rows),
        text_chunks=text_chunks,
        table_chunks=table_chunks,
        desc_chunks=desc_chunks,
        figure_chunks=figure_chunks,
        parser=document.parser_used,
        vision_used=vision_used,
        doc_type=doc_type,
    )

    return {
        "document_id":      doc_id,
        "file":             file_name,
        "chunks_stored":    len(chunk_rows),
        "text_chunks":      text_chunks,
        "table_chunks":     table_chunks,
        "description_chunks": desc_chunks,
        "figure_chunks":    figure_chunks,
        "parser":           document.parser_used,
        "page_count":       document.page_count,
        "table_count":      len(document.tables),
        "vision_used":      vision_used,
    }


# ---------------------------------------------------------------------------
# URL ingestion — unchanged
# ---------------------------------------------------------------------------

def ingest_url(url: str, user_id: str = "anonymous") -> dict:
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

    doc_id = insert_document(document.file_name, user_id=user_id)
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
                "content":     clean,
                "embedding":   embedding,
                "metadata": {
                    "page":       str(page.page_num),
                    "file":       document.file_name,
                    "chunk_type": "text",
                    "image_ref":  None,
                    "source_url": url,
                }
            })

    if not chunk_rows:
        return {"error": "No content could be extracted from this URL."}

    insert_chunks(chunk_rows)
    logger.info("URL ingestion complete", url=url, doc_id=doc_id, chunks=len(chunk_rows))

    return {
        "document_id":  doc_id,
        "file":         document.file_name,
        "title":        document.metadata.get("title", ""),
        "chunks_stored": len(chunk_rows),
        "page_count":   document.page_count,
        "parser":       "url",
    }