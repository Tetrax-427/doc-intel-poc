import os
import threading
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document as LlamaIndexDocument
from db import insert_document, insert_chunks
from core.config import config
from core.logger import get_logger
from core.errors import ParseError, UnsupportedFileTypeError
from parsers.router import AutoRouter

load_dotenv()

logger = get_logger("ingestion")

embed_model = None
_model_lock = threading.Lock()

SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".csv", ".xlsx",
    ".rtf", ".md", ".png", ".jpg", ".jpeg", ".webp", ".tiff"
]


def get_embed_model():
    global embed_model
    if embed_model is None:
        with _model_lock:
            if embed_model is None:
                logger.info("Loading embedding model (first time only)")
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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


# --- Main file ingestion ---

def ingest_file(file_path: str, use_llamaparse: bool = True, doc_type: str = "general") -> dict:
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    model = get_embed_model()

    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported file type", file=file_name, ext=ext)
        return {"error": f"Unsupported file type: {ext}"}

    # Parse file via AutoRouter — returns a structured Document
    try:
        logger.info("Routing file to parser", file=file_name)
        document = _router.parse(file_path)
        logger.info("Parse complete",
                    file=file_name,
                    parser=document.parser_used,
                    pages=document.page_count,
                    tables=len(document.tables))
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
        return {"error": "Could not extract text. File may be empty or image-only."}

    doc_id = insert_document(file_name)
    chunk_rows = []

    for page in document.pages:

        # 1. Text chunks
        llama_doc = LlamaIndexDocument(text=page.text)
        nodes = splitter.get_nodes_from_documents([llama_doc])

        for node in nodes:
            clean_text = node.text.replace("\x00", " ").strip()
            if not clean_text:
                continue
            embedding = model.get_text_embedding(clean_text)
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

        # 2. Table chunks — stored separately for precise retrieval
        for table in page.tables:
            if not table.raw_text.strip():
                continue
            table_text = f"[Table: {table.title or 'untitled'}]\n{table.raw_text}"
            clean_table = table_text.replace("\x00", " ").strip()
            embedding = model.get_text_embedding(clean_table)
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

        # 3. Vision description chunks
        for img in page.images:
            if not img.description:
                continue
            clean_desc = img.description.replace("\x00", " ").strip()
            if not clean_desc:
                continue
            content = f"[Visual Description — Page {page.page_num}]: {clean_desc}"
            embedding = model.get_text_embedding(content)
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

    if not chunk_rows:
        return {"error": "No content could be extracted for indexing."}

    insert_chunks(chunk_rows)

    text_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "text")
    table_chunks = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "table")
    desc_chunks  = sum(1 for c in chunk_rows if c["metadata"]["chunk_type"] == "description")

    logger.info("Ingestion complete",
                file=file_name,
                doc_id=doc_id,
                chunks=len(chunk_rows),
                text_chunks=text_chunks,
                table_chunks=table_chunks,
                desc_chunks=desc_chunks,
                parser=document.parser_used,
                vision_used=document.vision_used)

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
        "vision_used": document.vision_used,
    }


# --- URL ingestion ---

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
            embedding = model.get_text_embedding(clean)
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

    logger.info("URL ingestion complete",
                url=url,
                doc_id=doc_id,
                chunks=len(chunk_rows))

    return {
        "document_id": doc_id,
        "file": document.file_name,
        "title": document.metadata.get("title", ""),
        "chunks_stored": len(chunk_rows),
        "page_count": document.page_count,
        "parser": "url",
    }