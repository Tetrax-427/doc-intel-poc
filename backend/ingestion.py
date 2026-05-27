import os
import tempfile
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from db import insert_document, insert_chunks

load_dotenv()

embed_model = None

def get_embed_model():
    global embed_model
    if embed_model is None:
        print("Loading embedding model... (first time only)")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return embed_model

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

def parse_with_llamaparse(file_path: str) -> list[dict]:
    """Use LlamaParse for complex PDFs — handles tables, columns, scanned docs"""
    from llama_parse import LlamaParse

    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",        # returns clean markdown with tables preserved
        verbose=False
    )

    docs = parser.load_data(file_path)

    pages = []
    for i, doc in enumerate(docs):
        if doc.text and doc.text.strip():
            pages.append({
                "text": doc.text.strip(),
                "page": str(i + 1)
            })
    return pages

def parse_with_pypdf(file_path: str) -> list[dict]:
    """Fallback parser for simple PDFs"""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": str(i + 1)
            })
    return pages

def ingest_file(file_path: str, use_llamaparse: bool = True) -> dict:
    file_name = os.path.basename(file_path)
    model = get_embed_model()

    # Try LlamaParse first, fall back to pypdf
    pages = []
    if use_llamaparse and os.getenv("LLAMA_CLOUD_API_KEY"):
        try:
            print(f"Parsing with LlamaParse: {file_name}")
            pages = parse_with_llamaparse(file_path)
            print(f"LlamaParse extracted {len(pages)} pages")
        except Exception as e:
            print(f"LlamaParse failed: {e} — falling back to pypdf")

    if not pages:
        print(f"Parsing with pypdf: {file_name}")
        pages = parse_with_pypdf(file_path)

    if not pages:
        return {"error": "Could not extract text. PDF may be image-only with no OCR."}

    doc_id = insert_document(file_name)

    chunk_rows = []
    for page_data in pages:
        doc = Document(text=page_data["text"])
        nodes = splitter.get_nodes_from_documents([doc])

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
                    "page": page_data["page"],
                    "file": file_name
                }
            })

    insert_chunks(chunk_rows)

    return {
        "document_id": doc_id,
        "file": file_name,
        "chunks_stored": len(chunk_rows),
        "parser": "llamaparse" if pages else "pypdf"
    }