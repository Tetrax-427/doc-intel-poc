import os
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from pypdf import PdfReader
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

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """Extract text page by page using pypdf"""
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

def ingest_file(file_path: str) -> dict:
    file_name = os.path.basename(file_path)
    model = get_embed_model()

    # Extract text per page
    pages = extract_text_from_pdf(file_path)

    if not pages:
        return {"error": "Could not extract text from PDF. May be scanned/image-based."}

    doc_id = insert_document(file_name)

    chunk_rows = []
    for page_data in pages:
        # Split page text into chunks
        from llama_index.core import Document
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
        "chunks_stored": len(chunk_rows)
    }