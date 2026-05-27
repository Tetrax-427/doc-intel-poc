import os
import threading
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from db import insert_document, insert_chunks

load_dotenv()

embed_model = None
_model_lock = threading.Lock()

def get_embed_model():
    global embed_model
    if embed_model is None:
        with _model_lock:
            if embed_model is None:
                print("Loading embedding model... (first time only)")
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embed_model = HuggingFaceEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    device="cpu"
                )
    return embed_model

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".rtf", ".md"]

# --- Parsers ---

def parse_pdf(file_path: str, use_llamaparse: bool = True) -> list[dict]:
    if use_llamaparse and os.getenv("LLAMA_CLOUD_API_KEY"):
        try:
            from llama_parse import LlamaParse
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown",
                verbose=False
            )
            docs = parser.load_data(file_path)
            pages = [{"text": d.text.strip(), "page": str(i+1)} for i, d in enumerate(docs) if d.text.strip()]
            if pages:
                return pages
        except Exception as e:
            print(f"LlamaParse failed: {e} — falling back to pypdf")

    from pypdf import PdfReader
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"text": text.strip(), "page": str(i+1)})
    return pages


def parse_docx(file_path: str) -> list[dict]:
    from docx import Document as DocxDocument
    doc = DocxDocument(file_path)
    chunks = []
    current_text = []
    page_num = 1

    for para in doc.paragraphs:
        if para.text.strip():
            current_text.append(para.text.strip())

    # Also extract tables
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            current_text.append("\n".join(rows))

    # Split into page-sized chunks (~3000 chars each)
    full_text = "\n\n".join(current_text)
    chunk_size = 3000
    for i in range(0, len(full_text), chunk_size):
        chunk = full_text[i:i+chunk_size].strip()
        if chunk:
            chunks.append({"text": chunk, "page": str(page_num)})
            page_num += 1

    return chunks


def parse_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    if not text.strip():
        return []

    # Split into chunks of ~3000 chars
    chunks = []
    chunk_size = 3000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append({"text": chunk, "page": str(i // chunk_size + 1)})
    return chunks


def parse_md(file_path: str) -> list[dict]:
    return parse_txt(file_path)  # markdown is just text


def parse_csv(file_path: str) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(file_path)

    # Convert entire CSV to readable text in batches of 50 rows
    chunks = []
    batch_size = 50
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        text = f"Columns: {', '.join(df.columns.tolist())}\n\n"
        text += batch.to_string(index=False)
        chunks.append({
            "text": text,
            "page": f"rows {i+1}-{min(i+batch_size, len(df))}"
        })
    return chunks


def parse_xlsx(file_path: str) -> list[dict]:
    import pandas as pd
    chunks = []
    xl = pd.ExcelFile(file_path)

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if df.empty:
            continue

        batch_size = 50
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            text = f"Sheet: {sheet_name}\nColumns: {', '.join(df.columns.astype(str).tolist())}\n\n"
            text += batch.to_string(index=False)
            chunks.append({
                "text": text,
                "page": f"{sheet_name} rows {i+1}-{min(i+batch_size, len(df))}"
            })
    return chunks


def parse_rtf(file_path: str) -> list[dict]:
    from striprtf.striprtf import rtf_to_text
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        rtf_content = f.read()
    text = rtf_to_text(rtf_content)
    return parse_txt_from_string(text)


def parse_txt_from_string(text: str) -> list[dict]:
    chunks = []
    chunk_size = 3000
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size].strip()
        if chunk:
            chunks.append({"text": chunk, "page": str(i // chunk_size + 1)})
    return chunks


# --- Router ---

def parse_file(file_path: str, use_llamaparse: bool = True) -> list[dict]:
    ext = os.path.splitext(file_path)[1].lower()
    parsers = {
        ".pdf":  lambda: parse_pdf(file_path, use_llamaparse),
        ".docx": lambda: parse_docx(file_path),
        ".txt":  lambda: parse_txt(file_path),
        ".md":   lambda: parse_md(file_path),
        ".csv":  lambda: parse_csv(file_path),
        ".xlsx": lambda: parse_xlsx(file_path),
        ".rtf":  lambda: parse_rtf(file_path),
    }
    parser = parsers.get(ext)
    if not parser:
        return []
    return parser()


# --- Main ingestion ---

def ingest_file(file_path: str, use_llamaparse: bool = True) -> dict:
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    model = get_embed_model()

    if ext not in SUPPORTED_EXTENSIONS:
        return {"error": f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"}

    print(f"Parsing {file_name} as {ext}")
    pages = parse_file(file_path, use_llamaparse)

    if not pages:
        return {"error": "Could not extract text. File may be empty or image-only."}

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
        "parser": ext.replace(".", "")
    }