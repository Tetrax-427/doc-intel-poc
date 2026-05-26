import shutil
import os
from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from ingestion import ingest_file
from retrieval import query_document
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    document_id: str = None


load_dotenv()

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "doc-intel API running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save file temporarily
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Ingest
    result = ingest_file(temp_path)
    return result

class ExtractRequest(BaseModel):
    document_id: str
    schema: dict

@app.post("/extract")
def extract(req: ExtractRequest):
    from retrieval import extract_fields
    return extract_fields(req.document_id, req.schema)

@app.post("/query")
def query(req: QueryRequest):
    return query_document(req.question, req.document_id)

@app.get("/documents")
def list_documents():
    from db import get_all_documents
    return get_all_documents()

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    from db import delete_document_by_id
    delete_document_by_id(document_id)
    return {"status": "deleted"}

@app.get("/chats/{document_id}")
def get_chats(document_id: str):
    from db import get_chat_history
    return get_chat_history(document_id)

@app.post("/chats/{document_id}")
def save_chat(document_id: str, body: dict):
    from db import save_message
    save_message(document_id, body["role"], body["content"], body.get("sources", []))
    return {"status": "saved"}