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


@app.post("/query")
def query(req: QueryRequest):
    return query_document(req.question, req.document_id)