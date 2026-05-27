import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from ingestion import ingest_file
from retrieval import query_document, query_document_stream
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
class QueryRequest(BaseModel):
    question: str
    document_id: str = None

load_dotenv()
class QueryRequest(BaseModel):
    question: str
    document_id: str = None
    document_ids: list[str] = None
    history: list[dict] = []
    history_summary: str = ""

app = FastAPI()

@app.on_event("startup")
async def warmup():
    print("Warming up embedding model...")
    from ingestion import get_embed_model
    get_embed_model()
    print("Model ready.")


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "doc-intel API running"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    use_llamaparse: str = Form("True")
):
    allowed = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".rtf", ".md"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        return {"error": f"Unsupported file type: {ext}"}

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    use_lp = use_llamaparse.lower() == "true"
    result = ingest_file(temp_path, use_llamaparse=use_lp)
    return result

class ExtractRequest(BaseModel):
    document_id: str
    fields: dict

@app.post("/extract")
def extract(req: ExtractRequest):
    from retrieval import extract_fields
    return extract_fields(req.document_id, req.fields)

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

@app.get("/tables/{document_id}")
def get_tables(document_id: str):
    from retrieval import extract_tables
    tables = extract_tables(document_id)
    return {"tables": tables}

import base64

@app.post("/query")
def query(req: QueryRequest):
    return query_document(req.question, req.document_id, req.document_ids, req.history, req.history_summary)

@app.post("/query/stream")
def query_stream(req: QueryRequest):
    def event_stream():
        try:
            for token in query_document_stream(req.question, req.document_id, req.document_ids, req.history, req.history_summary):
                encoded = base64.b64encode(token.encode()).decode()
                yield f"data: {encoded}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
class CompressRequest(BaseModel):
    messages: list[dict]

@app.post("/compress")
def compress(req: CompressRequest):
    from retrieval import compress_history
    summary = compress_history(req.messages)
    return {"summary": summary}