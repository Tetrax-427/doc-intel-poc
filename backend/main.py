import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form
from dotenv import load_dotenv
from ingestion import ingest_file
from retrieval import query_document, query_document_stream
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    """Dependency — validates API key if provided"""
    if not api_key:
        return None  # no key = UI mode, allow through
    from api_keys import validate_api_key
    is_valid, reason = validate_api_key(api_key)
    if not is_valid:
        raise HTTPException(status_code=401, detail=reason)
    return api_key


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
    use_llamaparse: str = Form("True"),
    vision_template: str = Form("general")
):
    allowed = [".pdf", ".docx", ".txt", ".csv", ".xlsx", ".rtf", ".md",
               ".png", ".jpg", ".jpeg", ".webp", ".tiff"]
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed:
        return {"error": f"Unsupported file type: {ext}"}

    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    use_lp = use_llamaparse.lower() == "true"
    result = ingest_file(temp_path, use_llamaparse=use_lp, vision_template=vision_template)

    if "error" not in result:
        try:
            from retrieval import generate_summary
            from db import save_summary
            summary_data = generate_summary(result["document_id"])
            save_summary(result["document_id"], summary_data["summary"], summary_data["summary_short"])
            result["summary_short"] = summary_data["summary_short"]
        except Exception as e:
            print(f"Summary generation failed: {e}")

    return result

class ExtractRequest(BaseModel):
    document_id: str
    fields: dict

@app.post("/extract")
def extract(req: ExtractRequest):
    from retrieval import extract_fields
    from webhooks import trigger_webhooks
    result = extract_fields(req.document_id, req.fields)

    # Trigger webhooks
    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "extracted": result.get("extracted"),
        "validation": result.get("validation")
    })

    return result

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

@app.get("/summary/{document_id}")
def get_doc_summary(document_id: str):
    from db import get_summary
    from retrieval import generate_summary
    import json

    data = get_summary(document_id)

    # If no summary yet, generate it now
    if not data.get("summary"):
        summary_data = generate_summary(document_id)
        from db import save_summary
        save_summary(document_id, summary_data["summary"], summary_data["summary_short"])
        data = summary_data

    try:
        parsed = json.loads(data.get("summary", "{}"))
    except Exception:
        parsed = {}

    return {
        "summary_short": data.get("summary_short", ""),
        "details": parsed
    }

class URLRequest(BaseModel):
    url: str

@app.post("/ingest-url")
async def ingest_from_url(req: URLRequest):
    from ingestion import ingest_url
    result = ingest_url(req.url)

    if "error" not in result:
        try:
            from retrieval import generate_summary
            from db import save_summary
            summary_data = generate_summary(result["document_id"])
            save_summary(result["document_id"], summary_data["summary"], summary_data["summary_short"])
            result["summary_short"] = summary_data["summary_short"]
        except Exception as e:
            print(f"Summary generation failed: {e}")

    return result 

from fastapi.responses import Response

class ExportRequest(BaseModel):
    document_id: str
    file_name: str
    messages: list[dict]
    summary: dict = {}

@app.post("/export/pdf")
def export_pdf(req: ExportRequest):
    from export import export_chat_pdf
    pdf_bytes = export_chat_pdf(req.file_name, req.messages, req.summary)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=DocIntel_{req.file_name}.pdf"}
    )

@app.post("/export/docx")
def export_docx(req: ExportRequest):
    from export import export_chat_docx
    docx_bytes = export_chat_docx(req.file_name, req.messages, req.summary)
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=DocIntel_{req.file_name}.docx"}
    )


@app.get("/usage")
def get_usage():
    from llm.usage import get_usage_summary
    return get_usage_summary()

@app.get("/templates")
def get_templates():
    from schemas.templates import list_templates
    return list_templates()

@app.get("/templates/{template_id}")
def get_template(template_id: str):
    from schemas.templates import get_template
    template = get_template(template_id)
    if not template:
        return {"error": f"Template '{template_id}' not found"}
    return template

class NLExtractRequest(BaseModel):
    document_id: str
    instruction: str
    preview_only: bool = False  # if True, return schema without extracting

@app.post("/extract/nl")
def extract_natural_language(req: NLExtractRequest):
    from retrieval import nl_to_schema, extract_nl
    from webhooks import trigger_webhooks

    if req.preview_only:
        schema = nl_to_schema(req.instruction)
        return {"schema": schema, "extracted": None, "validation": None}

    result = extract_nl(req.document_id, req.instruction)

    trigger_webhooks("extraction.complete", {
        "document_id": req.document_id,
        "instruction": req.instruction,
        "extracted": result.get("extracted"),
        "validation": result.get("validation")
    })

    return result

class CreateKeyRequest(BaseModel):
    name: str
    rate_limit: int = 100

@app.post("/api-keys")
def create_key(req: CreateKeyRequest):
    from api_keys import create_api_key
    return create_api_key(req.name, req.rate_limit)

@app.get("/api-keys")
def list_keys():
    from api_keys import list_api_keys
    return list_api_keys()

@app.delete("/api-keys/{key_id}")
def revoke_key(key_id: str):
    from api_keys import revoke_api_key
    revoke_api_key(key_id)
    return {"status": "revoked"}

class CreateWebhookRequest(BaseModel):
    name: str
    url: str
    events: list[str] = ["extraction.complete"]
    secret: str = None

@app.post("/webhooks")
def create_webhook(req: CreateWebhookRequest):
    from db import supabase
    result = supabase.table("webhooks").insert({
        "name": req.name,
        "url": req.url,
        "events": req.events,
        "secret": req.secret
    }).execute()
    return result.data[0]

@app.get("/webhooks")
def list_webhooks():
    from db import supabase
    result = supabase.table("webhooks")\
        .select("id, name, url, events, is_active, last_triggered, fail_count, created_at")\
        .order("created_at", desc=True)\
        .execute()
    return result.data or []

@app.delete("/webhooks/{webhook_id}")
def delete_webhook(webhook_id: str):
    from db import supabase
    supabase.table("webhooks").delete().eq("id", webhook_id).execute()
    return {"status": "deleted"}

@app.get("/webhooks/logs")
def webhook_logs():
    from db import supabase
    result = supabase.table("webhook_logs")\
        .select("*")\
        .order("created_at", desc=True)\
        .limit(50)\
        .execute()
    return result.data or []

@app.post("/webhooks/{webhook_id}/test")
def test_webhook(webhook_id: str):
    from db import supabase
    from webhooks import send_webhook
    result = supabase.table("webhooks").select("*").eq("id", webhook_id).execute()
    if not result.data:
        return {"error": "Webhook not found"}
    webhook = result.data[0]
    success = send_webhook(webhook, "test.ping", {
        "message": "This is a test ping from DocIntel",
        "timestamp": datetime.utcnow().isoformat()
    })
    return {"success": success}