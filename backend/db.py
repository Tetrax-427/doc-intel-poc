import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def insert_document(name: str) -> str:
    result = supabase.table("documents").insert({"name": name}).execute()
    return result.data[0]["id"]

def insert_chunks(chunks: list[dict]):
    supabase.table("chunks").insert(chunks).execute()

def get_all_chunks() -> list[dict]:
    result = supabase.table("chunks").select("*").execute()

    return result.data

def get_all_documents() -> list[dict]:
    result = supabase.table("documents").select("id, name, summary_short, created_at").order("created_at", desc=True).execute()
    return result.data

def delete_document_by_id(document_id: str):
    supabase.table("chunks").delete().eq("document_id", document_id).execute()
    supabase.table("documents").delete().eq("id", document_id).execute()

def save_message(document_id: str, role: str, content: str, sources: list = None):
    supabase.table("chats").insert({
        "document_id": document_id,
        "role": role,
        "content": content,
        "sources": sources or []
    }).execute()

def get_chat_history(document_id: str) -> list[dict]:
    result = supabase.table("chats").select("*").eq("document_id", document_id).order("created_at").execute()
    return result.data


def save_summary(document_id: str, summary: str, summary_short: str):
    supabase.table("documents").update({
        "summary": summary,
        "summary_short": summary_short
    }).eq("id", document_id).execute()

def get_summary(document_id: str) -> dict:
    result = supabase.table("documents").select("summary, summary_short").eq("id", document_id).execute()
    if result.data:
        return result.data[0]
    return {"summary": None, "summary_short": None}