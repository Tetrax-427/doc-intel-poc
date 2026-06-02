"""
db.py
Supabase client and all table helpers.

Changes from original:
- get_all_documents now selects doc_type + classification_confidence
- Added: save_classification, get_classification
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)


# ── Documents ─────────────────────────────────────────────────────────────────

def insert_document(name: str) -> str:
    result = supabase.table("documents").insert({"name": name}).execute()
    return result.data[0]["id"]


def get_all_documents() -> list[dict]:
    """
    Return all documents ordered newest-first.
    Includes classification columns so the UI can show doc-type badges.
    """
    result = (
        supabase.table("documents")
        .select("id, name, summary_short, doc_type, classification_confidence, requires_review, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def delete_document_by_id(document_id: str):
    supabase.table("chunks").delete().eq("document_id", document_id).execute()
    supabase.table("documents").delete().eq("id", document_id).execute()


def save_summary(document_id: str, summary: str, summary_short: str):
    supabase.table("documents").update({
        "summary": summary,
        "summary_short": summary_short,
    }).eq("id", document_id).execute()


def get_summary(document_id: str) -> dict:
    result = (
        supabase.table("documents")
        .select("summary, summary_short")
        .eq("id", document_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return {"summary": None, "summary_short": None}


# ── Classification ────────────────────────────────────────────────────────────

def save_classification(document_id: str, classification: dict):
    """
    Persist classification result to the documents table.
    Updates: doc_type, classification_confidence, classification_data, requires_review.
    """
    supabase.table("documents").update({
        "doc_type": classification.get("doc_type"),
        "classification_confidence": classification.get("confidence"),
        "classification_data": classification,
        "requires_review": classification.get("requires_human_review", False),
    }).eq("id", document_id).execute()


def get_classification(document_id: str) -> dict | None:
    """
    Return classification fields for a document.
    Returns None if document doesn't exist.
    """
    result = (
        supabase.table("documents")
        .select("doc_type, classification_confidence, classification_data, requires_review")
        .eq("id", document_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


# ── Chunks ────────────────────────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]):
    supabase.table("chunks").insert(chunks).execute()


def get_all_chunks() -> list[dict]:
    result = supabase.table("chunks").select("*").execute()
    return result.data


# ── Chats ─────────────────────────────────────────────────────────────────────

def save_message(document_id: str, role: str, content: str, sources: list = None):
    supabase.table("chats").insert({
        "document_id": document_id,
        "role": role,
        "content": content,
        "sources": sources or [],
    }).execute()


def get_chat_history(document_id: str) -> list[dict]:
    result = (
        supabase.table("chats")
        .select("*")
        .eq("document_id", document_id)
        .order("created_at")
        .execute()
    )
    return result.data
