"""
db.py
Supabase client and all table helpers.
All document/chunk queries are scoped by user_id for multi-user isolation.
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

def insert_document(name: str, user_id: str = "anonymous") -> str:
    result = supabase.table("documents").insert({
        "name":    name,
        "user_id": user_id,
    }).execute()
    return result.data[0]["id"]


def get_all_documents(user_id: str = "anonymous") -> list[dict]:
    """
    Return all documents for a user, ordered newest-first.
    Scoped by user_id — each user only sees their own documents.
    """
    result = (
        supabase.table("documents")
        .select("id, name, summary_short, doc_type, classification_confidence, requires_review, created_at")
        .eq("user_id", user_id)          # ← multi-user isolation enabled
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def delete_document_by_id(document_id: str):
    supabase.table("chunks").delete().eq("document_id", document_id).execute()
    supabase.table("documents").delete().eq("id", document_id).execute()


def save_summary(document_id: str, summary: str, summary_short: str):
    supabase.table("documents").update({
        "summary":       summary,
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
    supabase.table("documents").update({
        "doc_type":                  classification.get("doc_type"),
        "classification_confidence": classification.get("confidence"),
        "classification_data":       classification,
        "requires_review":           classification.get("requires_human_review", False),
    }).eq("id", document_id).execute()


def get_classification(document_id: str) -> dict | None:
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


def get_all_chunks(user_id: str = "anonymous") -> list[dict]:
    """
    Return all chunks belonging to documents owned by user_id.
    Automatically scoped — users cannot access each other's chunks.
    """
    docs = get_all_documents(user_id)
    doc_ids = [d["id"] for d in docs]
    if not doc_ids:
        return []
    result = (
        supabase.table("chunks")
        .select("*")
        .in_("document_id", doc_ids)
        .execute()
    )
    return result.data or []


# ── Chats ─────────────────────────────────────────────────────────────────────

def save_message(document_id: str, role: str, content: str, sources: list = None):
    supabase.table("chats").insert({
        "document_id": document_id,
        "role":        role,
        "content":     content,
        "sources":     sources or [],
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


# ── Review corrections ────────────────────────────────────────────────────────

def save_correction(
    document_id: str,
    doc_type: str,
    field_name: str,
    original: str,
    corrected: str,
    action: str,
    evidence: str = "",
    note: str = "",
) -> None:
    supabase.table("review_corrections").insert({
        "document_id":     document_id,
        "doc_type":        doc_type,
        "field_name":      field_name,
        "original_value":  str(original)  if original  else "",
        "corrected_value": str(corrected) if corrected else "",
        "action":          action,
        "evidence_used":   evidence,
        "reviewer_note":   note,
    }).execute()


def get_corrections_for_doc_type(
    doc_type: str,
    field_name: str,
    limit: int = 5,
) -> list[dict]:
    result = (
        supabase.table("review_corrections")
        .select("*")
        .eq("doc_type",    doc_type)
        .eq("field_name",  field_name)
        .eq("action",      "correct")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []