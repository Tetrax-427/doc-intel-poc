"""
db.py
Supabase client and all table helpers.
All document/chunk queries are scoped by user_id for multi-user isolation.

- insert_chunks() now passes through pre-assigned `id` fields (parent chunks
  need stable IDs so child chunks can reference them via parent_chunk_id).
- get_parent_chunk() fetches a single chunk by ID for parent context expansion.
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

def get_document(document_id: str, user_id: str = "anonymous") -> dict | None:
    """
    Return a single document record by ID, scoped to the requesting user.

    Returns None if:
      - document not found
      - document belongs to a different user
      - any DB error occurs

    Note: file_path is NOT stored in the DB. Callers that need the file on
    disk must reconstruct it from the 'name' field:
        file_path = os.path.join(UPLOAD_DIR, doc["name"])
    """
    try:
        result = (
            supabase.table("documents")
            .select("id, name, summary_short, doc_type, classification_confidence, requires_review, created_at, user_id")
            .eq("id", document_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        data = result.data
        if not data:
            return None
        return data[0]
    except Exception:
        return None

def get_all_documents(user_id: str = "anonymous") -> list[dict]:
    result = (
        supabase.table("documents")
        .select("id, name, summary_short, doc_type, classification_confidence, requires_review, created_at")
        .eq("user_id", user_id)
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
    """
    Insert chunk rows into Supabase.

    D1 change: if a chunk dict includes an "id" key, it is passed through
    to Supabase so parent chunks get stable UUIDs that child chunks can
    reference via parent_chunk_id. Supabase will use the provided id
    instead of auto-generating one.

    Chunks without an "id" key are inserted as before (Supabase auto-assigns).
    Rows with embedding=None (parent chunks) are stored without an embedding
    vector — they are never searched directly, only fetched by ID.
    """
    if not chunks:
        return

    # Batch insert — Supabase handles up to 1000 rows per call cleanly.
    # Split into batches of 500 to stay well within limits.
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        supabase.table("chunks").insert(batch).execute()


def get_chunks_by_document(document_id: str) -> list[dict]:
    """
    Fetch all chunks for a document.

    D1 note: returns ALL chunk levels (parent, child, flat).
    Callers that only want searchable chunks (child + flat) must filter
    on metadata.chunk_level != "parent" themselves — retrieval.py does this.
    """
    result = (
        supabase.table("chunks")
        .select("*")
        .eq("document_id", document_id)
        .execute()
    )
    return result.data or []


def get_parent_chunk(parent_chunk_id: str) -> dict | None:
    """
    Fetch a single parent chunk by its ID.

    Used by retrieval.expand_to_parent_context() to retrieve the fuller
    parent text when a child chunk is matched during retrieval.

    Returns None if the chunk does not exist or the query fails.
    Never raises — callers fall back to child text on None.
    """
    try:
        result = (
            supabase.table("chunks")
            .select("id, content, metadata, document_id")
            .eq("id", parent_chunk_id)
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]
        return None
    except Exception:
        return None


def get_all_chunks(user_id: str) -> list[dict]:
    docs    = get_all_documents(user_id)
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
        .eq("doc_type",   doc_type)
        .eq("field_name", field_name)
        .eq("action",     "correct")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def mark_api_key_rotating(key_id: str, grace_expires_at: str) -> None:
    """
    Marks a key as 'rotating' — still valid until grace period expires.
    Called by rotate_api_key() in api_keys.py.
    """
    supabase.table("api_keys").update({
        "status":          "rotating",
        "grace_expires_at": grace_expires_at,
    }).eq("id", key_id).execute()
 
 
def get_api_key_by_id(key_id: str, user_id: str) -> dict | None:
    """
    Fetch one API key record by UUID + user_id ownership check.
    Returns None if not found or not owned by user_id.
    """
    resp = (
        supabase.table("api_keys")
        .select("*")
        .eq("id", key_id)
        .eq("user_id", user_id)
        .execute()
    )
    return resp.data[0] if resp.data else None
 