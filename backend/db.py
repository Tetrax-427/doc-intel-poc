"""
db.py
Supabase client and all table helpers.

Changes in this phase (Security + Org/Team):
  - get_document()         now checks visibility + org isolation
  - get_all_documents()    now filters by visibility rules
  - insert_document()      now accepts org_id, team_id, visibility
  - delete_document_by_id() cascade order updated for lineage_logs
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

def get_supabase_admin():
    """
    Service-role Supabase client — bypasses RLS.
    Used by db_*.py modules that need to read/write across users
    (org management, audit logs, usage aggregation, etc.) after
    permission checks have already happened in the router layer.
    """
    return create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_KEY"),
    )
# ── Documents ─────────────────────────────────────────────────────────────────

def insert_document(
    name: str,
    user_id: str = "anonymous",
    org_id: str | None = None,
    team_id: str | None = None,
    visibility: str = "private",
) -> str:
    result = get_supabase_admin().table("documents").insert({
        "name":       name,
        "user_id":    user_id,
        "org_id":     org_id,
        "team_id":    team_id,
        "visibility": visibility,
    }).execute()
    return result.data[0]["id"]


def get_document(document_id: str, user_id: str = "anonymous") -> dict | None:
    """
    Return a single document record by ID, scoped to the requesting user.
    Visibility enforcement is handled by RLS — this is a secondary check.
    Returns None if not found or access denied.
    """
    try:
        result = (
            get_supabase_admin().table("documents")
            .select(
                "id, name, summary_short, doc_type, classification_confidence, "
                "requires_review, created_at, user_id, org_id, team_id, visibility"
            )
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


def get_document_any_visibility(document_id: str) -> dict | None:
    """
    Fetch a document without user_id scoping.
    Used internally when we need to check visibility rules manually
    (e.g. in can_access_document() for team/org visibility checks).
    Caller is responsible for permission checks.
    """
    try:
        result = (
            get_supabase_admin().table("documents")
            .select(
                "id, name, summary_short, doc_type, classification_confidence, "
                "requires_review, created_at, user_id, org_id, team_id, visibility"
            )
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception:
        return None


def get_all_documents(
    user_id: str = "anonymous",
    org_id: str | None = None,
    team_id: str | None = None,
    visibility_filter: str | None = None,
) -> list[dict]:
    """
    Fetch documents visible to a user.

    Without org_id: returns only the user's own documents (backward compat).
    With org_id: RLS handles visibility — returns all documents the user
    can see (own + team + org depending on membership and visibility setting).
    """
    query = (
        get_supabase_admin().table("documents")
        .select(
            "id, name, summary_short, doc_type, classification_confidence, "
            "requires_review, created_at, user_id, org_id, team_id, visibility"
        )
        .eq("user_id", user_id)
        .order("created_at", desc=True)
    )

    if visibility_filter:
        query = query.eq("visibility", visibility_filter)

    result = query.execute()
    return result.data or []


def update_document_visibility(
    document_id: str,
    user_id: str,
    visibility: str,
    team_id: str | None = None,
    org_id: str | None = None,
) -> bool:
    """
    Update visibility of a document. Owner only.
    Returns True if updated, False if not found / not owner.
    """
    try:
        update_data: dict = {"visibility": visibility}
        if team_id is not None:
            update_data["team_id"] = team_id
        if org_id is not None:
            update_data["org_id"] = org_id

        result = (
            get_supabase_admin().table("documents")
            .update(update_data)
            .eq("id", document_id)
            .eq("user_id", user_id)
            .execute()
        )
        return bool(result.data)
    except Exception:
        return False


def delete_document_by_id(document_id: str, user_id: str = "anonymous") -> None:
    """
    Delete all data for a document in cascade order, then the document itself.

    Cascade order (respects FK dependencies):
      1. chunks               (document_id = uuid FK)
      2. extraction_results   (document_id = text FK)
      3. lineage_logs         (document_id = text column)
      4. llm_cache            (document_id = text column)
      5. documents            (scoped to user_id for security)

    user_id scopes the documents delete so a caller can only delete
    documents they own. Chunks/logs are deleted by document_id only.
    Silently returns if not found (idempotent).
    """
    get_supabase_admin().table("chunks").delete().eq("document_id", document_id).execute()
    get_supabase_admin().table("extraction_results").delete().eq("document_id", document_id).execute()
    get_supabase_admin().table("lineage_logs").delete().eq("document_id", document_id).execute()
    get_supabase_admin().table("llm_cache").delete().eq("document_id", document_id).execute()
    get_supabase_admin().table("documents").delete().eq("id", document_id).eq("user_id", user_id).execute()


def save_summary(document_id: str, summary: str, summary_short: str):
    get_supabase_admin().table("documents").update({
        "summary":       summary,
        "summary_short": summary_short,
    }).eq("id", document_id).execute()


def get_summary(document_id: str) -> dict:
    result = (
        get_supabase_admin().table("documents")
        .select("summary, summary_short")
        .eq("id", document_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return {"summary": None, "summary_short": None}


# ── Classification ────────────────────────────────────────────────────────────

def save_classification(document_id: str, classification: dict):
    get_supabase_admin().table("documents").update({
        "doc_type":                  classification.get("doc_type"),
        "classification_confidence": classification.get("confidence"),
        "classification_data":       classification,
        "requires_review":           classification.get("requires_human_review", False),
    }).eq("id", document_id).execute()


def get_classification(document_id: str) -> dict | None:
    result = (
        get_supabase_admin().table("documents")
        .select("doc_type, classification_confidence, classification_data, requires_review")
        .eq("id", document_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


# ── Chunks ────────────────────────────────────────────────────────────────────

def insert_chunks(chunks: list[dict]):
    if not chunks:
        return
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        get_supabase_admin().table("chunks").insert(batch).execute()


def get_chunks_by_document(document_id: str) -> list[dict]:
    result = (
        get_supabase_admin().table("chunks")
        .select("*")
        .eq("document_id", document_id)
        .execute()
    )
    return result.data or []


def get_parent_chunk(parent_chunk_id: str) -> dict | None:
    try:
        result = (
            get_supabase_admin().table("chunks")
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
        get_supabase_admin().table("chunks")
        .select("*")
        .in_("document_id", doc_ids)
        .execute()
    )
    return result.data or []


# ── Chats ─────────────────────────────────────────────────────────────────────

def save_message(document_id: str, role: str, content: str, sources: list = None):
    get_supabase_admin().table("chats").insert({
        "document_id": document_id,
        "role":        role,
        "content":     content,
        "sources":     sources or [],
    }).execute()


def get_chat_history(document_id: str) -> list[dict]:
    result = (
        get_supabase_admin().table("chats")
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
    get_supabase_admin().table("review_corrections").insert({
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
        get_supabase_admin().table("review_corrections")
        .select("*")
        .eq("doc_type",   doc_type)
        .eq("field_name", field_name)
        .eq("action",     "correct")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── API Keys ──────────────────────────────────────────────────────────────────

def mark_api_key_rotating(key_id: str, grace_expires_at: str) -> None:
    get_supabase_admin().table("api_keys").update({
        "status":           "rotating",
        "grace_expires_at": grace_expires_at,
    }).eq("id", key_id).execute()


def get_api_key_by_id(key_id: str, user_id: str) -> dict | None:
    resp = (
        get_supabase_admin().table("api_keys")
        .select("*")
        .eq("id", key_id)
        .eq("user_id", user_id)
        .execute()
    )
    return resp.data[0] if resp.data else None