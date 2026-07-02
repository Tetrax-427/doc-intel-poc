"""
Endpoints:
    POST   /api-keys
    GET    /api-keys
    DELETE /api-keys/{key_id}

    POST   /webhooks
    GET    /webhooks
    DELETE /webhooks/{webhook_id}
    POST   /webhooks/{webhook_id}/test
    GET    /webhooks/logs

Note: /webhooks/logs must be registered BEFORE /webhooks/{webhook_id}/test
so FastAPI doesn't swallow "logs" as a webhook_id path param.
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, validator

from core.responses import internal_error, not_found
from db import get_supabase_admin
from webhooks import send_webhook

class CreateWebhookRequest(BaseModel):
    name: str
    url: str
    events: list[str] = ["extraction.complete"]
    secret: str | None = None

    @validator("name")
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()

    @validator("url")
    def url_not_empty(cls, v):
        if not v.strip():
            raise ValueError("url cannot be empty")
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v.strip()

    @validator("events")
    def events_not_empty(cls, v):
        if not v:
            raise ValueError("events list cannot be empty")
        return v
    
router = APIRouter(tags=["Integration"])

# /webhooks/logs is registered first to avoid path-param collision.

@router.get("/webhooks/logs")
def webhook_logs():
    """Return the 50 most recent webhook delivery log entries."""
    result = (
        get_supabase_admin().table("webhook_logs")
        .select("*")
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return result.data or []


@router.post("/webhooks")
def create_webhook(req: CreateWebhookRequest):
    """Register a new webhook endpoint."""
    try:
        result = get_supabase_admin().table("webhooks").insert({
            "name": req.name,
            "url": req.url,
            "events": req.events,
            "secret": req.secret,
        }).execute()
        return result.data[0]
    except Exception as exc:
        return internal_error(f"Could not create webhook: {exc}")


@router.get("/webhooks")
def list_webhooks():
    """List all registered webhooks."""
    result = (
        get_supabase_admin().table("webhooks")
        .select("id, name, url, events, is_active, last_triggered, fail_count, created_at")
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


@router.delete("/webhooks/{webhook_id}")
def delete_webhook(webhook_id: str):
    """Permanently delete a webhook."""
    get_supabase_admin().table("webhooks").delete().eq("id", webhook_id).execute()
    return {"status": "deleted", "webhook_id": webhook_id}


@router.post("/webhooks/{webhook_id}/test")
def test_webhook(webhook_id: str):
    """Send a test ping to a webhook to verify it's reachable."""
    result = get_supabase_admin().table("webhooks").select("*").eq("id", webhook_id).execute()
    if not result.data:
        return not_found("Webhook")

    webhook = result.data[0]
    try:
        success = send_webhook(webhook, "test.ping", {
            "message": "This is a test ping from DocIntel",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return {"success": success}
    except Exception as exc:
        return internal_error(f"Test ping failed: {exc}")
