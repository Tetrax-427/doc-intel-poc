"""
routers/schema_templates.py
CRUD for DB-backed nested schema templates (schemas.dynamic.SchemaSpec).

The nested equivalent of GET /templates (schemas/templates.py's flat,
code-defined _TEMPLATES) — but stored, so templates can be added/edited
without a redeploy and shared across a user/team/org, or published globally.

Visibility/scoping is derived from the caller's own UserContext
(org_id/team_id/is_org_admin) — never trusted from the request body, so a
caller can't create a template scoped to a team/org they don't belong to,
or silently promote their own template to global without the required role.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator, model_validator

from core.auth import get_current_user_context, get_user_id, UserContext
from core.responses import bad_request, not_found, error_response
from schemas.dynamic import SchemaSpec
from db import (
    create_schema_template,
    get_schema_template as _get_schema_template,
    list_schema_templates as _list_schema_templates,
    update_schema_template,
    delete_schema_template,
)

router = APIRouter(prefix="/schema-templates", tags=["Schema Templates"])


# ── Input models ──────────────────────────────────────────────────────────────

class CreateSchemaTemplateRequest(BaseModel):
    name: str
    description: str = ""
    schema_spec: dict  # validated against schemas.dynamic.SchemaSpec below
    visibility: str = "personal"  # "personal" | "team" | "org" | "global"

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v.strip()

    @field_validator("visibility")
    @classmethod
    def visibility_must_be_valid(cls, v):
        if v not in ("personal", "team", "org", "global"):
            raise ValueError("visibility must be one of: personal, team, org, global")
        return v

    @model_validator(mode="after")
    def schema_spec_must_validate(self):
        try:
            SchemaSpec.model_validate(self.schema_spec)
        except Exception as exc:
            raise ValueError(f"Invalid schema_spec: {exc}")
        return self


class UpdateSchemaTemplateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    schema_spec: dict | None = None

    @model_validator(mode="after")
    def schema_spec_must_validate_if_given(self):
        if self.schema_spec is not None:
            try:
                SchemaSpec.model_validate(self.schema_spec)
            except Exception as exc:
                raise ValueError(f"Invalid schema_spec: {exc}")
        return self


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
def list_schema_templates(
    user: UserContext = Depends(get_current_user_context),
):
    """Every template visible to the caller: global + own + team + org."""
    uid = get_user_id(user)
    team_ids = [user.team_id_str] if user.team_id_str else None
    org_ids  = [user.org_id_str] if user.org_id_str else None
    return _list_schema_templates(uid, team_ids=team_ids, org_ids=org_ids)


@router.get("/{template_id}")
def get_schema_template(
    template_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)
    team_ids = [user.team_id_str] if user.team_id_str else None
    org_ids  = [user.org_id_str] if user.org_id_str else None
    template = _get_schema_template(template_id, uid, team_ids=team_ids, org_ids=org_ids)
    if template is None:
        return not_found(f"Schema template '{template_id}'")
    return template


@router.post("")
def create_template(
    req: CreateSchemaTemplateRequest,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)

    is_global   = req.visibility == "global"
    scoped_user = None
    scoped_team = None
    scoped_org  = None

    if req.visibility == "personal":
        scoped_user = uid
    elif req.visibility == "team":
        if not user.team_id_str:
            return bad_request(
                "You're not a member of a team — can't create a team-scoped template.",
                code="SCHEMA_TEMPLATE_NO_TEAM",
            )
        scoped_team = user.team_id_str
    elif req.visibility == "org":
        if not user.org_id_str:
            return bad_request(
                "You're not a member of an org — can't create an org-scoped template.",
                code="SCHEMA_TEMPLATE_NO_ORG",
            )
        scoped_org = user.org_id_str
    elif is_global:
        if not user.is_org_admin:
            return error_response(
                "Only org admins can create global schema templates.",
                code="SCHEMA_TEMPLATE_GLOBAL_FORBIDDEN",
                status_code=403,
            )

    template_id = create_schema_template(
        name=req.name,
        schema_spec=req.schema_spec,
        created_by=uid,
        description=req.description,
        is_global=is_global,
        user_id=scoped_user,
        team_id=scoped_team,
        org_id=scoped_org,
    )
    return {"id": template_id}


@router.patch("/{template_id}")
def update_template(
    template_id: str,
    req: UpdateSchemaTemplateRequest,
    user: UserContext = Depends(get_current_user_context),
):
    """
    Creator-only. Visibility (personal/team/org/global) is NOT editable here —
    delete and recreate with the desired visibility instead, to avoid a
    non-admin silently editing a template into global via a partial update.
    """
    uid = get_user_id(user)
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not fields:
        return bad_request("No fields to update.", code="SCHEMA_TEMPLATE_EMPTY_UPDATE")

    updated = update_schema_template(template_id, created_by=uid, **fields)
    if not updated:
        return not_found(f"Schema template '{template_id}' (or you're not its creator)")
    return {"id": template_id, "updated": True}


@router.delete("/{template_id}")
def delete_template(
    template_id: str,
    user: UserContext = Depends(get_current_user_context),
):
    uid = get_user_id(user)
    deleted = delete_schema_template(template_id, created_by=uid)
    if not deleted:
        return not_found(f"Schema template '{template_id}' (or you're not its creator)")
    return {"id": template_id, "deleted": True}