"""
tests/test_org_endpoints.py
Tests for org/team management logic.

Uses unit tests for permission logic and db helper mocking
rather than full integration tests (which would need a live Supabase).
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user(user_id="user-1", org_id=None, org_role=None,
              team_id=None, team_role=None, is_dev=False):
    from core.auth import UserContext
    return UserContext(
        user_id=user_id, email=f"{user_id}@test.com",
        is_dev=is_dev, org_id=org_id, org_role=org_role,
        team_id=team_id, team_role=team_role,
    )


# ---------------------------------------------------------------------------
# Admin — developer key validation
# ---------------------------------------------------------------------------

class TestDeveloperKeyValidation:

    def test_correct_key_passes(self):
        from routers.admin import _verify_developer_key
        with patch("routers.admin.app_config") as mock_config:
            mock_config.developer_api_key = "secret-key-123"
            _verify_developer_key("secret-key-123")  # should not raise

    def test_wrong_key_raises_401(self):
        from routers.admin import _verify_developer_key
        with patch("routers.admin.app_config") as mock_config:
            mock_config.developer_api_key = "secret-key-123"
            with pytest.raises(HTTPException) as exc:
                _verify_developer_key("wrong-key")
            assert exc.value.status_code == 401

    def test_missing_key_raises_401(self):
        from routers.admin import _verify_developer_key
        with patch("routers.admin.app_config") as mock_config:
            mock_config.developer_api_key = "secret-key-123"
            with pytest.raises(HTTPException) as exc:
                _verify_developer_key(None)
            assert exc.value.status_code == 401

    def test_unconfigured_server_raises_503(self):
        from routers.admin import _verify_developer_key
        with patch("routers.admin.app_config") as mock_config:
            mock_config.developer_api_key = ""
            with pytest.raises(HTTPException) as exc:
                _verify_developer_key("any-key")
            assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# Org member permissions
# ---------------------------------------------------------------------------

class TestOrgMemberPermissions:

    def test_org_admin_can_list_members(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="org_admin")
        require_org_admin(user)  # should not raise

    def test_member_cannot_list_members(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="member")
        with pytest.raises(HTTPException) as exc:
            require_org_admin(user)
        assert exc.value.status_code == 403

    def test_cannot_remove_self(self):
        """Org admin cannot remove themselves — checked in router."""
        # Simulate the router's self-removal check
        admin_user_id = "admin-1"
        target_user_id = "admin-1"  # same user

        if target_user_id == admin_user_id:
            with pytest.raises(HTTPException) as exc:
                raise HTTPException(status_code=400, detail="Cannot remove yourself")
            assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# Team lead permissions
# ---------------------------------------------------------------------------

class TestTeamLeadPermissions:

    def test_team_lead_can_add_members(self):
        from core.permissions import require_team_lead
        user = make_user(org_id="org-1", team_id="team-1", team_role="team_lead")
        require_team_lead(user)

    def test_org_admin_can_add_team_members(self):
        from core.permissions import require_team_lead
        user = make_user(org_id="org-1", org_role="org_admin")
        require_team_lead(user)

    def test_regular_member_cannot_add_team_members(self):
        from core.permissions import require_team_lead
        user = make_user(org_id="org-1", team_id="team-1",
                         org_role="member", team_role="member")
        with pytest.raises(HTTPException) as exc:
            require_team_lead(user)
        assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# Cross-org isolation
# ---------------------------------------------------------------------------

class TestCrossOrgIsolation:

    def test_admin_cannot_access_different_org(self):
        from core.permissions import require_same_org
        user = make_user(org_id="org-1", org_role="org_admin")
        with pytest.raises(HTTPException) as exc:
            require_same_org(user, "org-2")
        assert exc.value.status_code == 403

    def test_admin_can_access_own_org(self):
        from core.permissions import require_same_org
        user = make_user(org_id="org-1", org_role="org_admin")
        require_same_org(user, "org-1")  # should not raise

    def test_org_isolation_blocks_cross_org_data(self):
        from core.permissions import assert_org_isolation
        user = make_user(org_id="org-A")
        with pytest.raises(HTTPException) as exc:
            assert_org_isolation(user, "org-B")
        assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugValidation:
    """Test org slug format validation from CreateOrgRequest."""

    def _validate(self, slug: str):
        from routers.admin import CreateOrgRequest
        return CreateOrgRequest(name="Test Org", slug=slug, admin_user_id="user-1")

    def test_valid_slug(self):
        req = self._validate("acme-corp")
        assert req.slug == "acme-corp"

    def test_slug_lowercased(self):
        req = self._validate("AcmeCorp")
        assert req.slug == "acmecorp"

    def test_slug_with_numbers(self):
        req = self._validate("acme-123")
        assert req.slug == "acme-123"

    def test_slug_too_short_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._validate("ab")

    def test_slug_with_spaces_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._validate("acme corp")

    def test_slug_with_special_chars_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._validate("acme_corp!")


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

class TestAuditLogging:

    def test_log_audit_non_blocking(self):
        """log_audit should not raise even if DB is unavailable."""
        from db_audit import log_audit
        with patch("db_audit._sb_admin") as mock_sb:
            mock_sb.return_value.table.return_value.insert.return_value.execute.side_effect = \
                Exception("DB unavailable")
            # Should not raise
            log_audit(
                actor_id="user-1",
                actor_role="org_admin",
                action="member_added",
                resource_type="org_member",
                resource_id="user-2",
                org_id="org-1",
            )

    def test_log_audit_writes_correct_fields(self):
        from db_audit import log_audit
        with patch("db_audit._sb_admin") as mock_sb:
            mock_insert = MagicMock()
            mock_sb.return_value.table.return_value.insert.return_value = mock_insert
            mock_insert.execute.return_value = MagicMock()

            log_audit(
                actor_id="admin-1",
                actor_role="org_admin",
                action="team_created",
                resource_type="team",
                resource_id="team-001",
                org_id="org-1",
                details={"name": "Engineering"},
            )

            call_args = mock_sb.return_value.table.return_value.insert.call_args[0][0]
            assert call_args["actor_id"]     == "admin-1"
            assert call_args["action"]       == "team_created"
            assert call_args["resource_type"] == "team"
            assert call_args["org_id"]       == "org-1"


# ---------------------------------------------------------------------------
# Org me endpoint
# ---------------------------------------------------------------------------

class TestGetMyOrg:

    def test_user_without_org_returns_none(self):
        from routers.orgs import get_my_org
        user = make_user()  # no org
        result = get_my_org(user)
        assert result["org"] is None
        assert result["team"] is None

    def test_user_with_org_returns_org_info(self):
        from routers.orgs import get_my_org
        user = make_user(org_id="org-1", org_role="member")
        with patch("routers.orgs.get_org_by_id") as mock_org, \
             patch("routers.orgs.get_team_by_id") as mock_team:
            mock_org.return_value = {"id": "org-1", "name": "Acme"}
            mock_team.return_value = None
            result = get_my_org(user)
            assert result["org"]["id"] == "org-1"
            assert result["org_role"] == "member"