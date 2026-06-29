"""
tests/test_usage.py
Tests for:
  - core/quota_checker.py — quota resolution and enforcement
  - db_quotas.py          — quota CRUD validation
  - Usage endpoint logic
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user(user_id="user-1", org_id=None, team_id=None,
              org_role=None, is_dev=False):
    from core.auth import UserContext
    return UserContext(
        user_id=user_id, email=f"{user_id}@test.com",
        is_dev=is_dev, org_id=org_id, org_role=org_role,
        team_id=team_id,
    )


# ---------------------------------------------------------------------------
# Quota type validation
# ---------------------------------------------------------------------------

class TestQuotaTypeValidation:

    def test_valid_quota_types_accepted(self):
        from db_quotas import VALID_QUOTA_TYPES, set_quota
        for qt in VALID_QUOTA_TYPES:
            with patch("db_quotas._sb") as mock_sb:
                mock_sb.return_value.table.return_value.upsert.return_value.execute.return_value = \
                    MagicMock(data=[{"id": "q-1", "quota_type": qt, "limit_value": 10}])
                result = set_quota(
                    quota_type=qt,
                    limit_value=10,
                    set_by="admin-1",
                    user_id="user-1",
                )
                assert result["quota_type"] == qt

    def test_invalid_quota_type_raises(self):
        from db_quotas import set_quota
        with pytest.raises(ValueError) as exc:
            set_quota(
                quota_type="max_something_invalid",
                limit_value=10,
                set_by="admin-1",
                user_id="user-1",
            )
        assert "quota_type" in str(exc.value).lower()

    def test_negative_limit_raises(self):
        from routers.usage import SetQuotaRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SetQuotaRequest(
                quota_type="max_documents",
                limit_value=-5,
                user_id="user-1",
            )

    def test_zero_limit_raises(self):
        from routers.usage import SetQuotaRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SetQuotaRequest(
                quota_type="max_documents",
                limit_value=0,
                user_id="user-1",
            )


# ---------------------------------------------------------------------------
# Quota scope validation
# ---------------------------------------------------------------------------

class TestQuotaScopeValidation:

    def test_exactly_one_scope_required(self):
        from db_quotas import set_quota
        with pytest.raises(ValueError) as exc:
            set_quota(
                quota_type="max_documents",
                limit_value=10,
                set_by="admin-1",
                user_id="user-1",
                org_id="org-1",  # two scopes — invalid
            )
        assert "exactly one" in str(exc.value).lower()

    def test_no_scope_raises(self):
        from db_quotas import set_quota
        with pytest.raises(ValueError) as exc:
            set_quota(
                quota_type="max_documents",
                limit_value=10,
                set_by="admin-1",
                # no user_id, team_id, or org_id
            )
        assert "exactly one" in str(exc.value).lower()

    def test_user_scope_accepted(self):
        from db_quotas import set_quota
        with patch("db_quotas._sb") as mock_sb:
            mock_sb.return_value.table.return_value.upsert.return_value.execute.return_value = \
                MagicMock(data=[{"id": "q-1", "quota_type": "max_documents", "limit_value": 50}])
            result = set_quota(
                quota_type="max_documents",
                limit_value=50,
                set_by="admin-1",
                user_id="user-1",
            )
            assert result is not None

    def test_org_scope_accepted(self):
        from db_quotas import set_quota
        with patch("db_quotas._sb") as mock_sb:
            mock_sb.return_value.table.return_value.upsert.return_value.execute.return_value = \
                MagicMock(data=[{"id": "q-2", "quota_type": "max_llm_cost_month", "limit_value": 10.0}])
            result = set_quota(
                quota_type="max_llm_cost_month",
                limit_value=10.0,
                set_by="admin-1",
                org_id="org-1",
            )
            assert result is not None


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------

class TestQuotaEnforcement:

    def test_dev_user_skips_quota_check(self):
        from core.quota_checker import check_upload_quota
        user = make_user(is_dev=True)
        check_upload_quota(user)  # should not raise, no DB call

    def test_hard_limit_exceeded_raises_429(self):
        from core.quota_checker import _enforce
        with pytest.raises(HTTPException) as exc:
            _enforce(
                current=51,
                limit=50,
                is_hard=True,
                quota_type="max_documents",
            )
        assert exc.value.status_code == 429
        assert exc.value.detail["code"] == "QUOTA_EXCEEDED"

    def test_soft_limit_exceeded_does_not_raise(self):
        from core.quota_checker import _enforce
        # Soft limit — should log but not raise
        _enforce(
            current=51,
            limit=50,
            is_hard=False,
            quota_type="max_documents",
        )

    def test_within_limit_passes(self):
        from core.quota_checker import _enforce
        _enforce(
            current=10,
            limit=50,
            is_hard=True,
            quota_type="max_documents",
        )  # should not raise

    def test_exactly_at_limit_raises(self):
        """current >= limit triggers enforcement."""
        from core.quota_checker import _enforce
        with pytest.raises(HTTPException):
            _enforce(
                current=50,
                limit=50,
                is_hard=True,
                quota_type="max_documents",
            )


# ---------------------------------------------------------------------------
# Quota resolution order
# ---------------------------------------------------------------------------

class TestQuotaResolutionOrder:

    def test_user_quota_takes_precedence(self):
        """User-specific quota should win over team/org/default."""
        from core.quota_checker import _resolve_quota
        user = make_user(user_id="user-1", org_id="org-1", team_id="team-1")

        def mock_execute():
            m = MagicMock()
            m.data = [{"limit_value": 25, "is_hard_limit": True}]
            return m

        with patch("core.quota_checker._get_sb") as mock_sb:
            mock_table = MagicMock()
            mock_sb.return_value.table.return_value = mock_table
            mock_table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute = mock_execute

            limit, is_hard = _resolve_quota(user, "max_documents")
            assert limit == 25.0
            assert is_hard is True

    def test_falls_back_to_config_default(self):
        """When no quota is set, config defaults are used."""
        from core.quota_checker import _resolve_quota
        user = make_user(user_id="user-no-quota")

        with patch("core.quota_checker._get_sb") as mock_sb:
            # All DB queries return no data
            mock_execute = MagicMock()
            mock_execute.return_value.data = []
            mock_sb.return_value.table.return_value.select.return_value\
                .eq.return_value.eq.return_value.limit.return_value.execute = mock_execute

            with patch("core.quota_checker.app_config") as mock_config:
                mock_config.default_max_documents = 50
                limit, is_hard = _resolve_quota(user, "max_documents")
                assert limit == 50.0


# ---------------------------------------------------------------------------
# Usage endpoint permissions
# ---------------------------------------------------------------------------

class TestUsageEndpointPermissions:

    def test_non_admin_cannot_access_org_usage(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="member")
        with pytest.raises(HTTPException) as exc:
            require_org_admin(user)
        assert exc.value.status_code == 403

    def test_org_admin_can_access_org_usage(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="org_admin")
        require_org_admin(user)  # should not raise

    def test_team_lead_can_access_team_usage(self):
        from core.permissions import can_access_team
        user = make_user(org_id="org-1", team_id="team-1", team_role="team_lead")
        assert can_access_team(user, "team-1") is True

    def test_member_cannot_access_other_team_usage(self):
        from core.permissions import can_access_team
        user = make_user(org_id="org-1", team_id="team-1", team_role="member")
        assert can_access_team(user, "team-2") is False

    def test_non_admin_cannot_set_quota(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="member")
        with pytest.raises(HTTPException) as exc:
            require_org_admin(user)
        assert exc.value.status_code == 403


# ---------------------------------------------------------------------------
# SetQuotaRequest validation
# ---------------------------------------------------------------------------

class TestSetQuotaRequestValidation:

    def test_valid_request(self):
        from routers.usage import SetQuotaRequest
        req = SetQuotaRequest(
            quota_type="max_documents",
            limit_value=100,
            user_id="user-1",
        )
        assert req.quota_type == "max_documents"
        assert req.limit_value == 100

    def test_invalid_quota_type_raises(self):
        from routers.usage import SetQuotaRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SetQuotaRequest(
                quota_type="max_something_invalid",
                limit_value=100,
                user_id="user-1",
            )

    def test_multiple_scopes_raises(self):
        from routers.usage import SetQuotaRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SetQuotaRequest(
                quota_type="max_documents",
                limit_value=100,
                user_id="user-1",
                org_id="org-1",  # two scopes
            )

    def test_no_scope_raises(self):
        from routers.usage import SetQuotaRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SetQuotaRequest(
                quota_type="max_documents",
                limit_value=100,
                # no scope
            )


# ---------------------------------------------------------------------------
# Quota status
# ---------------------------------------------------------------------------

class TestGetQuotaStatus:

    def test_dev_user_returns_no_enforcement(self):
        from core.quota_checker import get_quota_status
        user = make_user(is_dev=True)
        result = get_quota_status(user)
        assert "note" in result
        assert "dev" in result["note"].lower()

    def test_returns_all_quota_types(self):
        from core.quota_checker import get_quota_status
        user = make_user(user_id="user-1")

        with patch("core.quota_checker._resolve_quota") as mock_quota, \
             patch("core.quota_checker._get_document_count") as mock_docs, \
             patch("core.quota_checker._get_uploads_today") as mock_up, \
             patch("core.quota_checker._get_llm_cost_this_month") as mock_cost, \
             patch("core.quota_checker._get_queries_today") as mock_q:

            mock_quota.return_value = (50.0, True)
            mock_docs.return_value  = 10
            mock_up.return_value    = 3
            mock_cost.return_value  = 1.5
            mock_q.return_value     = 25

            result = get_quota_status(user)
            assert "quotas" in result
            assert len(result["quotas"]) == 4

            types = {q["quota_type"] for q in result["quotas"]}
            assert "max_documents"       in types
            assert "max_uploads_per_day" in types
            assert "max_llm_cost_month"  in types
            assert "max_queries_per_day" in types