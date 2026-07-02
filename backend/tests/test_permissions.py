"""
tests/test_permissions.py
Tests for core/permissions.py — RBAC checks and document access.
"""

import pytest
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user(
    user_id="user-1",
    org_id=None,
    org_role=None,
    team_id=None,
    team_role=None,
    can_read_team_documents=False,
    is_dev=False,
):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.auth import UserContext
    return UserContext(
        user_id=user_id,
        email=f"{user_id}@test.com",
        is_dev=is_dev,
        org_id=org_id,
        org_role=org_role,
        team_id=team_id,
        team_role=team_role,
        can_read_team_documents=can_read_team_documents,
    )


def make_doc(
    user_id="user-1",
    org_id=None,
    team_id=None,
    visibility="private",
):
    return {
        "id":         "doc-001",
        "user_id":    user_id,
        "org_id":     org_id,
        "team_id":    team_id,
        "visibility": visibility,
    }


# ---------------------------------------------------------------------------
# require_org_admin
# ---------------------------------------------------------------------------

class TestRequireOrgAdmin:

    def test_org_admin_passes(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="org_admin")
        require_org_admin(user)  # should not raise

    def test_member_blocked(self):
        from core.permissions import require_org_admin
        user = make_user(org_id="org-1", org_role="member")
        with pytest.raises(HTTPException) as exc:
            require_org_admin(user)
        assert exc.value.status_code == 403

    def test_no_org_blocked(self):
        from core.permissions import require_org_admin
        user = make_user()
        with pytest.raises(HTTPException) as exc:
            require_org_admin(user)
        assert exc.value.status_code == 403

    def test_dev_always_passes(self):
        from core.permissions import require_org_admin
        user = make_user(is_dev=True)
        require_org_admin(user)  # dev mode bypasses all checks


# ---------------------------------------------------------------------------
# require_team_lead
# ---------------------------------------------------------------------------

class TestRequireTeamLead:

    def test_team_lead_passes(self):
        from core.permissions import require_team_lead
        user = make_user(org_id="org-1", team_id="team-1", team_role="team_lead")
        require_team_lead(user)

    def test_org_admin_passes(self):
        from core.permissions import require_team_lead
        user = make_user(org_id="org-1", org_role="org_admin")
        require_team_lead(user)

    def test_member_blocked(self):
        from core.permissions import require_team_lead
        user = make_user(org_id="org-1", org_role="member", team_role="member")
        with pytest.raises(HTTPException) as exc:
            require_team_lead(user)
        assert exc.value.status_code == 403

    def test_dev_passes(self):
        from core.permissions import require_team_lead
        user = make_user(is_dev=True)
        require_team_lead(user)


# ---------------------------------------------------------------------------
# require_same_org
# ---------------------------------------------------------------------------

class TestRequireSameOrg:

    def test_same_org_passes(self):
        from core.permissions import require_same_org
        user = make_user(org_id="org-1", org_role="org_admin")
        require_same_org(user, "org-1")

    def test_different_org_blocked(self):
        from core.permissions import require_same_org
        user = make_user(org_id="org-1", org_role="org_admin")
        with pytest.raises(HTTPException) as exc:
            require_same_org(user, "org-2")
        assert exc.value.status_code == 403

    def test_dev_passes_any_org(self):
        from core.permissions import require_same_org
        user = make_user(is_dev=True)
        require_same_org(user, "any-org-id")


# ---------------------------------------------------------------------------
# can_access_document
# ---------------------------------------------------------------------------

class TestCanAccessDocument:

    def test_owner_can_access_private(self):
        from core.permissions import can_access_document
        user = make_user(user_id="user-1")
        doc  = make_doc(user_id="user-1", visibility="private")
        assert can_access_document(user, doc) is True

    def test_non_owner_cannot_access_private(self):
        from core.permissions import can_access_document
        user = make_user(user_id="user-2")
        doc  = make_doc(user_id="user-1", visibility="private")
        assert can_access_document(user, doc) is False

    def test_team_member_can_access_team_doc(self):
        from core.permissions import can_access_document
        user = make_user(user_id="user-2", org_id="org-1", team_id="team-1")
        doc  = make_doc(user_id="user-1", org_id="org-1", team_id="team-1", visibility="team")
        assert can_access_document(user, doc) is True

    def test_different_team_cannot_access_team_doc(self):
        from core.permissions import can_access_document
        user = make_user(user_id="user-2", org_id="org-1", team_id="team-2")
        doc  = make_doc(user_id="user-1", org_id="org-1", team_id="team-1", visibility="team")
        assert can_access_document(user, doc) is False

    def test_org_admin_with_permission_can_access_team_doc(self):
        from core.permissions import can_access_document
        user = make_user(
            user_id="admin-1",
            org_id="org-1",
            org_role="org_admin",
            can_read_team_documents=True,
        )
        doc = make_doc(user_id="user-1", org_id="org-1", team_id="team-1", visibility="team")
        assert can_access_document(user, doc) is True

    def test_org_admin_without_permission_cannot_access_team_doc(self):
        from core.permissions import can_access_document
        user = make_user(
            user_id="admin-1",
            org_id="org-1",
            org_role="org_admin",
            can_read_team_documents=False,
        )
        doc = make_doc(user_id="user-1", org_id="org-1", team_id="team-1", visibility="team")
        assert can_access_document(user, doc) is False

    def test_org_member_can_access_org_doc(self):
        from core.permissions import can_access_document
        user = make_user(user_id="user-2", org_id="org-1")
        doc  = make_doc(user_id="user-1", org_id="org-1", visibility="org")
        assert can_access_document(user, doc) is True

    def test_different_org_cannot_access_org_doc(self):
        from core.permissions import can_access_document
        user = make_user(user_id="user-2", org_id="org-2")
        doc  = make_doc(user_id="user-1", org_id="org-1", visibility="org")
        assert can_access_document(user, doc) is False

    def test_dev_can_access_anything(self):
        from core.permissions import can_access_document
        user = make_user(is_dev=True)
        doc  = make_doc(user_id="user-1", visibility="private")
        assert can_access_document(user, doc) is True


# ---------------------------------------------------------------------------
# assert_document_access
# ---------------------------------------------------------------------------

class TestAssertDocumentAccess:

    def test_raises_403_when_blocked(self):
        from core.permissions import assert_document_access
        user = make_user(user_id="user-2")
        doc  = make_doc(user_id="user-1", visibility="private")
        with pytest.raises(HTTPException) as exc:
            assert_document_access(user, doc)
        assert exc.value.status_code == 403

    def test_no_raise_when_allowed(self):
        from core.permissions import assert_document_access
        user = make_user(user_id="user-1")
        doc  = make_doc(user_id="user-1", visibility="private")
        assert_document_access(user, doc)  # should not raise


# ---------------------------------------------------------------------------
# assert_org_isolation
# ---------------------------------------------------------------------------

class TestAssertOrgIsolation:

    def test_same_org_passes(self):
        from core.permissions import assert_org_isolation
        user = make_user(org_id="org-1")
        assert_org_isolation(user, "org-1")

    def test_different_org_raises(self):
        from core.permissions import assert_org_isolation
        user = make_user(org_id="org-1")
        with pytest.raises(HTTPException) as exc:
            assert_org_isolation(user, "org-2")
        assert exc.value.status_code == 403

    def test_none_resource_org_passes(self):
        from core.permissions import assert_org_isolation
        user = make_user(org_id="org-1")
        assert_org_isolation(user, None)  # personal resource

    def test_no_org_membership_with_org_resource_raises(self):
        from core.permissions import assert_org_isolation
        user = make_user()  # no org
        with pytest.raises(HTTPException) as exc:
            assert_org_isolation(user, "org-1")
        assert exc.value.status_code == 403

    def test_dev_passes_any(self):
        from core.permissions import assert_org_isolation
        user = make_user(is_dev=True)
        assert_org_isolation(user, "any-org")


# ---------------------------------------------------------------------------
# UserContext computed properties
# ---------------------------------------------------------------------------

class TestUserContextProperties:

    def test_is_org_admin_true(self):
        user = make_user(org_id="org-1", org_role="org_admin")
        assert user.is_org_admin is True

    def test_is_org_admin_false(self):
        user = make_user(org_id="org-1", org_role="member")
        assert user.is_org_admin is False

    def test_is_team_lead_true(self):
        user = make_user(team_id="team-1", team_role="team_lead")
        assert user.is_team_lead is True

    def test_has_org_true(self):
        user = make_user(org_id="org-1")
        assert user.has_org is True

    def test_has_org_false(self):
        user = make_user()
        assert user.has_org is False