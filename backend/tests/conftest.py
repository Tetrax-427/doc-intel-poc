"""
tests/conftest.py
Shared pytest fixtures for all test modules.

Changes in this phase:
  - Added sample_pdf_bytes, sample_png_bytes for file validation tests
  - Added app_config fixture for quota config override tests
  - Added mock_user_context, mock_org_admin_context fixtures
  - Existing fixtures unchanged
"""

import os
import csv
import struct
import pytest


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def set_test_env():
    """Set required env vars before any test runs."""
    os.environ.setdefault("SUPABASE_URL",         "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY",         "test-key-for-testing")
    os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key-for-testing")
    os.environ.setdefault("DEVELOPER_API_KEY",    "test-developer-key-for-testing")


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cfg():
    """Return a loaded Config instance."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.config import load_config
    return load_config()


@pytest.fixture
def app_config():
    """
    Return the live config proxy.
    Use this in tests that need to read config values (rate limits, quota defaults etc.)
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.config import config
    return config


# ---------------------------------------------------------------------------
# UserContext fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_user_context():
    """A regular authenticated user with no org membership."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.auth import UserContext
    return UserContext(
        user_id="test-user-001",
        email="user@test.com",
        is_dev=False,
        org_id=None,
        org_role=None,
        team_id=None,
        team_role=None,
    )


@pytest.fixture
def mock_org_admin_context():
    """An org admin user."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.auth import UserContext
    return UserContext(
        user_id="test-admin-001",
        email="admin@test.com",
        is_dev=False,
        org_id="test-org-uuid-001",
        org_role="org_admin",
        team_id=None,
        team_role=None,
        can_read_team_documents=True,
        can_read_all_usage=True,
    )


@pytest.fixture
def mock_team_lead_context():
    """A team lead user."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.auth import UserContext
    return UserContext(
        user_id="test-lead-001",
        email="lead@test.com",
        is_dev=False,
        org_id="test-org-uuid-001",
        org_role="member",
        team_id="test-team-uuid-001",
        team_role="team_lead",
    )


@pytest.fixture
def mock_dev_context():
    """Dev mode user — has all permissions."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
    from core.auth import UserContext
    return UserContext(
        user_id="dev_user",
        email="dev@local",
        is_dev=True,
        org_role="org_admin",
        can_read_team_documents=True,
        can_read_all_usage=True,
    )


# ---------------------------------------------------------------------------
# File byte fixtures — for magic byte validation tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_pdf_bytes():
    """Minimal valid PDF bytes (magic bytes only — not a real document)."""
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"


@pytest.fixture(scope="session")
def sample_png_bytes():
    """Minimal valid PNG bytes (8-byte PNG signature + IHDR chunk)."""
    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"
    # Minimal IHDR chunk (width=1, height=1, bit depth=8, color type=2 RGB)
    ihdr_data  = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc   = struct.pack(">I", 0x902753DE)  # precomputed CRC for this IHDR
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc
    return signature + ihdr_chunk


@pytest.fixture(scope="session")
def sample_docx_bytes():
    """Minimal DOCX magic bytes (ZIP PK header)."""
    return b"PK\x03\x04" + b"\x00" * 26


@pytest.fixture(scope="session")
def fake_pdf_bytes():
    """
    A file with .pdf extension but wrong magic bytes (actually a text file).
    Used to test magic byte mismatch detection.
    """
    return b"This is not a PDF file, just plain text."


@pytest.fixture(scope="session")
def sample_pdf_file(tmp_path_factory, sample_pdf_bytes):
    """Write sample PDF bytes to a temp file, return path."""
    path = tmp_path_factory.mktemp("files") / "test.pdf"
    path.write_bytes(sample_pdf_bytes)
    return str(path)


@pytest.fixture(scope="session")
def fake_pdf_file(tmp_path_factory, fake_pdf_bytes):
    """Write fake PDF bytes to a temp file named .pdf, return path."""
    path = tmp_path_factory.mktemp("files") / "fake.pdf"
    path.write_bytes(fake_pdf_bytes)
    return str(path)


@pytest.fixture(scope="session")
def oversized_file(tmp_path_factory):
    """Create a file larger than the default 50MB limit for size tests."""
    path = tmp_path_factory.mktemp("files") / "big.txt"
    # Write 51MB of data
    path.write_bytes(b"x" * (51 * 1024 * 1024))
    return str(path)


@pytest.fixture(scope="session")
def empty_file(tmp_path_factory):
    """Empty file — should fail validation."""
    path = tmp_path_factory.mktemp("files") / "empty.pdf"
    path.write_bytes(b"")
    return str(path)


# ---------------------------------------------------------------------------
# Existing file fixtures 
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixtures") / "sample_table.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Amount", "Date", "Status"])
        writer.writerow(["Widget A", "100.00", "2024-01-01", "Paid"])
        writer.writerow(["Widget B", "250.50", "2024-01-02", "Pending"])
        writer.writerow(["Widget C", "175.00", "2024-01-03", "Paid"])
    return str(path)


@pytest.fixture(scope="session")
def sample_txt(tmp_path_factory):
    """A plain text document."""
    path = tmp_path_factory.mktemp("fixtures") / "sample_doc.txt"
    path.write_text(
        "Sample Document\n\n"
        "This is a test document with multiple paragraphs.\n\n"
        "It contains names like John Smith and organisations like Acme Corp.\n\n"
        "The total amount is $500.00 due on 2024-03-15.\n"
    )
    return str(path)


@pytest.fixture(scope="session")
def sample_md(tmp_path_factory):
    """A markdown document."""
    path = tmp_path_factory.mktemp("fixtures") / "sample_doc.md"
    path.write_text(
        "# Sample Markdown Document\n\n"
        "## Introduction\n\n"
        "This document contains **bold** and *italic* text.\n\n"
        "## Data Table\n\n"
        "| Item | Price |\n"
        "|------|-------|\n"
        "| Widget | $10 |\n"
        "| Gadget | $20 |\n\n"
        "## Conclusion\n\n"
        "End of document.\n"
    )
    return str(path)


@pytest.fixture(scope="session")
def sample_empty_txt(tmp_path_factory):
    """An empty text file — parsers should raise ParseError."""
    path = tmp_path_factory.mktemp("fixtures") / "empty.txt"
    path.write_text("")
    return str(path)


@pytest.fixture(scope="session")
def sample_empty_csv(tmp_path_factory):
    """An empty CSV file — parsers should raise ParseError."""
    path = tmp_path_factory.mktemp("fixtures") / "empty.csv"
    path.write_text("")
    return str(path)


@pytest.fixture(scope="session")
def unknown_extension_file(tmp_path_factory):
    """A file with unsupported extension."""
    path = tmp_path_factory.mktemp("fixtures") / "unknown.xyz"
    path.write_text("some content")
    return str(path)