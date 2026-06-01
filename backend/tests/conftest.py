"""
tests/conftest.py

Shared pytest fixtures for all test modules.
Creates real fixture files so parsers are tested against actual content.
"""

import os
import csv
import pytest


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def set_test_env():
    """Set required env vars before any test runs."""
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key-for-testing")


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


# ---------------------------------------------------------------------------
# File fixtures — real files on disk
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory):
    """A real CSV file with headers and data rows."""
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