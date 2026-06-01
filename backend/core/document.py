"""
backend/core/document.py

The central data model for DocIntel.
Every parser returns a Document. Every downstream component consumes a Document.
This is the shared contract between all parts of the system.

COMMIT EARLY — do not change the public interface after committing
without team discussion.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity extracted from the document."""
    text: str
    entity_type: str        # "person", "org", "date", "amount", "id", "location"
    confidence: float       # 0.0 - 1.0
    page_num: int
    char_start: int         # character offset in page text — for exact evidence
    char_end: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "page_num": self.page_num,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


@dataclass
class TableCell:
    """A single cell in a table."""
    row: int
    col: int
    value: str
    header: str             # column header this cell belongs to

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "header": self.header,
        }


@dataclass
class Table:
    """A structured table extracted from the document."""
    page_num: int
    title: str                      # descriptive title if detectable
    headers: list[str]
    rows: list[list[str]]           # raw row data
    cells: list[TableCell]          # typed cell data for precise extraction
    raw_text: str                   # original text representation (fallback)

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "title": self.title,
            "headers": self.headers,
            "rows": self.rows,
            "cells": [c.to_dict() for c in self.cells],
            "raw_text": self.raw_text,
        }

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return len(self.headers)


@dataclass
class LayoutElement:
    """A structural element detected in the document layout."""
    element_type: str       # "header", "footer", "signature", "stamp", "logo", "watermark"
    page_num: int
    text: str               # text content if any
    confidence: float

    def to_dict(self) -> dict:
        return {
            "element_type": self.element_type,
            "page_num": self.page_num,
            "text": self.text,
            "confidence": self.confidence,
        }


@dataclass
class ImageElement:
    """An image or visual element within the document."""
    page_num: int
    image_ref: str              # filename or "page_N_image_M"
    ocr_text: str               # text extracted via OCR
    description: str            # natural language description from vision model
                                # empty string if no vision model configured
    chunk_type: str             # "text" or "description" — how it's stored in vector DB
    vision_prompt_used: str     # which vision prompt template was used

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "image_ref": self.image_ref,
            "ocr_text": self.ocr_text,
            "description": self.description,
            "chunk_type": self.chunk_type,
            "vision_prompt_used": self.vision_prompt_used,
        }


@dataclass
class DocumentPage:
    """A single page of the document with all its extracted content."""
    page_num: int
    text: str                       # full extracted text
    tables: list[Table]             # structured tables on this page
    images: list[ImageElement]      # images/visual elements on this page
    layout: list[LayoutElement]     # headers, footers, signatures etc.
    entities: list[Entity]          # named entities on this page
    word_count: int
    ocr_confidence: float           # 1.0 for text PDFs, lower for scanned

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "text": self.text,
            "word_count": self.word_count,
            "ocr_confidence": self.ocr_confidence,
            "tables": [t.to_dict() for t in self.tables],
            "images": [i.to_dict() for i in self.images],
            "layout": [l.to_dict() for l in self.layout],
            "entities": [e.to_dict() for e in self.entities],
        }


@dataclass
class Classification:
    """The result of classifying a document's type."""
    doc_type: str               # "cv_resume", "invoice", "gst_return", "contract" etc.
    confidence: float           # 0.0 - 1.0
    sub_types: list[str]        # additional types if document is multi-type
    schema_template: str        # which extraction template to auto-select
    validation_ruleset: str     # which validation rules to apply
    vision_prompt: str          # which vision prompt to use
    requires_human_review: bool # True if confidence < threshold

    def to_dict(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "confidence": self.confidence,
            "sub_types": self.sub_types,
            "schema_template": self.schema_template,
            "validation_ruleset": self.validation_ruleset,
            "vision_prompt": self.vision_prompt,
            "requires_human_review": self.requires_human_review,
        }


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    The central data model for DocIntel.
    Every parser returns this. Every downstream component consumes this.
    This is the contract between all parts of the system.

    PUBLIC INTERFACE — do not change field names or types after committing
    without team discussion. Add new optional fields only.
    """

    # Identity
    id: str                             # UUID assigned at creation
    file_name: str
    file_type: str                      # ".pdf", ".docx", ".csv" etc.
    file_path: str                      # path to original file on disk

    # Content
    pages: list[DocumentPage]
    full_text: str                      # all page text concatenated
    tables: list[Table]                 # all tables across all pages (flattened)
    entities: dict                      # aggregated: {"person": [Entity], "amount": [Entity], ...}

    # Metadata
    metadata: dict                      # page_count, file_size_bytes, parser_used,
                                        # parse_duration_ms, is_scanned, language
    classifications: list[Classification]   # ordered by confidence, highest first

    # State
    summary: str                        # auto-generated on upload; empty string until set
    version: int                        # 1 for new docs, increments on re-upload
    parent_id: Optional[str]            # UUID of previous version if this is a re-upload
    created_at: str                     # ISO 8601 timestamp
    parser_used: str                    # "llamaparse", "pypdf", "docx", "csv", "url", "text"
    vision_used: bool                   # True if vision model ran during ingestion

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_path": self.file_path,
            "full_text": self.full_text,
            "page_count": self.page_count,
            "table_count": len(self.tables),
            "entity_count": sum(len(v) for v in self.entities.values()),
            "parser_used": self.parser_used,
            "vision_used": self.vision_used,
            "version": self.version,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "summary": self.summary,
            "metadata": self.metadata,
            "classifications": [c.to_dict() for c in self.classifications],
            "pages": [p.to_dict() for p in self.pages],
            "tables": [t.to_dict() for t in self.tables],
            "entities": {
                k: [e.to_dict() for e in v]
                for k, v in self.entities.items()
            },
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def primary_classification(self) -> Optional[Classification]:
        """Returns the highest-confidence classification, or None."""
        return self.classifications[0] if self.classifications else None

    @property
    def is_scanned(self) -> bool:
        """True if the document was identified as a scanned (image-based) PDF."""
        return self.metadata.get("is_scanned", False)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        """Total word count across all pages."""
        return sum(p.word_count for p in self.pages)

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def has_entities(self) -> bool:
        return any(len(v) > 0 for v in self.entities.values())


# ---------------------------------------------------------------------------
# Factory helper — used by parsers to build a Document cleanly
# ---------------------------------------------------------------------------

def make_document(
    *,
    id: str,
    file_name: str,
    file_type: str,
    file_path: str,
    pages: list[DocumentPage],
    tables: list[Table] = None,
    entities: dict = None,
    metadata: dict = None,
    parser_used: str,
    vision_used: bool = False,
    classifications: list[Classification] = None,
    summary: str = "",
    version: int = 1,
    parent_id: Optional[str] = None,
) -> Document:
    """
    Factory for creating a Document from parser output.
    Handles defaults so parsers don't have to repeat boilerplate.

    Usage:
        return make_document(
            id=str(uuid.uuid4()),
            file_name=os.path.basename(file_path),
            file_type=".pdf",
            file_path=file_path,
            pages=pages,
            parser_used="llamaparse",
            metadata={"page_count": len(pages), "is_scanned": False}
        )
    """
    all_tables = tables if tables is not None else [t for p in pages for t in p.tables]
    full_text = "\n\n".join(p.text for p in pages if p.text.strip())

    # Aggregate page-level entities into the top-level entities dict
    # if caller didn't supply one explicitly
    if entities is None:
        aggregated: dict = {}
        for p in pages:
            for ent in p.entities:
                aggregated.setdefault(ent.entity_type, []).append(ent)
        entities = aggregated

    base_metadata = {
        "page_count": len(pages),
        "parser_used": parser_used,
        "is_scanned": False,
        "language": "en",
    }
    if metadata:
        base_metadata.update(metadata)

    return Document(
        id=id,
        file_name=file_name,
        file_type=file_type,
        file_path=file_path,
        pages=pages,
        full_text=full_text,
        tables=all_tables,
        entities=entities or {},
        metadata=base_metadata,
        classifications=classifications or [],
        summary=summary,
        version=version,
        parent_id=parent_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        parser_used=parser_used,
        vision_used=vision_used,
    )