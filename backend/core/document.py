"""
The central data model for DocIntel.
Every parser returns a Document. Every downstream component consumes a Document.
This is the shared contract between all parts of the system.

  - BoundingBox dataclass
  - LayoutElement.bbox, LayoutElement.state 
  - ImageElement.bbox, ImageElement.caption, ImageElement.element_type 
  - DocumentPage.reading_order_bboxes 
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    Normalized bounding box for a document element.

    Coordinates are normalized to [0.0, 1.0] relative to page dimensions
    so they remain valid regardless of rendering resolution or zoom level.

    x, y       — top-left corner (origin is top-left of page)
    width      — box width as fraction of page width
    height     — box height as fraction of page height
    page_num   — 1-indexed page number
    page_width / page_height — original page dimensions in points
    """
    x: float
    y: float
    width: float
    height: float
    page_num: int
    page_width: float
    page_height: float

    def to_dict(self) -> dict:
        return {
            "x":           round(self.x, 4),
            "y":           round(self.y, 4),
            "width":       round(self.width, 4),
            "height":      round(self.height, 4),
            "page":        self.page_num,
            "page_width":  self.page_width,
            "page_height": self.page_height,
        }

    @classmethod
    def from_dict(cls, d) -> "Optional[BoundingBox]":
        """Reconstruct from a stored dict. Returns None if d is None."""
        if d is None:
            return None
        return cls(
            x=d["x"],
            y=d["y"],
            width=d["width"],
            height=d["height"],
            page_num=d["page"],
            page_width=d["page_width"],
            page_height=d["page_height"],
        )


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A named entity extracted from the document."""
    text: str
    entity_type: str
    confidence: float
    page_num: int
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "entity_type": self.entity_type,
            "confidence":  self.confidence,
            "page_num":    self.page_num,
            "char_start":  self.char_start,
            "char_end":    self.char_end,
        }


@dataclass
class TableCell:
    """A single cell in a table."""
    row: int
    col: int
    value: str
    header: str

    def to_dict(self) -> dict:
        return {
            "row":    self.row,
            "col":    self.col,
            "value":  self.value,
            "header": self.header,
        }


@dataclass
class Table:
    """A structured table extracted from the document."""
    page_num: int
    title: str
    headers: list[str]
    rows: list[list]
    cells: list[TableCell]
    raw_text: str

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "title":    self.title,
            "headers":  self.headers,
            "rows":     self.rows,
            "cells":    [c.to_dict() for c in self.cells],
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
    """
    A structural element detected in the document layout.

   additions:
      bbox   — normalized bounding box; None for non-Docling parsers
      state  — element state:
                   checkbox  -> 'checked' | 'unchecked'
                   signature -> 'signed'  | 'unsigned' | 'unknown'
                   Other types leave state as None.
    """
    element_type: str
    page_num: int
    text: str
    confidence: float
    bbox: Optional["BoundingBox"] = None
    state: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "element_type": self.element_type,
            "page_num":     self.page_num,
            "text":         self.text,
            "confidence":   self.confidence,
            "bbox":         self.bbox.to_dict() if self.bbox else None,
            "state":        self.state,
        }


@dataclass
class ImageElement:
    """
    An image or visual element within the document.

    additions :
      bbox         — normalized bounding box of the figure on the page
      caption      — caption text if Docling detected one (may be "")
      element_type — "figure" | "photo" | "diagram"; defaults to "figure"

    """
    page_num: int
    image_ref: str
    ocr_text: str
    description: str
    chunk_type: str
    vision_prompt_used: str
    bbox: Optional["BoundingBox"] = None
    caption: str = ""
    element_type: str = "figure"

    def to_dict(self) -> dict:
        return {
            "page_num":           self.page_num,
            "image_ref":          self.image_ref,
            "ocr_text":           self.ocr_text,
            "description":        self.description,
            "chunk_type":         self.chunk_type,
            "vision_prompt_used": self.vision_prompt_used,
            "bbox":               self.bbox.to_dict() if self.bbox else None,
            "caption":            self.caption,
            "element_type":       self.element_type,
        }


@dataclass
class DocumentPage:
    """
    A single page of the document with all its extracted content.

    Same index = same segment. Empty list = no bbox data.
    Consumers must check len(reading_order_bboxes) == len(reading_order)
    before indexing — may be shorter if bbox extraction partially failed.
    """
    page_num: int
    text: str
    tables: list
    images: list
    layout: list
    entities: list
    word_count: int
    ocr_confidence: float
    reading_order: list = field(default_factory=list)
    reading_order_bboxes: list = field(default_factory=list)
    position_type: str = "page"
    
    
    def to_dict(self) -> dict:
        return {
            "page_num":             self.page_num,
            "text":                 self.text,
            "word_count":           self.word_count,
            "ocr_confidence":       self.ocr_confidence,
            "reading_order":        self.reading_order,
            "reading_order_bboxes": [
                b.to_dict() if b else None
                for b in self.reading_order_bboxes
            ],
            "tables":   [t.to_dict() for t in self.tables],
            "images":   [i.to_dict() for i in self.images],
            "layout":   [l.to_dict() for l in self.layout],
            "entities": [e.to_dict() for e in self.entities],
        }


@dataclass
class Classification:
    """The result of classifying a document's type."""
    doc_type: str
    confidence: float
    sub_types: list
    schema_template: str
    validation_ruleset: str
    vision_prompt: str
    requires_human_review: bool

    def to_dict(self) -> dict:
        return {
            "doc_type":              self.doc_type,
            "confidence":            self.confidence,
            "sub_types":             self.sub_types,
            "schema_template":       self.schema_template,
            "validation_ruleset":    self.validation_ruleset,
            "vision_prompt":         self.vision_prompt,
            "requires_human_review": self.requires_human_review,
        }


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """
    The central data model for DocIntel.
    PUBLIC INTERFACE — do not change field names or types without team discussion.
    """
    id: str
    file_name: str
    file_type: str
    file_path: str
    pages: list[DocumentPage]
    full_text: str
    tables: list[Table]
    entities: dict
    metadata: dict
    classifications: list[Classification]
    summary: str
    version: int
    parent_id: Optional[str]
    created_at: str
    parser_used: str
    vision_used: bool

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "file_name":       self.file_name,
            "file_type":       self.file_type,
            "file_path":       self.file_path,
            "full_text":       self.full_text,
            "page_count":      self.page_count,
            "table_count":     len(self.tables),
            "entity_count":    sum(len(v) for v in self.entities.values()),
            "parser_used":     self.parser_used,
            "vision_used":     self.vision_used,
            "version":         self.version,
            "parent_id":       self.parent_id,
            "created_at":      self.created_at,
            "summary":         self.summary,
            "metadata":        self.metadata,
            "classifications": [c.to_dict() for c in self.classifications],
            "pages":           [p.to_dict() for p in self.pages],
            "tables":          [t.to_dict() for t in self.tables],
            "entities": {
                k: [e.to_dict() for e in v]
                for k, v in self.entities.items()
            },
        }

    @property
    def primary_classification(self) -> Optional[Classification]:
        return self.classifications[0] if self.classifications else None

    @property
    def is_scanned(self) -> bool:
        return self.metadata.get("is_scanned", False)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def has_entities(self) -> bool:
        return any(len(v) > 0 for v in self.entities.values())


# ---------------------------------------------------------------------------
# Factory helper
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
    """Factory for creating a Document from parser output."""
    all_tables = tables if tables is not None else [t for p in pages for t in p.tables]
    full_text  = "\n\n".join(p.text for p in pages if p.text.strip())

    if entities is None:
        aggregated: dict = {}
        for p in pages:
            for ent in p.entities:
                aggregated.setdefault(ent.entity_type, []).append(ent)
        entities = aggregated

    base_metadata = {
        "page_count":  len(pages),
        "parser_used": parser_used,
        "is_scanned":  False,
        "language":    "en",
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