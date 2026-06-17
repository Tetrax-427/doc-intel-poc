"""
DoclingParser — local parser using Docling (IBM / Linux Foundation).

Group B additions:
  - BoundingBox extracted per reading_order segment (B1)
  - reading_order_bboxes populated on each DocumentPage (B1)
  - Checkboxes and signature regions detected via layout_elements (B2)
  - Figures detected and attached as ImageElement with chunk_type='figure' (B3)
"""

import os
import uuid
import time

from parsers.base import BaseParser
from parsers.layout_elements import (
    extract_checkboxes,
    extract_signature_regions,
    extract_figures,
)
from core.document import (
    Document, DocumentPage, Table, TableCell,
    LayoutElement, BoundingBox, make_document,
)
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError

logger = get_logger("docling_parser")


class DoclingParser(BaseParser):
    """
    Local parser using Docling.
    No API key required. Handles PDFs, DOCX, PPTX, images.
    Skipped gracefully by AutoRouter if docling is not installed.
    """

    def get_name(self) -> str:
        return "docling"

    def can_handle(self, file_path: str) -> bool:
        supported = {".pdf", ".docx", ".pptx", ".xlsx",
                     ".html", ".png", ".jpg", ".jpeg", ".tiff"}
        return os.path.splitext(file_path)[1].lower() in supported

    def is_available(self, config: Config) -> bool:
        try:
            import docling  # noqa: F401
            return True
        except ImportError:
            logger.warning("Docling not installed — parser unavailable. Run: pip install docling")
            return False

    def parse(self, file_path: str, config: Config) -> Document:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        start = time.time()
        file_name = os.path.basename(file_path)
        logger.info("Starting Docling parse", file=file_name)

        try:
            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.do_ocr = True
            pipeline_opts.do_table_structure = True

            converter = DocumentConverter()
            result = converter.convert(file_path)
            docling_doc = result.document

        except Exception as e:
            raise ParseError(
                f"Docling failed to parse {file_name}: {e}",
                file_name=file_name,
                retryable=True,
            )

        page_count = len(result.pages) if hasattr(result, "pages") and result.pages else 1

        pages = []
        all_tables = []

        for page_num in range(1, page_count + 1):
            # Resolve page size for bbox normalization
            page_size = self._get_page_size(result, page_num)

            page_md = self._export_page_markdown(docling_doc, page_num)

            # --- B1: reading_order + reading_order_bboxes ---
            reading_order_segments, reading_order_bboxes = (
                self._extract_reading_order_with_bboxes(docling_doc, page_num, page_size)
            )

            # --- Tables ---
            page_tables = self._extract_page_tables(docling_doc, page_num)
            all_tables.extend(page_tables)

            # --- B2: checkboxes + signatures ---
            checkboxes = extract_checkboxes(docling_doc, page_num, page_size)
            signatures = extract_signature_regions(
                docling_doc, page_num, page_size, page_md
            )

            # --- B1: layout furniture (headers/footers) ---
            furniture = self._extract_layout_elements(docling_doc, page_num, page_size)

            layout_elements = furniture + checkboxes + signatures

            # --- B3: figures ---
            figures = extract_figures(docling_doc, result, page_num, page_size)

            pages.append(DocumentPage(
                page_num=page_num,
                text=page_md,
                tables=page_tables,
                images=figures,
                layout=layout_elements,
                entities=[],
                word_count=len(page_md.split()),
                ocr_confidence=0.95,
                reading_order=reading_order_segments,
                reading_order_bboxes=reading_order_bboxes,
            ))

        duration_ms = int((time.time() - start) * 1000)
        is_scanned = (
            all(p.word_count < config.vision_min_words for p in pages)
            if pages else False
        )

        # Summarize Group B outputs for metadata
        total_checkboxes = sum(
            1 for p in pages for l in p.layout if l.element_type == "checkbox"
        )
        total_signatures = sum(
            1 for p in pages for l in p.layout if l.element_type == "signature"
        )
        total_figures = sum(len(p.images) for p in pages)

        logger.info(
            "Docling parse complete",
            file=file_name,
            pages=len(pages),
            tables=len(all_tables),
            checkboxes=total_checkboxes,
            signatures=total_signatures,
            figures=total_figures,
            duration_ms=duration_ms,
        )

        return make_document(
            id=str(uuid.uuid4()),
            file_name=file_name,
            file_type=os.path.splitext(file_path)[1].lower(),
            file_path=file_path,
            pages=pages,
            tables=all_tables,
            parser_used="docling",
            vision_used=False,
            metadata={
                "page_count":        len(pages),
                "parser_used":       "docling",
                "parse_duration_ms": duration_ms,
                "is_scanned":        is_scanned,
                "file_size_bytes":   os.path.getsize(file_path),
                "has_reading_order": True,
                "has_bboxes":        True,
                "checkbox_count":    total_checkboxes,
                "signature_count":   total_signatures,
                "figure_count":      total_figures,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_page_size(self, result, page_num: int):
        """
        Return the Docling page size object for a given page.
        Returns None if unavailable — callers must handle None gracefully.
        """
        try:
            if hasattr(result, "pages") and result.pages:
                page = result.pages[page_num - 1]
                return getattr(page, "size", None)
        except (IndexError, AttributeError):
            pass
        return None

    def _export_page_markdown(self, docling_doc, page_num: int) -> str:
        """Export a single page as markdown. Falls back silently."""
        try:
            return docling_doc.export_to_markdown(page_no=page_num)
        except TypeError:
            logger.debug("export_to_markdown page_no not supported — exporting full doc")
            return docling_doc.export_to_markdown()
        except Exception as e:
            logger.warning("Page markdown export failed", page=page_num, error=str(e))
            return ""

    def _extract_reading_order_with_bboxes(
        self,
        docling_doc,
        page_num: int,
        page_size,
    ) -> tuple[list[str], list]:
        """
        B1 — Extract reading order segments and their bounding boxes.

        Strategy: walk docling_doc.texts in document order (Docling guarantees
        reading order). For each element on this page, emit the text as a
        segment and attempt to extract a BoundingBox.

        Falls back to markdown-split approach (Group A behaviour) if
        docling_doc.texts is unavailable.

        Returns:
            (segments, bboxes) — parallel lists of same length.
            bboxes entries may be None if extraction failed for that segment.
        """
        segments = []
        bboxes = []

        if not hasattr(docling_doc, "texts"):
            # Fallback: split page markdown on double newlines (Group A behaviour)
            page_md = self._export_page_markdown(docling_doc, page_num)
            segments = [s.strip() for s in page_md.split("\n\n") if s.strip()]
            bboxes = [None] * len(segments)
            return segments, bboxes

        for element in docling_doc.texts:
            try:
                if not element.prov or element.prov[0].page_no != page_num:
                    continue

                text = getattr(element, "text", "") or ""
                if not text.strip():
                    continue

                segments.append(text)

                bbox = None
                if page_size:
                    try:
                        raw = element.prov[0].bbox
                        w = page_size.width
                        h = page_size.height
                        if w and h:
                            bbox = BoundingBox(
                                x=raw.l / w,
                                y=raw.t / h,
                                width=(raw.r - raw.l) / w,
                                height=(raw.b - raw.t) / h,
                                page_num=page_num,
                                page_width=w,
                                page_height=h,
                            )
                    except Exception:
                        pass  # bbox stays None for this segment

                bboxes.append(bbox)

            except Exception as e:
                logger.warning("Reading order element failed",
                               page=page_num, error=str(e))

        # If texts walk yielded nothing, fall back to markdown split
        if not segments:
            page_md = self._export_page_markdown(docling_doc, page_num)
            segments = [s.strip() for s in page_md.split("\n\n") if s.strip()]
            bboxes = [None] * len(segments)

        return segments, bboxes

    def _extract_page_tables(self, docling_doc, page_num: int) -> list[Table]:
        """Extract structured tables for a given page."""
        page_tables = []

        if not hasattr(docling_doc, "tables"):
            return page_tables

        for table in docling_doc.tables:
            table_page = 1
            if hasattr(table, "prov") and table.prov:
                try:
                    table_page = table.prov[0].page_no
                except (AttributeError, IndexError):
                    pass

            if table_page != page_num:
                continue

            try:
                rows = []
                for row in table.data:
                    rows.append([cell.text.strip() for cell in row])

                if not rows:
                    continue

                headers = rows[0]
                data_rows = rows[1:] if len(rows) > 1 else []

                cells = [
                    TableCell(
                        row=r_idx,
                        col=c_idx,
                        value=cell,
                        header=headers[c_idx] if c_idx < len(headers) else "",
                    )
                    for r_idx, row in enumerate(data_rows)
                    for c_idx, cell in enumerate(row)
                ]

                page_tables.append(Table(
                    page_num=page_num,
                    title="",
                    headers=headers,
                    rows=data_rows,
                    cells=cells,
                    raw_text="\n".join(["|".join(r) for r in rows]),
                ))

            except Exception as e:
                logger.warning("Table extraction failed",
                               page=page_num, error=str(e))

        return page_tables

    def _extract_layout_elements(
        self,
        docling_doc,
        page_num: int,
        page_size,
    ) -> list[LayoutElement]:
        """Extract furniture elements (headers, footers, etc.) with bboxes."""
        layout_elements = []

        if not hasattr(docling_doc, "furniture"):
            return layout_elements

        for elem in docling_doc.furniture:
            try:
                elem_page = 1
                if hasattr(elem, "prov") and elem.prov:
                    elem_page = elem.prov[0].page_no

                if elem_page != page_num:
                    continue

                from parsers.layout_elements import _extract_bbox
                bbox = _extract_bbox(elem.prov[0], page_num, page_size) if elem.prov else None

                layout_elements.append(LayoutElement(
                    element_type=str(elem.label).lower(),
                    page_num=page_num,
                    text=elem.text if hasattr(elem, "text") else "",
                    confidence=0.9,
                    bbox=bbox,
                    state=None,
                ))
            except Exception:
                pass

        return layout_elements