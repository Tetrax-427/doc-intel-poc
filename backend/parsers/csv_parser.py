"""
backend/parsers/csv_parser.py

Handles .csv and .xlsx files via pandas.
Produces structured Table objects alongside text representations.
"""

import os
import uuid
import time
from core.document import Document, DocumentPage, Table, TableCell, make_document
from core.config import Config
from core.logger import get_logger
from core.errors import ParseError
from parsers.base import BaseParser

logger = get_logger("csv")

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}
BATCH_SIZE = 50  # rows per logical page


class CsvParser(BaseParser):

    def get_name(self) -> str:
        return "csv"

    def can_handle(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS

    def is_available(self, config: Config) -> bool:
        return True

    def parse(self, file_path: str, config: Config) -> Document:
        start = time.time()
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            import pandas as pd
            sheets = self._load_sheets(file_path, ext, pd)
        except Exception as e:
            raise ParseError(
                f"Could not read {ext.upper()} file: {e}",
                file_name=file_name,
                retryable=False,
            )

        if not sheets:
            raise ParseError(
                f"{ext.upper()} file is empty",
                file_name=file_name,
                retryable=False,
            )

        pages = []
        all_tables = []
        page_num = 1

        for sheet_name, df in sheets:
            if df.empty:
                continue

            col_names = df.columns.astype(str).tolist()

            # Build a structured Table for the full sheet
            rows_data = df.astype(str).values.tolist()
            cells = [
                TableCell(row=r, col=c, value=str(val), header=col_names[c])
                for r, row in enumerate(rows_data)
                for c, val in enumerate(row)
            ]
            sheet_table = Table(
                page_num=page_num,
                title=sheet_name if sheet_name != "Sheet1" else "",
                headers=col_names,
                rows=rows_data,
                cells=cells,
                raw_text=df.to_string(index=False),
            )
            all_tables.append(sheet_table)

            # Chunk into BATCH_SIZE-row pages for vector storage
            for batch_start in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[batch_start: batch_start + BATCH_SIZE]
                prefix = f"Sheet: {sheet_name}\n" if ext == ".xlsx" else ""
                text = (
                    f"{prefix}Columns: {', '.join(col_names)}\n\n"
                    + batch.to_string(index=False)
                )
                row_label = f"rows {batch_start + 1}–{min(batch_start + BATCH_SIZE, len(df))}"

                pages.append(DocumentPage(
                    page_num=page_num,
                    text=text,
                    tables=[sheet_table] if batch_start == 0 else [],
                    images=[],
                    layout=[],
                    entities=[],
                    word_count=len(text.split()),
                    ocr_confidence=1.0,
                ))
                page_num += 1

        if not pages:
            raise ParseError(
                f"{ext.upper()} produced no content after parsing",
                file_name=file_name,
                retryable=False,
            )

        duration_ms = int((time.time() - start) * 1000)
        logger.info("CSV/XLSX parse complete",
                    file=file_name,
                    sheets=len(sheets),
                    pages=len(pages),
                    duration_ms=duration_ms)

        return make_document(
            id=str(uuid.uuid4()),
            file_name=file_name,
            file_type=ext,
            file_path=file_path,
            pages=pages,
            tables=all_tables,
            parser_used="csv",
            metadata={
                "is_scanned": False,
                "parse_duration_ms": duration_ms,
                "file_size_bytes": os.path.getsize(file_path),
                "sheet_count": len(sheets),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_sheets(self, file_path: str, ext: str, pd) -> list[tuple[str, object]]:
        """Return list of (sheet_name, DataFrame) tuples."""
        if ext == ".csv":
            df = pd.read_csv(file_path)
            return [("Sheet1", df)]
        else:  # .xlsx
            xl = pd.ExcelFile(file_path)
            return [
                (sheet, pd.read_excel(file_path, sheet_name=sheet))
                for sheet in xl.sheet_names
            ]