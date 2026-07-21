"""
results_store.py
-----------------
Stores extraction results as local CSV "tables" under data/tables/.
Each table is one CSV file. Supports creating a new table or appending
to an existing one (columns are unioned; missing values become blank).
"""
import io
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TABLES_DIR = DATA_DIR / "tables"


def _ensure_store():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()]
    safe = "".join(keep).strip("_")
    return safe or "table"


def table_path(name: str) -> Path:
    _ensure_store()
    return TABLES_DIR / f"{_safe_name(name)}.csv"


def list_tables() -> list:
    _ensure_store()
    return sorted(p.stem for p in TABLES_DIR.glob("*.csv"))


def load_table(name: str) -> pd.DataFrame:
    path = table_path(name)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # File exists but has no content/headers - e.g. an interrupted
        # write, or a table that was created with zero rows.
        return pd.DataFrame()


def save_table(name: str, df: pd.DataFrame, mode: str = "new") -> pd.DataFrame:
    """
    mode: "new"    -> overwrite/create table with df
          "append" -> union columns with any existing table and append rows
    Returns the resulting full DataFrame.
    """
    path = table_path(name)
    if mode == "append" and path.exists():
        existing = load_table(name)
        combined = pd.concat([existing, df], ignore_index=True, sort=False)
    else:
        combined = df
    combined.to_csv(path, index=False)
    return combined


def delete_table(name: str):
    path = table_path(name)
    if path.exists():
        path.unlink()


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Extraction Results") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
    return buffer.getvalue()