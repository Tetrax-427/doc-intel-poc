"""
DocIntel Extraction Helper
---------------------------
A standalone Streamlit app that talks to an existing DocIntel FastAPI
backend to run repeatable, schema-based extraction over batches of
documents and collect the results into downloadable tables.

Auth: interactive sign-in only (POST /auth/login). Tokens live in
st.session_state for the duration of the browser session - nothing is
persisted to disk, and there is no env-var token fallback.

Run with:
    streamlit run app.py
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from api_client import ApiError, DocIntelClient, extract_value
from batch_runner import run_batch
from config_store import get_base_url, get_timeout, get_max_parallel_uploads
from results_store import list_tables, load_table, save_table, delete_table, to_excel_bytes
from schema_store import (
    list_schema_names, get_schema, save_schema, delete_schema,
    normalize_field, flatten_fields_for_table,
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".xlsx", ".txt", ".md", ".jpg", ".jpeg", ".png"}
FIELD_TYPES = ["string", "integer", "date", "list"]
EMPTY_FIELD = {"name": "", "type": "string", "description": "", "properties": None}

st.set_page_config(page_title="DocIntel Extraction Helper", layout="wide")

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key, default in [
    ("access_token", None),
    ("refresh_token", None),
    ("user_email", None),
    ("manual_fields", [dict(EMPTY_FIELD)]),
    ("last_run_results", None),       # list of {"filename": ..., "extracted": {...}} - nested, pre-flatten
    ("last_run_schema_name", None),
    ("nl_previewed_instruction", None),
    ("nl_preview_fields", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_client() -> DocIntelClient:
    return DocIntelClient(get_base_url(), access_token=st.session_state.access_token or "", timeout=get_timeout())


def _clear_session():
    for k in ("access_token", "refresh_token", "user_email"):
        st.session_state[k] = None


def ensure_valid_session() -> bool:
    """Confirms the current token still works; tries one silent refresh if not."""
    client = get_client()
    try:
        client.me()
        return True
    except ApiError as e:
        if e.status_code == 401 and st.session_state.refresh_token:
            try:
                refreshed = client.refresh(st.session_state.refresh_token)
                st.session_state.access_token = refreshed["access_token"]
                st.session_state.refresh_token = refreshed.get("refresh_token", st.session_state.refresh_token)
                return True
            except ApiError:
                pass
        _clear_session()
        return False


def render_login():
    st.title("📄 DocIntel Extraction Helper")
    st.subheader("Sign in")
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if not email.strip() or not password:
            st.error("Enter both email and password.")
        else:
            try:
                client = DocIntelClient(get_base_url(), timeout=get_timeout())
                result = client.login(email.strip(), password)
                st.session_state.access_token = result["access_token"]
                st.session_state.refresh_token = result.get("refresh_token")
                st.session_state.user_email = result.get("user", {}).get("email", email.strip())
                st.rerun()
            except ApiError as e:
                st.error(f"Sign in failed: {e}")


# ----------------------------------------------------------------------
# Recursive field editor - used by both "Manual fields" and the NL preview,
# so a list field can be expanded into its own nested sub-field rows.
# ----------------------------------------------------------------------
def render_field_editor(fields: list, key_prefix: str, show_include: bool = False, depth: int = 0):
    """
    Renders one row per field (name / type / description / delete), and for
    any field with type == "list", a nested, indented editor for its
    "properties" sub-fields. Mutates `fields` in place.
    """
    indent = "　" * depth  # ideographic space, gives a visible indent in Streamlit text
    to_delete = None
    for i, f in enumerate(fields):
        f.setdefault("type", "string")
        f.setdefault("properties", None)
        cols = st.columns([0.4, 1.6, 1, 3, 0.4]) if show_include else st.columns([1.6, 1, 3, 0.4])
        c = iter(cols)
        if show_include:
            f["include"] = next(c).checkbox("", value=f.get("include", True), key=f"{key_prefix}_inc_{i}")
        f["name"] = next(c).text_input(f"{indent}Field name", value=f["name"], key=f"{key_prefix}_name_{i}")
        f["type"] = next(c).selectbox(
            "Type", FIELD_TYPES, index=FIELD_TYPES.index(f["type"]) if f["type"] in FIELD_TYPES else 0,
            key=f"{key_prefix}_type_{i}",
        )
        f["description"] = next(c).text_input("Description", value=f["description"], key=f"{key_prefix}_desc_{i}")
        if next(c).button("✕", key=f"{key_prefix}_del_{i}"):
            to_delete = i

        if f["type"] == "list":
            st.caption(f"{indent}↳ sub-fields of **{f['name'] or '(unnamed list)'}**")
            if not f.get("properties"):
                f["properties"] = [dict(EMPTY_FIELD)]
            render_field_editor(f["properties"], f"{key_prefix}_sub_{i}", show_include=False, depth=depth + 1)
            if st.button(f"{indent}+ Add sub-field", key=f"{key_prefix}_addsub_{i}"):
                f["properties"].append(dict(EMPTY_FIELD))
                st.rerun()
        else:
            f["properties"] = None

    if to_delete is not None and len(fields) > 1:
        fields.pop(to_delete)
        st.rerun()


def clean_fields_for_save(fields: list, require_include: bool = False) -> list:
    """Drops unnamed rows and (for NL preview) unchecked ones, recursively."""
    out = []
    for f in fields:
        if not f["name"].strip():
            continue
        if require_include and not f.get("include", True):
            continue
        cleaned = normalize_field(f)
        if cleaned["type"] == "list" and f.get("properties"):
            cleaned["properties"] = clean_fields_for_save(f["properties"], require_include=False)
        out.append(cleaned)
    return out


# ----------------------------------------------------------------------
# Nested results rendering - one document per expander: scalar fields as
# a key/value summary, each list field as its own sub-table underneath.
# ----------------------------------------------------------------------
def render_nested_result(filename: str, extracted: dict, error: str | None = None):
    with st.expander(f"📄 {filename}", expanded=False):
        if error:
            st.error(error)
            return
        scalar_row = {}
        list_fields = {}
        for field_name, raw_value in extracted.items():
            value = extract_value(raw_value)
            if isinstance(value, list):
                list_fields[field_name] = value
            else:
                scalar_row[field_name] = value

        if scalar_row:
            st.table(pd.DataFrame([scalar_row]))

        for field_name, items in list_fields.items():
            count = len(items)
            st.caption(f"**{field_name}** — {count} item(s)")
            if count:
                st.dataframe(pd.DataFrame(items), use_container_width=True)
            else:
                st.info("None found.")


def flatten_results_for_export(results: list) -> pd.DataFrame:
    """
    Denormalized flat export: scalar fields repeat across one row per
    list-item. If a document has multiple list fields, rows are the
    cross product's first list only (kept simple - the per-document
    nested view above is the source of truth; this is just for Excel).
    """
    rows = []
    for r in results:
        filename = r.get("filename")
        if r.get("error"):
            rows.append({"filename": filename, "error": r["error"]})
            continue
        extracted = r.get("extracted", {})
        scalar = {}
        list_fields = {}
        for field_name, raw_value in extracted.items():
            value = extract_value(raw_value)
            if isinstance(value, list):
                list_fields[field_name] = value
            else:
                scalar[field_name] = value

        if not list_fields:
            rows.append({"filename": filename, **scalar})
            continue

        primary_field, primary_items = next(iter(list_fields.items()))
        if not primary_items:
            rows.append({"filename": filename, **scalar})
        for item in primary_items:
            row = {"filename": filename, **scalar}
            row.update({f"{primary_field}.{k}": v for k, v in item.items()})
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Auth gate
# ----------------------------------------------------------------------
if not st.session_state.access_token:
    render_login()
    st.stop()

if not ensure_valid_session():
    st.warning("Your session expired. Please sign in again.")
    render_login()
    st.stop()

# ----------------------------------------------------------------------
# Top bar - account controls (no sidebar)
# ----------------------------------------------------------------------
top_l, top_r = st.columns([4, 1])
with top_l:
    st.title("📄 DocIntel Extraction Helper")
    st.caption("Define an extraction schema once, run it over batches of documents, and build up a results table.")
with top_r:
    st.write("")
    st.caption(f"Signed in as **{st.session_state.user_email}**")
    if st.button("Sign out", use_container_width=True):
        try:
            get_client().logout()
        except ApiError:
            pass  # best-effort - clear local session regardless
        _clear_session()
        st.rerun()

with st.expander("🔒 Change password"):
    with st.form("change_password_form"):
        old_password = st.text_input("Current password", type="password")
        new_password = st.text_input("New password", type="password")
        confirm_password = st.text_input("Confirm new password", type="password")
        change_submitted = st.form_submit_button("Change password")
    if change_submitted:
        if not old_password or not new_password:
            st.error("Fill in both current and new password.")
        elif new_password != confirm_password:
            st.error("New password and confirmation don't match.")
        elif len(new_password) < 8:
            st.error("New password must be at least 8 characters.")
        else:
            try:
                verify_client = DocIntelClient(get_base_url(), timeout=get_timeout())
                verify_result = verify_client.login(st.session_state.user_email, old_password)
                verify_client.reset_password(verify_result["access_token"], new_password)
                st.session_state.access_token = verify_result["access_token"]
                st.session_state.refresh_token = verify_result.get("refresh_token")
                st.success("Password changed.")
            except ApiError as e:
                if e.status_code == 401:
                    st.error("Current password is incorrect.")
                else:
                    st.error(f"Could not change password: {e}")

tab_schema, tab_run, tab_tables = st.tabs(["1️⃣ Build / Save Schema", "2️⃣ Run Extraction", "3️⃣ Saved Tables"])

# ----------------------------------------------------------------------
# TAB 1 - Schema builder
# ----------------------------------------------------------------------
with tab_schema:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Saved schemas")
        names = list_schema_names()
        if names:
            for n in names:
                s = get_schema(n)
                with st.expander(f"**{n}**"):
                    st.table(pd.DataFrame(flatten_fields_for_table(s.get("fields", []))))
                    if st.button("Delete schema", key=f"del_schema_{n}"):
                        delete_schema(n)
                        st.rerun()
        else:
            st.info("No schemas saved yet — build one on the right.")

    with right:
        st.subheader("Create a new schema")
        schema_name = st.text_input("Schema name", placeholder="e.g. resume_extraction")
        mode = st.radio("How do you want to define fields?", ["Manual fields", "From NL instruction (preview)"])

        # ---------------- Manual fields ----------------
        if mode == "Manual fields":
            st.caption("Define the fields you want extracted. Set a field's type to 'list' to add nested sub-fields.")
            render_field_editor(st.session_state.manual_fields, "manual")
            if st.button("+ Add field"):
                st.session_state.manual_fields.append(dict(EMPTY_FIELD))
                st.rerun()

            if st.button("💾 Save schema", type="primary"):
                fields = clean_fields_for_save(st.session_state.manual_fields)
                if not schema_name.strip():
                    st.error("Give the schema a name first.")
                elif not fields:
                    st.error("Add at least one field with a name.")
                else:
                    save_schema(schema_name.strip(), {"mode": "fields", "fields": fields})
                    st.success(f"Saved schema '{schema_name}'")
                    st.session_state.manual_fields = [dict(EMPTY_FIELD)]
                    st.rerun()

        # ---------------- NL instruction -> schema preview ----------------
        else:
            st.caption(
                "Describe in plain English what to extract, then preview the schema it derives — "
                "no document needed. Review/edit the fields (including nested sub-fields for list "
                "fields), then save."
            )
            instruction = st.text_area(
                "Instruction",
                placeholder="Extract candidate name, education (institute, course, branch, start/end year), "
                            "and work experience (company, location, start/end date, designation).",
                height=120,
            )

            if st.button("🔍 Preview schema"):
                if not instruction.strip():
                    st.error("Write an instruction first.")
                else:
                    try:
                        client = get_client()
                        with st.spinner("Generating schema preview..."):
                            preview_fields = client.preview_nl_schema(instruction.strip())
                        st.session_state.nl_previewed_instruction = instruction.strip()
                        st.session_state.nl_preview_fields = [
                            {**normalize_field(f), "include": True} for f in preview_fields
                        ]
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))

            previewed = st.session_state.nl_previewed_instruction
            instruction_changed = previewed is not None and previewed != instruction.strip()

            if previewed and not instruction_changed:
                if st.session_state.nl_preview_fields:
                    st.success("Preview generated — review and edit the fields below:")
                    render_field_editor(st.session_state.nl_preview_fields, "nl", show_include=True)
                else:
                    st.warning("Preview came back with no fields — try rewording the instruction.")
            elif instruction_changed:
                st.warning("Instruction changed since your last preview — preview again before saving.")
            else:
                st.info("Preview the schema above before you can save it.")

            can_save = bool(previewed) and not instruction_changed and bool(st.session_state.nl_preview_fields)
            if st.button("💾 Save schema", type="primary", key="save_nl_schema", disabled=not can_save):
                fields = clean_fields_for_save(st.session_state.nl_preview_fields, require_include=True)
                if not schema_name.strip():
                    st.error("Give the schema a name first.")
                elif not fields:
                    st.error("At least one previewed field needs to stay included with a name.")
                else:
                    save_schema(schema_name.strip(), {"mode": "fields", "fields": fields})
                    st.success(f"Saved schema '{schema_name}' ({len(fields)} field(s))")
                    st.session_state.nl_previewed_instruction = None
                    st.session_state.nl_preview_fields = []
                    st.rerun()

# ----------------------------------------------------------------------
# TAB 2 - Run extraction (parallel, pipelined upload -> extract per file)
# ----------------------------------------------------------------------
with tab_run:
    schema_names = list_schema_names()
    if not schema_names:
        st.warning("No saved schemas yet. Create one in the 'Build / Save Schema' tab first.")
    else:
        chosen_schema_name = st.selectbox("Schema to use", schema_names)
        schema = get_schema(chosen_schema_name)

        st.subheader("Select documents")
        source = st.radio("Document source", ["Upload files", "Local folder path"], horizontal=True)

        files_to_process = []  # list of (filename, bytes)

        if source == "Upload files":
            uploaded = st.file_uploader("Choose documents", accept_multiple_files=True)
            if uploaded:
                files_to_process = [(f.name, f.getvalue()) for f in uploaded]
        else:
            folder = st.text_input("Folder path (on the machine running this app)")
            if folder:
                p = Path(folder).expanduser()
                if p.is_dir():
                    found = [f for f in sorted(p.iterdir()) if f.suffix.lower() in ALLOWED_EXTENSIONS and f.is_file()]
                    st.write(f"Found {len(found)} matching file(s).")
                    if found:
                        st.write([f.name for f in found])
                        files_to_process = [(f.name, f.read_bytes()) for f in found]
                else:
                    st.error("That path doesn't exist or isn't a folder.")

        st.subheader("Destination table")
        st.caption(
            "Saved as a flattened Excel/CSV table (list fields exploded into rows). "
            "The nested view below the run button is always the full-fidelity source of truth."
        )
        existing_tables = list_tables()
        dest_mode = st.radio("Save results to", ["New table", "Append to existing table"], horizontal=True)
        if dest_mode == "New table":
            default_name = f"{chosen_schema_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
            table_name = st.text_input("New table name", value=default_name)
        else:
            if existing_tables:
                table_name = st.selectbox("Existing table", existing_tables)
            else:
                st.info("No existing tables yet — a new one will be created instead.")
                table_name = f"{chosen_schema_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
                dest_mode = "New table"

        max_parallel = get_max_parallel_uploads()

        run_disabled = not files_to_process
        if st.button("▶️ Run extraction", type="primary", disabled=run_disabled):
            progress = st.progress(0.0)
            status = st.empty()

            def _on_progress(completed, total, fname, ok):
                progress.progress(completed / total)
                mark = "✅" if ok else "❌"
                status.write(f"{mark} {fname} ({completed}/{total})")

            with st.spinner(f"Processing {len(files_to_process)} file(s)...."):
                rows, errors = run_batch(
                    base_url=get_base_url(),
                    access_token=st.session_state.access_token,
                    timeout=get_timeout(),
                    files=files_to_process,
                    fields=schema["fields"],
                    max_parallel=max_parallel,
                    schema_name=chosen_schema_name,
                    progress_cb=_on_progress,
                )

            status.write("Done.")
            st.session_state.last_run_results = rows  # expected: list of {"filename","extracted","error"?}
            st.session_state.last_run_schema_name = chosen_schema_name

            if errors:
                st.warning(f"{len(errors)} file(s) had errors — see details below.")
                with st.expander("Error details"):
                    for e in errors:
                        st.write("- " + e)

            flat_df = flatten_results_for_export(rows)
            saved_df = save_table(table_name, flat_df, mode="append" if dest_mode == "Append to existing table" else "new")
            st.success(f"Saved {len(flat_df)} row(s) to table '{table_name}' ({len(saved_df)} total rows).")

        if st.session_state.last_run_results is not None:
            st.subheader("Latest run results")
            for r in st.session_state.last_run_results:
                render_nested_result(r.get("filename", "unknown"), r.get("extracted", {}), r.get("error"))

            flat_df = flatten_results_for_export(st.session_state.last_run_results)
            st.download_button(
                "⬇️ Download this run as Excel (flattened)",
                data=to_excel_bytes(flat_df),
                file_name=f"{st.session_state.last_run_schema_name or 'results'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ----------------------------------------------------------------------
# TAB 3 - Saved tables (flattened - see caption above in Tab 2)
# ----------------------------------------------------------------------
with tab_tables:
    tables = list_tables()
    if not tables:
        st.info("No saved tables yet. Run an extraction to create one.")
    else:
        selected_table = st.selectbox("Table", tables)
        df = load_table(selected_table)
        st.write(f"{len(df)} row(s)")
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Download as Excel",
                data=to_excel_bytes(df, sheet_name=selected_table),
                file_name=f"{selected_table}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c2:
            if st.button("🗑️ Delete table"):
                delete_table(selected_table)
                st.rerun()