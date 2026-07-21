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

from api_client import ApiError, DocIntelClient
from batch_runner import run_batch
from config_store import get_base_url, get_timeout, get_max_parallel_uploads
from results_store import list_tables, load_table, save_table, delete_table, to_excel_bytes
from schema_store import list_schema_names, get_schema, save_schema, delete_schema

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".csv", ".xlsx", ".txt", ".md", ".jpg", ".jpeg", ".png"}
EMPTY_FIELD_ROW = {"name": "", "description": ""}

st.set_page_config(page_title="DocIntel Extraction Helper", layout="wide")

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key, default in [
    ("access_token", None),
    ("refresh_token", None),
    ("user_email", None),
    ("manual_fields", [dict(EMPTY_FIELD_ROW)]),
    ("last_run_df", None),
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
                # That login was already a fresh valid session - keep using it.
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
                    st.table(pd.DataFrame(s.get("fields", [])))
                    if st.button("Delete schema", key=f"del_schema_{n}"):
                        delete_schema(n)
                        st.rerun()
        else:
            st.info("No schemas saved yet — build one on the right.")

    with right:
        st.subheader("Create a new schema")
        schema_name = st.text_input("Schema name", placeholder="e.g. invoice_basic")
        mode = st.radio("How do you want to define fields?", ["Manual fields", "From NL instruction (preview)"])

        # ---------------- Manual fields ----------------
        if mode == "Manual fields":
            st.caption("Define the fields you want extracted from every document.")
            for i, f in enumerate(st.session_state.manual_fields):
                c1, c2, c3 = st.columns([2, 4, 1])
                f["name"] = c1.text_input("Field name", value=f["name"], key=f"fname_{i}")
                f["description"] = c2.text_input("Description", value=f["description"], key=f"fdesc_{i}")
                if c3.button("✕", key=f"fdel_{i}") and len(st.session_state.manual_fields) > 1:
                    st.session_state.manual_fields.pop(i)
                    st.rerun()
            if st.button("+ Add field"):
                st.session_state.manual_fields.append(dict(EMPTY_FIELD_ROW))
                st.rerun()

            if st.button("💾 Save schema", type="primary"):
                fields = [f for f in st.session_state.manual_fields if f["name"].strip()]
                if not schema_name.strip():
                    st.error("Give the schema a name first.")
                elif not fields:
                    st.error("Add at least one field with a name.")
                else:
                    save_schema(schema_name.strip(), {"mode": "fields", "fields": fields})
                    st.success(f"Saved schema '{schema_name}'")
                    st.session_state.manual_fields = [dict(EMPTY_FIELD_ROW)]
                    st.rerun()

        # ---------------- NL instruction -> schema preview ----------------
        else:
            st.caption(
                "Describe in plain English what to extract, then preview the schema it derives — "
                "no document needed. Review/edit the fields, then save."
            )
            instruction = st.text_area(
                "Instruction",
                placeholder="Extract candidate name, email, phone number, and total years of experience.",
                height=120,
            )

            if st.button("🔍 Preview schema"):
                if not instruction.strip():
                    st.error("Write an instruction first.")
                else:
                    try:
                        client = get_client()
                        with st.spinner("Generating schema preview..."):
                            schema_dict = client.preview_nl_schema(instruction.strip())
                        st.session_state.nl_previewed_instruction = instruction.strip()
                        st.session_state.nl_preview_fields = [
                            {"name": k, "description": v, "include": True}
                            for k, v in schema_dict.items()
                        ]
                        st.rerun()
                    except ApiError as e:
                        st.error(str(e))

            previewed = st.session_state.nl_previewed_instruction
            instruction_changed = previewed is not None and previewed != instruction.strip()

            if previewed and not instruction_changed:
                if st.session_state.nl_preview_fields:
                    st.success("Preview generated — review and edit the fields below:")
                    for i, f in enumerate(st.session_state.nl_preview_fields):
                        c0, c1, c2 = st.columns([0.5, 2, 5])
                        f["include"] = c0.checkbox("", value=f["include"], key=f"nl_inc_{i}")
                        f["name"] = c1.text_input("Field name", value=f["name"], key=f"nl_name_{i}")
                        f["description"] = c2.text_input("Description", value=f["description"], key=f"nl_desc_{i}")
                else:
                    st.warning("Preview came back with no fields — try rewording the instruction.")
            elif instruction_changed:
                st.warning("Instruction changed since your last preview — preview again before saving.")
            else:
                st.info("Preview the schema above before you can save it.")

            can_save = bool(previewed) and not instruction_changed and bool(st.session_state.nl_preview_fields)
            if st.button("💾 Save schema", type="primary", key="save_nl_schema", disabled=not can_save):
                fields = [
                    {"name": f["name"], "description": f["description"]}
                    for f in st.session_state.nl_preview_fields
                    if f["include"] and f["name"].strip()
                ]
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
        # st.caption(f"Running up to {max_parallel} file(s) at a time (set via DOCINTEL_MAX_PARALLEL_UPLOADS).")

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
                    progress_cb=_on_progress,
                )

            status.write("Done.")
            df = pd.DataFrame(rows)
            st.session_state.last_run_df = df
            st.session_state.last_run_schema_name = chosen_schema_name

            if errors:
                st.warning(f"{len(errors)} file(s) had errors — see 'error' column / details below.")
                with st.expander("Error details"):
                    for e in errors:
                        st.write("- " + e)

            saved_df = save_table(table_name, df, mode="append" if dest_mode == "Append to existing table" else "new")
            st.success(f"Saved {len(df)} row(s) to table '{table_name}' ({len(saved_df)} total rows).")

        if st.session_state.last_run_df is not None:
            st.subheader("Latest run results")
            st.dataframe(st.session_state.last_run_df, use_container_width=True)
            st.download_button(
                "⬇️ Download this run as Excel",
                data=to_excel_bytes(st.session_state.last_run_df),
                file_name=f"{st.session_state.last_run_schema_name or 'results'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ----------------------------------------------------------------------
# TAB 3 - Saved tables
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