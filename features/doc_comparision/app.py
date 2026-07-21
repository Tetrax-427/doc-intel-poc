"""
features/doc_comparison/app.py

Standalone Streamlit app for the document comparison feature. Run with:
    streamlit run features/doc_comparison/app.py

Requires features/doc_comparison/.env (copy from .env.example) with
BACKEND_URL pointing at your running engine (main.py).

This app talks ONLY to the engine's own API (/auth/*, /compare) — no
Supabase URL/keys live here, matching the "engine as the only API
surface" design.
"""

import streamlit as st

from auth_client import is_logged_in, logout
from login import render_login_screen
from compare_client import compare_documents, CompareError
from render import render_compare_result

st.set_page_config(page_title="Document Comparison", page_icon="🔀", layout="wide")


def _init_session_state():
    defaults = {
        "auth": None,
        "compare_result": None,
        "compare_error": None,
        "auth_expired_notice": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar():
    with st.sidebar:
        user = st.session_state["auth"]["user"]
        st.write(f"Signed in as **{user['email']}**")
        if st.button("Log out"):
            logout()
            st.rerun()


def render_compare_tab():
    st.subheader("Compare two document versions")
    st.caption("Supported for now: PDF and DOCX (non-scanned).")

    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Original version", type=["pdf", "docx"], key="compare_file_a")
    with col2:
        file_b = st.file_uploader("New version", type=["pdf", "docx"], key="compare_file_b")

    include_summary = st.checkbox("Include AI summary of changes", value=False)

    if st.button("Compare documents", disabled=not (file_a and file_b)):
        with st.spinner("Comparing documents — this can take a while for large files..."):
            try:
                st.session_state["compare_result"] = compare_documents(file_a, file_b, include_summary)
                st.session_state["compare_error"] = None
            except CompareError as e:
                st.session_state["compare_error"] = e.detail
                st.session_state["compare_result"] = None
                if e.status_code == 401:
                    # Auto-refresh already tried once inside compare_client;
                    # a 401 here means the refresh token itself is dead.
                    st.rerun()

    if st.session_state["compare_error"]:
        st.error(st.session_state["compare_error"])

    if st.session_state["compare_result"]:
        render_compare_result(st.session_state["compare_result"])


def main():
    _init_session_state()

    if not is_logged_in():
        render_login_screen()
        return

    render_sidebar()
    render_compare_tab()


if __name__ == "__main__":
    main()