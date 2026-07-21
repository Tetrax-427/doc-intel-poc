"""
features/doc_comparison/login.py

Interactive email/password login screen. Calls auth_client.login(),
which hits POST /auth/login on the engine directly — no Supabase
credentials in this app at all.
"""

import streamlit as st

from auth_client import login, AuthError


def render_login_screen():
    st.title("🔀 Document Comparison")
    st.caption("Sign in to compare document versions.")

    if st.session_state.get("auth_expired_notice"):
        st.warning("Your session expired. Please log in again.")
        st.session_state["auth_expired_notice"] = False

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if not email or not password:
            st.error("Enter both email and password.")
            return

        with st.spinner("Signing in..."):
            try:
                session = login(email, password)
                st.session_state["auth"] = session
                st.rerun()
            except AuthError as e:
                # Mirrors routers/auth.py: 401 on bad creds, 403 if
                # REQUIRE_EMAIL_VERIFICATION is on and the email isn't verified.
                if e.status_code == 403:
                    st.error(e.detail)
                else:
                    st.error("Invalid email or password.")