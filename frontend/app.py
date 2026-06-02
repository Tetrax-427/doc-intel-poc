import streamlit as st
import requests
import json
import mimetypes
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocIntel", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# --- Global styles ---
st.markdown("""
<style>
    /* Hide default Streamlit header/footer */
    #MainMenu, footer, header { visibility: hidden; }

    /* App background */
    .stApp { background-color: #f8f9fb; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e8eaed;
    }

    /* Brand header */
    .brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px;
        border-bottom: 1px solid #e8eaed;
        margin-bottom: 16px;
    }
    .brand-icon {
        font-size: 26px;
        line-height: 1;
    }
    .brand-name {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a2e;
        letter-spacing: -0.3px;
    }
    .brand-tag {
        font-size: 11px;
        color: #6b7280;
        margin-top: -2px;
    }

    /* Doc list items */
    .doc-active {
        background: #eff6ff;
        border-left: 3px solid #2563eb;
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 13px;
        color: #1d4ed8;
        font-weight: 500;
        margin-bottom: 4px;
    }

    /* Doc type badges */
    .doc-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 10px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-left: 4px;
    }
    .badge-invoice    { background: #fef3c7; color: #92400e; }
    .badge-resume     { background: #dbeafe; color: #1e40af; }
    .badge-cv_resume  { background: #dbeafe; color: #1e40af; }
    .badge-contract   { background: #ede9fe; color: #5b21b6; }
    .badge-report     { background: #d1fae5; color: #065f46; }
    .badge-financial  { background: #fce7f3; color: #9d174d; }
    .badge-medical    { background: #fee2e2; color: #991b1b; }
    .badge-legal      { background: #fef9c3; color: #713f12; }
    .badge-general    { background: #f3f4f6; color: #6b7280; }
    .badge-unknown    { background: #f3f4f6; color: #6b7280; }

    /* Review flag badge */
    .badge-review {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 1px 6px;
        border-radius: 10px;
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
    }

    /* Classification correction widget */
    .correction-box {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0;
        font-size: 12px;
    }
    .correction-box .title {
        font-weight: 600;
        color: #92400e;
        margin-bottom: 4px;
        font-size: 12px;
    }
    .correction-box .subtitle {
        color: #78350f;
        margin-bottom: 8px;
        font-size: 11px;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        color: #9ca3af;
    }
    .empty-state .icon { font-size: 48px; margin-bottom: 12px; }
    .empty-state .title { font-size: 18px; font-weight: 600; color: #374151; margin-bottom: 6px; }
    .empty-state .desc { font-size: 14px; color: #6b7280; line-height: 1.6; }

    /* Page title */
    .page-title {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0 4px;
        border-bottom: 1px solid #e8eaed;
        margin-bottom: 20px;
    }
    .page-title .icon { font-size: 22px; }
    .page-title .name { font-size: 20px; font-weight: 600; color: #1a1a2e; }
    .page-title .badge {
        font-size: 11px;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 20px;
        font-weight: 500;
    }

    /* Chat input fixed to bottom */
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 0;
        background: #f8f9fb;
        padding: 12px 0;
        z-index: 999;
        border-top: 1px solid #e8eaed;
    }
    .stChatMessageContainer { padding-bottom: 90px; }

    /* Error toast */
    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #dc2626;
        margin: 8px 0;
    }

    /* Success toast */
    .success-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #16a34a;
        margin: 8px 0;
    }

    /* Dark mode */
    @media (prefers-color-scheme: dark) {
        .stApp { background-color: #0e1117; }
        .stChatFloatingInputContainer { background: #0e1117; }
        [data-testid="stSidebar"] { background-color: #1a1a2e; border-right: 1px solid #2d2d44; }
        .brand-name { color: #f1f5f9; }
        .page-title .name { color: #f1f5f9; }
    }
</style>
""", unsafe_allow_html=True)

# --- Session state ---
if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []
if "multi_mode" not in st.session_state:
    st.session_state.multi_mode = False
if "history_summary" not in st.session_state:
    st.session_state.history_summary = ""
if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None
if "api_error" not in st.session_state:
    st.session_state.api_error = None
# Classification state
if "doc_classification" not in st.session_state:
    st.session_state.doc_classification = None
# NL extraction state — persists preview schema for edit-then-extract flow
if "nl_generated_schema" not in st.session_state:
    st.session_state.nl_generated_schema = {}
if "nl_instruction" not in st.session_state:
    st.session_state.nl_instruction = ""


# --- Helpers ---

# Maps doc_type values to (emoji, css_class) for badges
DOC_TYPE_DISPLAY = {
    "invoice":          ("🧾", "badge-invoice"),
    "receipt":          ("🧾", "badge-invoice"),
    "resume":           ("👤", "badge-resume"),
    "cv":               ("👤", "badge-cv_resume"),
    "cv_resume":        ("👤", "badge-cv_resume"),
    "contract":         ("📑", "badge-contract"),
    "agreement":        ("📑", "badge-contract"),
    "nda":              ("📑", "badge-contract"),
    "report":           ("📊", "badge-report"),
    "research paper":   ("📊", "badge-report"),
    "financial statement": ("💰", "badge-financial"),
    "balance sheet":    ("💰", "badge-financial"),
    "income statement": ("💰", "badge-financial"),
    "medical record":   ("🏥", "badge-medical"),
    "prescription":     ("🏥", "badge-medical"),
    "legal document":   ("⚖️",  "badge-legal"),
    "court filing":     ("⚖️",  "badge-legal"),
    "general":          ("📄", "badge-general"),
}

def doc_type_badge_html(doc_type: str | None) -> str:
    """Return an HTML badge span for a given doc_type."""
    if not doc_type or doc_type == "general":
        return ""
    _, css = DOC_TYPE_DISPLAY.get(doc_type, ("📄", "badge-unknown"))
    label = doc_type.replace("_", " ").title()
    return f'<span class="doc-badge {css}">{label}</span>'


def load_document(doc_id: str, doc_name: str):
    st.session_state.document_id = doc_id
    st.session_state.file_name = doc_name
    st.session_state.history_summary = ""
    st.session_state.api_error = None
    st.session_state["tables"] = []
    st.session_state.doc_summary = None
    st.session_state.doc_classification = None
    st.session_state["nl_instruction"] = ""
    st.session_state["nl_generated_schema"] = {}
    st.session_state["injected_schema"] = ""

    try:
        res = requests.get(f"{API_URL}/chats/{doc_id}", timeout=5)
        st.session_state.messages = res.json() if res.status_code == 200 else []
    except Exception:
        st.session_state.messages = []

    # Fetch summary
    try:
        res = requests.get(f"{API_URL}/summary/{doc_id}", timeout=15)
        if res.status_code == 200:
            st.session_state.doc_summary = res.json()
    except Exception:
        pass

    # Fetch classification
    try:
        res = requests.get(f"{API_URL}/documents/{doc_id}/classification", timeout=5)
        if res.status_code == 200:
            st.session_state.doc_classification = res.json()
    except Exception:
        pass


COMPRESSION_THRESHOLD = 10

def maybe_compress_history():
    if len(st.session_state.messages) >= COMPRESSION_THRESHOLD:
        try:
            res = requests.post(
                f"{API_URL}/compress",
                json={"messages": st.session_state.messages[:-4]},
                timeout=20
            )
            if res.status_code == 200:
                st.session_state.history_summary = res.json()["summary"]
                st.session_state.messages = st.session_state.messages[-4:]
        except Exception:
            pass


@st.cache_data(ttl=10)
def check_api():
    try:
        res = requests.get(f"{API_URL}/", timeout=5)
        return res.status_code == 200
    except Exception:
        return False


# --- Sidebar ---
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">📄</div>
        <div>
            <div class="brand-name">DocIntel</div>
            <div class="brand-tag">AI Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    api_ok = check_api()
    if api_ok:
        st.markdown('<div style="font-size:12px;color:#16a34a;margin-bottom:12px">🟢 API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:12px;color:#dc2626;margin-bottom:12px">🔴 API offline — start the FastAPI server</div>', unsafe_allow_html=True)

    # Upload
    st.markdown("**Upload Document**")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "csv", "xlsx", "rtf", "md", "png", "jpg", "jpeg", "webp", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        ext = uploaded_file.name.lower().split(".")[-1]
        is_llamaparse_type = ext in ["pdf", "png", "jpg", "jpeg", "webp", "tiff"]
        use_llamaparse = False
        vision_template = "general"

        if is_llamaparse_type:
            use_llamaparse = st.toggle("LlamaParse (better for tables/scans/images)", value=True)

        if ext in ["png", "jpg", "jpeg", "webp", "tiff"] or use_llamaparse:
            vision_template = st.selectbox(
                "Vision context",
                ["general", "cv_resume", "invoice", "construction_loan",
                "gst_return", "id_document", "bank_statement"],
                help="Select context so image descriptions are more accurate",
                key="vision_template_select"
            )

        if st.button("⬆️ Process Document", type="primary", use_container_width=True, disabled=not api_ok):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, mime_type)},
                        data={
                            "use_llamaparse": str(use_llamaparse),
                            "vision_template": vision_template
                        },
                        timeout=120
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("error"):
                            st.error(f"⚠️ {data['message'] if data.get('message') else data.get('error')}")
                        else:
                            vision_note = " · 🖼️ vision descriptions generated" if data.get("vision_used") else ""
                            st.success(
                                f"✅ {data['chunks_stored']} chunks · "
                                f"{data.get('parser', 'pypdf')}{vision_note}"
                            )
                            load_document(data["document_id"], data["file"])
                            st.rerun()
                    else:
                        st.error(f"Upload failed (HTTP {response.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Upload timed out.")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    st.divider()

    # URL Ingestion
    st.markdown("**Or ingest from URL**")
    url_input = st.text_input("Paste a URL", placeholder="https://example.com/article", label_visibility="collapsed")
    if url_input:
        if st.button("🔗 Ingest URL", use_container_width=True, disabled=not api_ok):
            with st.spinner("Fetching and indexing URL..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ingest-url",
                        json={"url": url_input},
                        timeout=60
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("error"):
                            st.error(f"⚠️ {data['message'] if data.get('message') else data.get('error')}")
                        else:
                            load_document(data["document_id"], data["file"])
                            st.success(f"✅ {data['chunks_stored']} chunks from URL")
                            st.rerun()
                    else:
                        st.error("Failed to ingest URL.")
                except requests.exceptions.Timeout:
                    st.error("⏱️ URL fetch timed out.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # ── Documents list ──────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Documents**")
    with col2:
        st.session_state.multi_mode = st.toggle("Multi", value=st.session_state.multi_mode, help="Query across multiple documents")

    try:
        docs_response = requests.get(f"{API_URL}/documents", timeout=5)
        if docs_response.status_code == 200:
            docs = docs_response.json()
            if docs:
                for doc in docs:
                    is_active = doc["id"] == st.session_state.document_id
                    is_selected = doc["id"] in st.session_state.selected_doc_ids
                    doc_type = doc.get("doc_type") or "general"
                    requires_review = doc.get("requires_review", False)
                    badge_html = doc_type_badge_html(doc_type)

                    cols = st.columns([5, 1])
                    with cols[0]:
                        if st.session_state.multi_mode:
                            checked = st.checkbox(
                                doc["name"],
                                value=is_selected,
                                key=f"chk_{doc['id']}"
                            )
                            if checked and doc["id"] not in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.append(doc["id"])
                            elif not checked and doc["id"] in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.remove(doc["id"])
                        else:
                            label = f"{'🟢 ' if is_active else '📄 '}{doc['name']}"
                            if st.button(label, key=f"load_{doc['id']}", use_container_width=True):
                                load_document(doc["id"], doc["name"])
                                st.session_state.selected_doc_ids = [doc["id"]]
                                st.rerun()

                            # Doc type badge + review flag
                            badge_parts = []
                            if badge_html:
                                badge_parts.append(badge_html)
                            if requires_review:
                                badge_parts.append('<span class="badge-review">⚠️ review</span>')
                            if badge_parts:
                                st.markdown(" ".join(badge_parts), unsafe_allow_html=True)

                            if doc.get("summary_short"):
                                st.caption(doc["summary_short"])

                    with cols[1]:
                        if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                            with st.spinner("Deleting..."):
                                requests.delete(f"{API_URL}/documents/{doc['id']}", timeout=5)
                            if st.session_state.document_id == doc["id"]:
                                st.session_state.document_id = None
                                st.session_state.file_name = None
                                st.session_state.messages = []
                                st.session_state.doc_classification = None
                            if doc["id"] in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.remove(doc["id"])
                            st.rerun()

                if st.session_state.multi_mode and st.session_state.selected_doc_ids:
                    st.info(f"📚 {len(st.session_state.selected_doc_ids)} doc(s) selected")
            else:
                st.markdown("""
                <div style="text-align:center;padding:20px 0;color:#9ca3af;font-size:13px">
                    No documents yet.<br>Upload a PDF above to get started.
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.warning("⚠️ Could not load documents. Is the API running?")

    # ── Classification correction widget ───────────────────────────────────
    # Shown only when the active document has a low-confidence classification
    clf = st.session_state.doc_classification
    if (
        clf
        and st.session_state.document_id
        and (clf.get("requires_review") or clf.get("classification_confidence", 1.0) < 0.75)
        and not clf.get("manually_overridden")
    ):
        st.divider()
        confidence_pct = int((clf.get("classification_confidence") or 0) * 100)
        current_type = clf.get("doc_type", "general")

        st.markdown(f"""
        <div class="correction-box">
            <div class="title">⚠️ Low-confidence classification</div>
            <div class="subtitle">
                Detected as <strong>{current_type.replace("_"," ").title()}</strong>
                ({confidence_pct}% confidence) — please confirm or correct.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # All known doc types the user can pick from
        ALL_DOC_TYPES = [
            "invoice", "receipt", "resume", "cv_resume", "contract",
            "agreement", "nda", "report", "research paper",
            "financial statement", "medical record", "prescription",
            "legal document", "court filing", "article", "email",
            "letter", "general",
        ]
        default_idx = ALL_DOC_TYPES.index(current_type) if current_type in ALL_DOC_TYPES else len(ALL_DOC_TYPES) - 1

        corrected_type = st.selectbox(
            "Correct document type",
            options=ALL_DOC_TYPES,
            index=default_idx,
            format_func=lambda x: x.replace("_", " ").title(),
            key="correction_select",
            label_visibility="collapsed"
        )
        if st.button("✅ Confirm classification", use_container_width=True, key="confirm_classification"):
            try:
                res = requests.post(
                    f"{API_URL}/documents/{st.session_state.document_id}/classification",
                    json={"doc_type": corrected_type},
                    timeout=10
                )
                if res.status_code == 200:
                    # Refresh local classification state
                    st.session_state.doc_classification = res.json().get("classification", clf)
                    st.session_state.doc_classification["manually_overridden"] = True
                    st.success(f"✅ Saved as {corrected_type.replace('_',' ').title()}")
                    st.rerun()
                else:
                    st.error("Could not save correction.")
            except Exception as e:
                st.error(f"Error: {e}")

    # Usage stats
    try:
        usage_res = requests.get(f"{API_URL}/usage", timeout=3)
        if usage_res.status_code == 200:
            usage = usage_res.json()
            if usage["total_calls"] > 0:
                st.divider()
                st.markdown("**Session Usage**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("LLM Calls", usage["total_calls"])
                with col2:
                    st.metric("~Tokens", f"{usage['total_tokens']:,}")
    except Exception:
        pass

    # Footer
    st.divider()
    st.markdown('<div style="font-size:11px;color:#9ca3af;text-align:center">DocIntel POC · v0.1</div>', unsafe_allow_html=True)

# --- Main area ---
is_multi = st.session_state.multi_mode and len(st.session_state.selected_doc_ids) > 1

if is_multi:
    st.markdown(f"""
    <div class="page-title">
        <div class="icon">📚</div>
        <div class="name">Multi-doc mode</div>
        <div class="badge">{len(st.session_state.selected_doc_ids)} documents selected</div>
    </div>
    """, unsafe_allow_html=True)
elif st.session_state.file_name:
    clf = st.session_state.doc_classification
    type_badge = ""
    if clf and clf.get("doc_type") and clf["doc_type"] != "general":
        emoji, _ = DOC_TYPE_DISPLAY.get(clf["doc_type"], ("📄", ""))
        confidence_pct = int((clf.get("classification_confidence") or 0) * 100)
        type_badge = f'<div class="badge">{emoji} {clf["doc_type"].replace("_"," ").title()} · {confidence_pct}%</div>'
    st.markdown(f"""
    <div class="page-title">
        <div class="icon">📄</div>
        <div class="name">{st.session_state.file_name}</div>
        {type_badge}
    </div>
    """, unsafe_allow_html=True)
else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📄</div>
        <div class="title">No document selected</div>
        <div class="desc">
            Upload a PDF from the sidebar to get started.<br>
            You can ask questions, extract fields, and query across multiple documents.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Document Summary ---
if st.session_state.doc_summary:
    summary = st.session_state.doc_summary
    details = summary.get("details", {})

    with st.expander(f"📋 Document Summary — {details.get('document_type', 'Document')}", expanded=True):
        col1, col2 = st.columns([3, 2])
        with col1:
            if details.get("overview"):
                st.markdown(f"**Overview**\n\n{details['overview']}")
            if details.get("key_topics"):
                st.markdown("**Key Topics**")
                for topic in details["key_topics"]:
                    st.markdown(f"- {topic}")
        with col2:
            if details.get("entities"):
                st.markdown("**People & Organizations**")
                st.markdown(", ".join(details["entities"]))
            if details.get("dates"):
                st.markdown("**Key Dates**")
                st.markdown(", ".join(details["dates"]))
            if details.get("amounts"):
                st.markdown("**Key Amounts**")
                st.markdown(", ".join([str(a) for a in details["amounts"]]))

        if not any([details.get("overview"), details.get("key_topics")]):
            st.caption(summary.get("summary_short", "No summary available."))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🗂️ Extract", "🤖 Smart Extract", "📊 Charts", "⚙️ Settings"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:

    col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
    with col2:
        if st.button("📄 PDF", help="Export chat as PDF", use_container_width=True):
            with st.spinner("Generating PDF..."):
                try:
                    res = requests.post(
                        f"{API_URL}/export/pdf",
                        json={
                            "document_id": st.session_state.document_id,
                            "file_name": st.session_state.file_name,
                            "messages": st.session_state.messages,
                            "summary": st.session_state.doc_summary or {}
                        },
                        timeout=30
                    )
                    if res.status_code == 200:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=res.content,
                            file_name=f"DocIntel_{st.session_state.file_name}.pdf",
                            mime="application/pdf",
                            key="pdf_dl"
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")
    with col3:
        if st.button("📝 Word", help="Export chat as Word doc", use_container_width=True):
            with st.spinner("Generating Word doc..."):
                try:
                    res = requests.post(
                        f"{API_URL}/export/docx",
                        json={
                            "document_id": st.session_state.document_id,
                            "file_name": st.session_state.file_name,
                            "messages": st.session_state.messages,
                            "summary": st.session_state.doc_summary or {}
                        },
                        timeout=30
                    )
                    if res.status_code == 200:
                        st.download_button(
                            label="⬇️ Download Word",
                            data=res.content,
                            file_name=f"DocIntel_{st.session_state.file_name}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="docx_dl"
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")
    with col4:
        if st.button("🗑️ Clear", help="Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history_summary = ""
            st.rerun()

    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#9ca3af">
            <div style="font-size:36px;margin-bottom:10px">💬</div>
            <div style="font-size:15px;font-weight:600;color:#374151;margin-bottom:6px">Ask anything about this document</div>
            <div style="font-size:13px;color:#6b7280">
                Try: "Summarise this document" · "What are the key dates?" · "Extract all amounts"
            </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        chunk_type = s.get("chunk_type", "text")
                        image_ref = s.get("image_ref")
                        file_info = f" · {s['file']}" if s.get("file") else ""
                        if chunk_type == "description":
                            type_icon = "🖼️ Image description"
                        elif chunk_type == "table":
                            type_icon = "📊 Table"
                        else:
                            type_icon = "📄 Text"
                        st.caption(f"{type_icon} · Page {s['page']}{file_info}" + (f" · ref: {image_ref}" if image_ref else ""))
                        st.text(s["preview"])
                        if chunk_type == "description" and image_ref:
                            st.info(f"💡 This answer references an image — see {image_ref} in the original document.")

    question = st.chat_input("Ask anything about the document...")

    if question:
        if is_multi:
            query_payload = {
                "question": question,
                "document_ids": st.session_state.selected_doc_ids,
                "history": st.session_state.messages[-4:],
                "history_summary": st.session_state.history_summary
            }
        else:
            query_payload = {
                "question": question,
                "document_id": st.session_state.document_id,
                "history": st.session_state.messages[-4:],
                "history_summary": st.session_state.history_summary
            }
        save_doc_id = st.session_state.selected_doc_ids[0] if is_multi else st.session_state.document_id

        requests.post(f"{API_URL}/chats/{save_doc_id}", json={"role": "user", "content": question})
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            full_response = ""

            try:
                with requests.post(
                    f"{API_URL}/query/stream",
                    json=query_payload,
                    stream=True,
                    timeout=60
                ) as stream_res:
                    if stream_res.status_code != 200:
                        placeholder.markdown(f'<div class="error-box">⚠️ Query failed (HTTP {stream_res.status_code})</div>', unsafe_allow_html=True)
                    else:
                        for line in stream_res.iter_lines(chunk_size=1):
                            if line:
                                decoded = line.decode("utf-8")
                                if decoded.startswith("data: "):
                                    token = decoded[6:]
                                    if token == "[DONE]":
                                        break
                                    import base64
                                    token = base64.b64decode(token.encode()).decode()
                                    full_response += token
                                    placeholder.markdown(full_response + "▌")
            except requests.exceptions.Timeout:
                placeholder.markdown('<div class="error-box">⏱️ Request timed out. Try a shorter question.</div>', unsafe_allow_html=True)
            except Exception:
                pass

            if full_response:
                placeholder.markdown(full_response)

            sources = []
            try:
                sources_res = requests.post(f"{API_URL}/query", json=query_payload, timeout=30)
                if sources_res.status_code == 200:
                    data = sources_res.json()
                    sources = data.get("sources", [])
                    answer_type = data.get("type", "document")

                    if answer_type == "general":
                        st.caption("💬 General answer — not from documents")
                    elif sources:
                        with st.expander("📎 Sources"):
                            for s in sources:
                                chunk_type = s.get("chunk_type", "text")
                                image_ref = s.get("image_ref")
                                file_info = f" · {s['file']}" if s.get("file") else ""
                                if chunk_type == "description":
                                    type_icon = "🖼️ Image description"
                                elif chunk_type == "table":
                                    type_icon = "📊 Table"
                                else:
                                    type_icon = "📄 Text"
                                st.caption(f"{type_icon} · Page {s['page']}{file_info}" + (f" · ref: {image_ref}" if image_ref else ""))
                                st.text(s["preview"])
                                if chunk_type == "description" and image_ref:
                                    st.info(f"💡 This answer references an image — see {image_ref} in the original document.")
            except Exception:
                pass

            if full_response:
                requests.post(
                    f"{API_URL}/chats/{save_doc_id}",
                    json={"role": "assistant", "content": full_response, "sources": sources}
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                maybe_compress_history()

# ── Tab 2: Extract ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("**Extract structured fields from this document.**")

    # Load templates
    try:
        templates_res = requests.get(f"{API_URL}/templates", timeout=5)
        templates = templates_res.json() if templates_res.status_code == 200 else []
    except Exception:
        templates = []

    # ── Auto-template selection based on classification ─────────────────────
    clf = st.session_state.doc_classification
    auto_template_id = None
    if clf and clf.get("doc_type") and clf["doc_type"] != "general":
        # classification_data may carry schema_template; fall back to doc_type
        clf_data = clf.get("classification_data") or {}
        auto_template_id = clf_data.get("schema_template") or clf.get("doc_type")

    template_options = {"custom": "✏️ Custom schema"}
    template_options.update({t["id"]: t["label"] for t in templates})

    # Determine default index: auto-select if classification matches a template
    template_keys = list(template_options.keys())
    default_template = "custom"
    if auto_template_id and auto_template_id in template_keys:
        default_template = auto_template_id

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Choose a template or define custom fields**")
        # Show auto-select notice if we pre-selected something
        if default_template != "custom" and clf:
            confidence_pct = int((clf.get("classification_confidence") or 0) * 100)
            st.caption(
                f"🤖 Auto-selected **{template_options[default_template]}** "
                f"based on document classification ({confidence_pct}% confidence). "
                f"Change below if needed."
            )
    with col2:
        selected_template = st.selectbox(
            "Template",
            options=template_keys,
            index=template_keys.index(default_template),
            format_func=lambda x: template_options[x],
            label_visibility="collapsed",
            key="extract_template_select"
        )

    # Load template schema
    if selected_template != "custom":
        try:
            tmpl_res = requests.get(f"{API_URL}/templates/{selected_template}", timeout=5)
            if tmpl_res.status_code == 200:
                tmpl_data = tmpl_res.json()
                template_schema = json.dumps(tmpl_data.get("schema", {}), indent=2)
                st.caption(tmpl_data.get("description", ""))
            else:
                template_schema = "{}"
        except Exception:
            template_schema = "{}"
    else:
        template_schema = json.dumps({
            "field_name": "description of what to extract",
            "another_field": "description of this field"
        }, indent=2)

    schema_input = st.text_area(
        "Extraction schema (JSON)",
        value=st.session_state.get("injected_schema") or template_schema,
        height=220,
        help="Keys = field names. Values = descriptions to guide extraction.",
        key=f"schema_{selected_template}"
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        extract_btn = st.button("🗂️ Extract Fields", type="primary", use_container_width=True)

    if extract_btn:
        try:
            fields = json.loads(schema_input)
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON in schema editor. Fix the JSON and try again.")
            fields = None

        if fields:
            with st.spinner("Extracting fields..."):
                try:
                    res = requests.post(
                        f"{API_URL}/extract",
                        json={"document_id": st.session_state.document_id, "fields": fields},
                        timeout=45
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("error"):
                            st.error(f"⚠️ {data.get('message', data.get('error'))}")
                        else:
                            validation = data.get("validation")
                            extracted = data.get("extracted", {})

                            if validation:
                                overall = validation["overall_confidence"]
                                found = validation["found_count"]
                                total = validation["total_count"]
                                if overall >= 0.85:
                                    st.success(f"✅ {found}/{total} fields extracted · {int(overall*100)}% confidence")
                                elif overall >= 0.5:
                                    st.warning(f"⚠️ {found}/{total} fields extracted · {int(overall*100)}% confidence")
                                else:
                                    st.error(f"❌ {found}/{total} fields extracted · {int(overall*100)}% confidence")

                                st.divider()
                                st.markdown("**Extracted Fields**")
                                for field_name, field_data in validation["fields"].items():
                                    status = field_data["status"]
                                    confidence = field_data["confidence"]
                                    value = field_data["value"]
                                    note = field_data.get("validation_note", "")
                                    icon = "🟢" if status == "FOUND" else "🟡" if status == "LOW_CONFIDENCE" else "🔴"
                                    c1, c2, c3 = st.columns([2, 3, 1])
                                    with c1:
                                        st.markdown(f"{icon} **{field_name}**")
                                        if note:
                                            st.caption(f"⚠️ {note}")
                                    with c2:
                                        if isinstance(value, list):
                                            st.caption(", ".join(str(v) for v in value) if value else "—")
                                        else:
                                            st.caption(str(value) if value else "—")
                                    with c3:
                                        st.caption(f"{int(confidence*100)}%")

                            st.divider()
                            with st.expander("📄 Raw JSON"):
                                st.json(extracted)
                            st.download_button(
                                label="⬇️ Download JSON",
                                data=json.dumps(extracted, indent=2),
                                file_name="extracted_fields.json",
                                mime="application/json"
                            )
                    else:
                        st.error(f"Extraction failed (HTTP {res.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timed out. The document may be too large.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Tab 3: Smart Extract ──────────────────────────────────────────────────────
with tab3:
    st.markdown("**Describe what you want to extract in plain English.**")
    st.caption("No need to define a schema — just tell DocIntel what you need.")

    # Example instructions
    st.markdown("**Examples:**")
    examples = [
        "Extract the candidate's name, email, phone, skills, and total experience",
        "Get all financial figures including invoice amount, tax, and payment terms",
        "Extract all dates, parties involved, and key obligations from this contract",
        "Pull out the applicant's income, loan amount requested, and employment details",
        "Get company name, GSTIN, total tax liability, and filing period"
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💡 {example[:50]}...", key=f"ex_{i}", use_container_width=True):
                st.session_state["nl_instruction"] = example
                st.session_state["nl_generated_schema"] = {}  # reset schema on new example

    st.divider()

    instruction = st.text_area(
        "Your extraction instruction",
        value=st.session_state.get("nl_instruction", ""),
        height=100,
        placeholder="e.g. Extract the candidate's name, email, current company, skills list, and years of experience",
        key="nl_input"
    )

    # Keep instruction in session state as user types
    if instruction != st.session_state.get("nl_instruction", ""):
        st.session_state["nl_instruction"] = instruction
        st.session_state["nl_generated_schema"] = {}  # reset schema when instruction changes

    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        preview_btn = st.button("👁️ Preview Schema", use_container_width=True)
    with col2:
        extract_nl_btn = st.button("🤖 Extract", type="primary", use_container_width=True)

    # ── Step 1: Preview / generate schema ───────────────────────────────────
    if preview_btn and instruction:
        with st.spinner("Generating schema from instruction..."):
            try:
                res = requests.post(
                    f"{API_URL}/extract/nl",
                    json={
                        "document_id": st.session_state.document_id,
                        "instruction": instruction,
                        "preview_only": True
                    },
                    timeout=20
                )
                if res.status_code == 200:
                    schema = res.json().get("schema", {})
                    st.session_state["nl_generated_schema"] = schema
                    st.session_state["nl_instruction"] = instruction
                else:
                    st.error("Failed to generate schema.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Step 2: Schema editor (shown after preview or if schema already exists) ─
    if st.session_state.get("nl_generated_schema"):
        st.markdown("**Generated Schema — edit if needed before extracting:**")
        st.caption("You can add, remove, or rename fields. Changes here will be used when you click Extract.")

        edited_schema_str = st.text_area(
            "Schema editor",
            value=json.dumps(st.session_state["nl_generated_schema"], indent=2),
            height=180,
            key="nl_schema_editor",
            label_visibility="collapsed"
        )

        # Parse and validate edits live
        schema_valid = True
        try:
            edited_schema = json.loads(edited_schema_str)
        except json.JSONDecodeError:
            st.warning("⚠️ Invalid JSON — fix the schema before extracting.")
            schema_valid = False
            edited_schema = st.session_state["nl_generated_schema"]

        # Action row: extract with edited schema OR copy to Extract tab
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            run_from_preview = st.button(
                "🤖 Extract with this schema",
                type="primary",
                use_container_width=True,
                disabled=not schema_valid,
                key="nl_extract_from_preview"
            )
        with action_col2:
            if st.button("📋 Copy to Extract tab", use_container_width=True, key="nl_copy_to_extract"):
                st.session_state["injected_schema"] = json.dumps(edited_schema, indent=2)
                st.success("✅ Schema copied — switch to the Extract tab.")

        if run_from_preview and schema_valid:
            with st.spinner("Extracting with your schema..."):
                try:
                    res = requests.post(
                        f"{API_URL}/extract",
                        json={
                            "document_id": st.session_state.document_id,
                            "fields": edited_schema
                        },
                        timeout=45
                    )
                    if res.status_code == 200:
                        _render_extraction_result(res.json(), instruction)
                    else:
                        st.error(f"Extraction failed (HTTP {res.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timed out.")
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()

    # ── Step 3: One-shot extract (no preview) ───────────────────────────────
    if extract_nl_btn and instruction:
        with st.spinner("Understanding instruction and extracting..."):
            try:
                res = requests.post(
                    f"{API_URL}/extract/nl",
                    json={
                        "document_id": st.session_state.document_id,
                        "instruction": instruction,
                        "preview_only": False
                    },
                    timeout=45
                )
                if res.status_code == 200:
                    data = res.json()
                    if data.get("error"):
                        st.error(f"⚠️ {data['error']}")
                    else:
                        # Store the generated schema for editing
                        if data.get("schema"):
                            st.session_state["nl_generated_schema"] = data["schema"]

                        with st.expander("🔍 Generated Schema", expanded=False):
                            st.json(data.get("schema", {}))

                        st.divider()
                        _render_extraction_result(data, instruction)
                else:
                    st.error(f"Extraction failed (HTTP {res.status_code})")
            except requests.exceptions.Timeout:
                st.error("⏱️ Timed out. Try a simpler instruction.")
            except Exception as e:
                st.error(f"Error: {e}")

    elif extract_nl_btn and not instruction:
        st.warning("Please enter an instruction first.")


# ── Tab 4: Charts ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("**Tables & Charts extracted from your document**")
    st.caption("DocIntel automatically detects tables and visualizes them.")

    if st.button("🔍 Extract Tables & Charts", type="primary"):
        with st.spinner("Scanning document for tables..."):
            try:
                res = requests.get(
                    f"{API_URL}/tables/{st.session_state.document_id}",
                    timeout=30
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["tables"] = data.get("tables", [])
                else:
                    st.error("Failed to extract tables.")
            except Exception as e:
                st.error(f"Error: {e}")

    tables = st.session_state.get("tables", [])

    if not tables:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px;color:#9ca3af">
            <div style="font-size:36px;margin-bottom:10px">📊</div>
            <div style="font-size:15px;font-weight:600;color:#374151;margin-bottom:6px">No tables extracted yet</div>
            <div style="font-size:13px;color:#6b7280">Click the button above to scan your document for tables and charts.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Found {len(tables)} table(s)")

        for i, table in enumerate(tables):
            st.divider()
            st.subheader(f"📋 {table.get('title', f'Table {i+1}')}")

            headers = table.get("headers", [])
            rows = table.get("rows", [])
            chart_type = table.get("chart_type", "bar")

            if not headers or not rows:
                st.caption("Could not parse this table.")
                continue

            try:
                df = pd.DataFrame(rows, columns=headers)
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.caption("Could not render table.")
                continue

            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                chart_choice = st.selectbox(
                    "Chart type",
                    ["bar", "line", "pie"],
                    index=["bar", "line", "pie"].index(chart_type) if chart_type in ["bar", "line", "pie"] else 0,
                    key=f"chart_type_{i}"
                )
            with col2:
                numeric_cols = [h for h in headers if any(
                    str(r[headers.index(h)]).replace('.','').replace('-','').replace(',','').isdigit()
                    for r in rows if len(r) > headers.index(h)
                )]
                x_col = st.selectbox("X axis", headers, index=0, key=f"x_{i}")
            with col3:
                y_options = numeric_cols if numeric_cols else headers
                y_col = st.selectbox("Y axis", y_options, index=0, key=f"y_{i}")

            try:
                df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(',', ''), errors='coerce')
                df = df.dropna(subset=[y_col])

                if chart_choice == "bar":
                    fig = px.bar(df, x=x_col, y=y_col, title=table.get("title", ""), color_discrete_sequence=["#2563eb"])
                elif chart_choice == "line":
                    fig = px.line(df, x=x_col, y=y_col, title=table.get("title", ""), markers=True, color_discrete_sequence=["#2563eb"])
                elif chart_choice == "pie":
                    fig = px.pie(df, names=x_col, values=y_col, title=table.get("title", ""))

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#374151",
                    margin=dict(t=40, b=20, l=20, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)

                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"{table.get('title', f'table_{i+1}')}.csv",
                    mime="text/csv",
                    key=f"dl_{i}"
                )
            except Exception as e:
                st.caption(f"Could not render chart: {e}")

# ── Tab 5: Settings ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("## ⚙️ Settings")

    # --- API Keys ---
    st.markdown("### 🔑 API Keys")
    st.caption("Generate keys to call DocIntel API from external systems.")

    col1, col2 = st.columns([3, 1])
    with col1:
        key_name = st.text_input("Key name", placeholder="e.g. Production, Zapier, Client A")
    with col2:
        rate_limit = st.number_input("Calls/day", value=100, min_value=1, max_value=10000)

    if st.button("➕ Generate API Key", type="primary"):
        if key_name:
            try:
                res = requests.post(
                    f"{API_URL}/api-keys",
                    json={"name": key_name, "rate_limit": rate_limit},
                    timeout=10
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success("✅ API key created — copy it now, it won't be shown again!")
                    st.code(data["key"], language=None)
                    st.caption(f"Prefix: `{data['prefix']}` · Rate limit: {data['rate_limit']} calls/day")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter a key name first.")

    try:
        keys_res = requests.get(f"{API_URL}/api-keys", timeout=5)
        if keys_res.status_code == 200:
            keys = keys_res.json()
            if keys:
                st.divider()
                st.markdown("**Existing Keys**")
                for k in keys:
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{k['name']}**")
                        st.caption(f"Prefix: `{k['key_prefix']}`")
                    with col2:
                        st.caption(f"{'🟢 Active' if k['is_active'] else '🔴 Revoked'}")
                    with col3:
                        st.caption(f"{k['calls_today']}/{k['rate_limit']} calls today")
                    with col4:
                        if st.button("🗑️", key=f"revoke_{k['id']}", help="Revoke key"):
                            requests.delete(f"{API_URL}/api-keys/{k['id']}", timeout=5)
                            st.rerun()
    except Exception:
        pass

    st.divider()

    # --- Webhooks ---
    st.markdown("### 🔗 Webhooks")
    st.caption("Send extraction results automatically to any URL after processing.")

    col1, col2 = st.columns([3, 1])
    with col1:
        wh_name = st.text_input("Webhook name", placeholder="e.g. Zapier, CRM, Slack")
        wh_url = st.text_input("Endpoint URL", placeholder="https://your-endpoint.com/webhook")
    with col2:
        wh_secret = st.text_input("Secret (optional)", placeholder="for signature verification", type="password")
        wh_events = st.multiselect(
            "Events",
            ["extraction.complete", "test.ping"],
            default=["extraction.complete"]
        )

    if st.button("➕ Add Webhook", type="primary"):
        if wh_name and wh_url:
            try:
                res = requests.post(
                    f"{API_URL}/webhooks",
                    json={
                        "name": wh_name,
                        "url": wh_url,
                        "events": wh_events,
                        "secret": wh_secret or None
                    },
                    timeout=10
                )
                if res.status_code == 200:
                    st.success("✅ Webhook added!")
                    st.rerun()
                else:
                    st.error("Failed to add webhook.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Enter webhook name and URL.")

    try:
        wh_res = requests.get(f"{API_URL}/webhooks", timeout=5)
        if wh_res.status_code == 200:
            webhooks = wh_res.json()
            if webhooks:
                st.divider()
                st.markdown("**Active Webhooks**")
                for wh in webhooks:
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    with col1:
                        st.markdown(f"**{wh['name']}**")
                        st.caption(wh['url'][:40] + "..." if len(wh['url']) > 40 else wh['url'])
                    with col2:
                        st.caption(f"{'🟢 Active' if wh['is_active'] else '🔴 Inactive'}")
                        if wh.get("last_triggered"):
                            st.caption(f"Last: {wh['last_triggered'][:10]}")
                    with col3:
                        st.caption(f"Fails: {wh.get('fail_count', 0)}")
                        st.caption(f"Events: {', '.join(wh.get('events', []))}")
                    with col4:
                        if st.button("🧪", key=f"test_{wh['id']}", help="Test webhook"):
                            test_res = requests.post(f"{API_URL}/webhooks/{wh['id']}/test", timeout=15)
                            if test_res.status_code == 200 and test_res.json().get("success"):
                                st.success("✅ Test sent!")
                            else:
                                st.error("Test failed.")
                        if st.button("🗑️", key=f"del_wh_{wh['id']}", help="Delete webhook"):
                            requests.delete(f"{API_URL}/webhooks/{wh['id']}", timeout=5)
                            st.rerun()

            st.divider()
            st.markdown("**Recent Webhook Logs**")
            try:
                logs_res = requests.get(f"{API_URL}/webhooks/logs", timeout=5)
                if logs_res.status_code == 200:
                    logs = logs_res.json()
                    if logs:
                        for log in logs[:10]:
                            icon = "✅" if log.get("success") else "❌"
                            st.caption(
                                f"{icon} {log.get('event')} · "
                                f"HTTP {log.get('response_status', '?')} · "
                                f"{log.get('created_at', '')[:16]}"
                            )
                    else:
                        st.caption("No webhook deliveries yet.")
            except Exception:
                pass
    except Exception:
        pass

    st.divider()

    # --- API Reference ---
    st.markdown("### 📡 API Reference")
    st.caption("Use these endpoints to integrate DocIntel into your own systems.")

    with st.expander("View API endpoints"):
        st.markdown(f"""
**Base URL:** `{API_URL}`

**Authentication:** Add header `X-API-Key: your_key` to any request.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document |
| POST | `/query` | Query a document |
| POST | `/extract` | Extract structured fields |
| POST | `/extract/nl` | Natural language extraction |
| POST | `/extract/batch` | Batch extraction across documents |
| GET | `/documents` | List all documents |
| GET | `/documents/{{id}}/classification` | Get document classification |
| POST | `/documents/{{id}}/classification` | Override classification |
| GET | `/summary/{{id}}` | Get document summary |
| GET | `/health` | System health check |
| POST | `/compress` | Compress chat history |
| GET | `/usage` | Get usage stats |

**Example — Extract fields via API:**
```bash
curl -X POST {API_URL}/extract/nl \\
  -H "X-API-Key: your_key" \\
  -H "Content-Type: application/json" \\
  -d '{{"document_id": "abc-123", "instruction": "extract name, email and skills"}}'
```
        """)


# ── Shared extraction result renderer ────────────────────────────────────────
# Defined after all tabs so it can be referenced above via forward reference.
# In Python this works fine since the function is called at runtime, not parse time.

def _render_extraction_result(data: dict, instruction: str = ""):
    """Render a validated extraction result (used in both NL flows)."""
    validation = data.get("validation")
    extracted = data.get("extracted", {})

    if validation:
        overall = validation["overall_confidence"]
        found = validation["found_count"]
        total = validation["total_count"]

        if overall >= 0.85:
            st.success(f"✅ {found}/{total} fields extracted · {int(overall*100)}% confidence")
        elif overall >= 0.5:
            st.warning(f"⚠️ {found}/{total} fields extracted · {int(overall*100)}% confidence")
        else:
            st.error(f"❌ {found}/{total} fields extracted · {int(overall*100)}% confidence")

        st.divider()
        st.markdown("**Extracted Fields**")

        for field_name, field_data in validation["fields"].items():
            status = field_data["status"]
            confidence = field_data["confidence"]
            value = field_data["value"]
            note = field_data.get("validation_note", "")
            icon = "🟢" if status == "FOUND" else "🟡" if status == "LOW_CONFIDENCE" else "🔴"

            c1, c2, c3 = st.columns([2, 3, 1])
            with c1:
                st.markdown(f"{icon} **{field_name}**")
                if note:
                    st.caption(f"⚠️ {note}")
            with c2:
                if isinstance(value, list):
                    st.caption(", ".join(str(v) for v in value) if value else "—")
                else:
                    st.caption(str(value) if value else "—")
            with c3:
                st.caption(f"{int(confidence*100)}%")

    st.divider()

    with st.expander("📄 Raw JSON"):
        st.json(extracted)

    col1, col2 = st.columns(2)
    with col1:
        download_data = json.dumps({
            "instruction": instruction,
            "extracted": extracted,
            "validation": {
                "overall_confidence": validation["overall_confidence"],
                "found_count": validation["found_count"],
                "total_count": validation["total_count"]
            } if validation else {}
        }, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=download_data,
            file_name="nl_extracted.json",
            mime="application/json",
            key=f"dl_nl_{hash(instruction)}"
        )
    with col2:
        if st.button("📋 Copy schema to Extract tab", key=f"copy_nl_{hash(instruction)}"):
            if data.get("schema"):
                st.session_state["injected_schema"] = json.dumps(data["schema"], indent=2)
                st.success("Schema copied! Switch to Extract tab.")