"""
DocIntel — app.py
Rebuilt: dark theme, deduplication, bug fixes, improved first-user flow.
"""

# ── Stdlib / third-party imports ─────────────────────────────────────────────
import time
import base64
import json
import mimetypes

import requests
import streamlit as st
import plotly.express as px
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"
COMPRESSION_THRESHOLD = 10

DOC_TYPE_DISPLAY = {
    "invoice":              ("🧾", "badge-invoice"),
    "receipt":              ("🧾", "badge-invoice"),
    "resume":               ("👤", "badge-resume"),
    "cv":                   ("👤", "badge-cv_resume"),
    "cv_resume":            ("👤", "badge-cv_resume"),
    "contract":             ("📑", "badge-contract"),
    "agreement":            ("📑", "badge-contract"),
    "nda":                  ("📑", "badge-contract"),
    "report":               ("📊", "badge-report"),
    "research paper":       ("📊", "badge-report"),
    "financial statement":  ("💰", "badge-financial"),
    "balance sheet":        ("💰", "badge-financial"),
    "income statement":     ("💰", "badge-financial"),
    "medical record":       ("🏥", "badge-medical"),
    "prescription":         ("🏥", "badge-medical"),
    "legal document":       ("⚖️",  "badge-legal"),
    "court filing":         ("⚖️",  "badge-legal"),
    "general":              ("📄", "badge-general"),
}

ALL_DOC_TYPES = [
    "invoice", "receipt", "resume", "cv_resume", "contract",
    "agreement", "nda", "report", "research paper",
    "financial statement", "medical record", "prescription",
    "legal document", "court filing", "article", "email",
    "letter", "general",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocIntel",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles (dark theme) ────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ── */
    #MainMenu, footer { visibility: hidden; }

    /* Keep the header visible so the sidebar toggle always works.
       Style it to blend into the dark theme instead of hiding it. */
    header[data-testid="stHeader"] {
        background-color: #0f1117 !important;
        border-bottom: 1px solid #2a2d3e !important;
    }
    header[data-testid="stHeader"] * {
        color: #94a3b8 !important;
    }

    .stApp {
        background-color: #0f1117;
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2a2d3e;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stMarkdown strong { color: #f1f5f9 !important; }

    /* ── Brand header ── */
    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0 18px;
        border-bottom: 1px solid #2a2d3e;
        margin-bottom: 18px;
    }
    .brand-icon { font-size: 28px; line-height: 1; }
    .brand-name {
        font-size: 20px;
        font-weight: 700;
        color: #f1f5f9 !important;
        letter-spacing: -0.4px;
    }
    .brand-tag {
        font-size: 11px;
        color: #64748b !important;
        margin-top: -2px;
        letter-spacing: 0.3px;
    }

    /* ── API status pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 20px;
        margin-bottom: 14px;
    }
    .status-ok   { background: #0d2b1f; color: #4ade80 !important; border: 1px solid #16a34a; }
    .status-err  { background: #2b0d0d; color: #f87171 !important; border: 1px solid #dc2626; }

    /* ── Doc type badges ── */
    .doc-badge {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        margin-left: 4px;
    }
    .badge-invoice   { background: #2d2008; color: #fcd34d !important; }
    .badge-resume    { background: #0d1f3c; color: #93c5fd !important; }
    .badge-cv_resume { background: #0d1f3c; color: #93c5fd !important; }
    .badge-contract  { background: #1e1040; color: #c4b5fd !important; }
    .badge-report    { background: #0a2318; color: #6ee7b7 !important; }
    .badge-financial { background: #2b0d2b; color: #f0abfc !important; }
    .badge-medical   { background: #2b0d0d; color: #fca5a5 !important; }
    .badge-legal     { background: #2b2508; color: #fde68a !important; }
    .badge-general   { background: #1e2130; color: #94a3b8 !important; }
    .badge-unknown   { background: #1e2130; color: #94a3b8 !important; }

    /* ── Review flag badge ── */
    .badge-review {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 10px;
        background: #2b2000;
        color: #fbbf24 !important;
        border: 1px solid #f59e0b;
    }

    /* ── Cards / containers ── */
    .doc-card {
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .doc-card-active {
        background: #0d1f3c;
        border: 1px solid #2563eb;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }

    /* ── Page title bar ── */
    .page-title {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0 6px;
        border-bottom: 1px solid #2a2d3e;
        margin-bottom: 22px;
    }
    .page-title .icon  { font-size: 22px; }
    .page-title .name  { font-size: 20px; font-weight: 600; color: #f1f5f9; }
    .page-title .badge {
        font-size: 11px;
        background: #0d1f3c;
        color: #93c5fd;
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 500;
        border: 1px solid #1d4ed8;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 70px 20px;
        color: #475569;
    }
    .empty-state .icon  { font-size: 52px; margin-bottom: 16px; }
    .empty-state .title { font-size: 18px; font-weight: 600; color: #94a3b8; margin-bottom: 8px; }
    .empty-state .desc  { font-size: 14px; color: #64748b; line-height: 1.7; }
    .empty-state .steps {
        display: inline-block;
        margin-top: 20px;
        text-align: left;
        background: #1e2130;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 16px 22px;
        font-size: 13px;
        color: #94a3b8;
        line-height: 2;
    }

    /* ── Correction box ── */
    .correction-box {
        background: #1e1a08;
        border: 1px solid #854d0e;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 12px;
    }
    .correction-box .title    { font-weight: 600; color: #fcd34d !important; margin-bottom: 4px; }
    .correction-box .subtitle { color: #fde68a !important; margin-bottom: 8px; font-size: 11px; }

    /* ── Chat ── */
    .stChatFloatingInputContainer {
        background: #0f1117 !important;
        border-top: 1px solid #2a2d3e;
        padding: 12px 0;
        z-index: 999;
    }
    .stChatMessageContainer { padding-bottom: 90px; }

    /* ── Streamlit widget overrides ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #1e2130 !important;
        border-color: #2a2d3e !important;
        color: #e2e8f0 !important;
    }
    .stButton > button {
        background-color: #1e2130;
        border-color: #2a2d3e;
        color: #e2e8f0;
        border-radius: 8px;
    }
    .stButton > button[kind="primary"] {
        background-color: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
    }
    .stButton > button:hover { border-color: #3b82f6 !important; }

    .stTabs [data-baseweb="tab-list"] {
        background: #161b27;
        border-bottom: 1px solid #2a2d3e;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 6px 6px 0 0;
        font-size: 13px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2130 !important;
        color: #93c5fd !important;
        border-bottom: 2px solid #2563eb !important;
    }

    .stExpander {
        background: #1e2130;
        border: 1px solid #2a2d3e !important;
        border-radius: 8px;
    }
    .stExpander summary { color: #94a3b8 !important; }

    div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    div[data-testid="stMetricLabel"] { color: #64748b !important; }

    .stAlert { border-radius: 8px !important; }

    /* ── Divider ── */
    hr { border-color: #2a2d3e !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0f1117; }
    ::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "document_id":        None,
    "file_name":          None,
    "messages":           [],
    "selected_doc_ids":   [],
    "multi_mode":         False,
    "history_summary":    "",
    "doc_summary":        None,
    "api_error":          None,
    "doc_classification": None,
    "nl_generated_schema": {},
    "nl_instruction":     "",
    "injected_schema":    "",
    "tables":             [],
    "review_data":        {},
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def doc_type_badge_html(doc_type: str | None) -> str:
    """Return an HTML badge span for a given doc_type."""
    if not doc_type or doc_type == "general":
        return ""
    _, css = DOC_TYPE_DISPLAY.get(doc_type, ("📄", "badge-unknown"))
    label = doc_type.replace("_", " ").title()
    return f'<span class="doc-badge {css}">{label}</span>'


def render_sources(sources: list):
    """Render source citations with type icons and exact evidence."""
    for s in sources:
        chunk_type  = s.get("chunk_type", "text")
        image_ref   = s.get("image_ref")
        exact       = s.get("exact_sentence")
        file_info   = f" · {s['file']}" if s.get("file") else ""

        if chunk_type == "description":
            type_icon, type_label = "🖼️", "Image description"
        elif chunk_type == "table":
            type_icon, type_label = "📊", "Table"
        else:
            type_icon, type_label = "📄", "Text"

        st.caption(
            f"{type_icon} {type_label} · Page {s['page']}{file_info}"
            + (f" · ref: {image_ref}" if image_ref else "")
        )
        if exact:
            st.markdown(f"> *\"{exact}\"*")
        elif s.get("preview"):
            st.text(s["preview"])
        if chunk_type == "description" and image_ref:
            st.info(f"💡 Answer references a visual element — see **{image_ref}** in the original document.")
        st.divider()


def render_extraction_result(data: dict, instruction: str = ""):
    """Render a validated extraction result. Used in Tab 2 and Tab 3."""
    validation = data.get("validation")
    extracted  = data.get("extracted", {})

    if validation:
        overall = validation["overall_confidence"]
        found   = validation["found_count"]
        total   = validation["total_count"]
        pct     = int(overall * 100)

        if overall >= 0.85:
            st.success(f"✅ {found}/{total} fields extracted · {pct}% confidence")
        elif overall >= 0.5:
            st.warning(f"⚠️ {found}/{total} fields extracted · {pct}% confidence")
        else:
            st.error(f"❌ {found}/{total} fields extracted · {pct}% confidence")

        st.divider()
        st.markdown("**Extracted Fields**")
        for field_name, field_data in validation["fields"].items():
            status     = field_data["status"]
            confidence = field_data["confidence"]
            value      = field_data["value"]
            note       = field_data.get("validation_note", "")
            icon       = "🟢" if status == "FOUND" else "🟡" if status == "LOW_CONFIDENCE" else "🔴"
            c1, c2, c3 = st.columns([2, 3, 1])
            with c1:
                st.markdown(f"{icon} **{field_name}**")
                if note:
                    st.caption(f"⚠️ {note}")
            with c2:
                st.caption(", ".join(str(v) for v in value) if isinstance(value, list) else str(value or "—"))
            with c3:
                st.caption(f"{int(confidence * 100)}%")

    # Business validation
    render_business_validation(data.get("business_validation", {}))

    st.divider()
    with st.expander("📄 Raw JSON"):
        st.json(extracted)

    col1, col2 = st.columns(2)
    with col1:
        dl_data = json.dumps({
            "instruction": instruction,
            "extracted":   extracted,
            "validation":  {
                "overall_confidence": validation["overall_confidence"],
                "found_count":        validation["found_count"],
                "total_count":        validation["total_count"],
            } if validation else {}
        }, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=dl_data,
            file_name="extracted_fields.json",
            mime="application/json",
            key=f"dl_extraction_{abs(hash(instruction))}",
        )
    with col2:
        if data.get("schema") and st.button("📋 Copy schema to Extract tab", key=f"copy_schema_{abs(hash(instruction))}"):
            st.session_state.injected_schema = json.dumps(data["schema"], indent=2)
            st.success("Schema copied — switch to the Extract tab.")


def render_business_validation(bv: dict):
    """Render business rule validation results. Shared across Tab 2, Tab 5."""
    if not bv or bv.get("rules_run", 0) == 0:
        return
    st.divider()
    st.markdown("**Business Rule Validation**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rules Run", bv.get("rules_run", 0))
    c2.metric("✅ Passed",  bv.get("passed", 0))
    c3.metric("❌ Failed",  bv.get("failed", 0))

    if bv.get("is_valid"):
        st.success("All business rules passed")
    else:
        with st.expander("🔍 Rule Results"):
            for r in bv.get("results", []):
                if r["status"] == "PASS":
                    st.markdown(f"✅ **{r['field']}** — {r['message']}")
                elif r["status"] == "FAIL":
                    icon = "🔴" if r["blocking"] else "🟡"
                    st.markdown(f"{icon} **{r['field']}** — {r['message']}")
                    st.caption(f"`{r['rule_code']}` · {'Blocking' if r['blocking'] else 'Warning'}")
                elif r["status"] == "WARNING":
                    st.markdown(f"⚠️ **{r['field']}** — {r['message']}")


def load_document(doc_id: str, doc_name: str):
    """Load a document into session state and prefetch summary + classification."""
    st.session_state.document_id        = doc_id
    st.session_state.file_name          = doc_name
    st.session_state.history_summary    = ""
    st.session_state.api_error          = None
    st.session_state.tables             = []
    st.session_state.doc_summary        = None
    st.session_state.doc_classification = None
    st.session_state.nl_instruction     = ""
    st.session_state.nl_generated_schema = {}
    st.session_state.injected_schema    = ""
    st.session_state.review_data        = {}

    try:
        res = requests.get(f"{API_URL}/chats/{doc_id}", timeout=5)
        st.session_state.messages = res.json() if res.status_code == 200 else []
    except Exception:
        st.session_state.messages = []

    try:
        res = requests.get(f"{API_URL}/summary/{doc_id}", timeout=15)
        if res.status_code == 200:
            st.session_state.doc_summary = res.json()
    except Exception:
        pass

    try:
        res = requests.get(f"{API_URL}/documents/{doc_id}/classification", timeout=5)
        if res.status_code == 200:
            st.session_state.doc_classification = res.json()
    except Exception:
        pass


def maybe_compress_history():
    """Compress chat history when it exceeds the threshold."""
    if len(st.session_state.messages) >= COMPRESSION_THRESHOLD:
        try:
            res = requests.post(
                f"{API_URL}/compress",
                json={"messages": st.session_state.messages[:-4]},
                timeout=20,
            )
            if res.status_code == 200:
                st.session_state.history_summary = res.json()["summary"]
                st.session_state.messages = st.session_state.messages[-4:]
        except Exception:
            pass


def show_upload_success(data: dict):
    """Show upload success feedback with classification info."""
    vision_note    = " · 🖼️ vision" if data.get("vision_used") else ""
    parser         = data.get("parser_used", data.get("parser", "unknown"))
    chunks         = data.get("chunks_stored", 0)
    st.success(f"✅ {chunks} chunks indexed · {parser}{vision_note}")

    clf        = data.get("classification", {})
    doc_type   = clf.get("doc_type", "")
    confidence = clf.get("confidence", 0.0)
    if doc_type and doc_type != "general":
        label    = doc_type.replace("_", " ").title()
        requires = clf.get("requires_human_review", False)
        if requires:
            st.warning(f"🏷️ Detected: **{label}** · {int(confidence*100)}% — please verify type")
        else:
            st.info(f"🏷️ Detected: **{label}** · {int(confidence*100)}% confidence")
    st.session_state.doc_classification = clf


@st.cache_data(ttl=10)
def check_api() -> bool:
    try:
        res = requests.get(f"{API_URL}/", timeout=5)
        return res.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">📄</div>
        <div>
            <div class="brand-name">DocIntel</div>
            <div class="brand-tag">AI Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API status ────────────────────────────────────────────────────────────
    api_ok = check_api()
    if api_ok:
        st.markdown('<div class="status-pill status-ok">🟢 API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-err">🔴 API offline — start the FastAPI server</div>', unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("**Upload Document**")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "csv", "xlsx", "rtf", "md",
              "png", "jpg", "jpeg", "webp", "tiff"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        ext                = uploaded_file.name.lower().rsplit(".", 1)[-1]
        is_llamaparse_type = ext in ["pdf", "png", "jpg", "jpeg", "webp", "tiff"]
        use_llamaparse     = False
        vision_template    = "general"

        if is_llamaparse_type:
            use_llamaparse = st.toggle("LlamaParse (better for tables/scans/images)", value=True)

        if ext in ["png", "jpg", "jpeg", "webp", "tiff"] or use_llamaparse:
            vision_template = st.selectbox(
                "Vision context",
                ["general", "cv_resume", "invoice", "construction_loan",
                 "gst_return", "id_document", "bank_statement"],
                help="Select context so image descriptions are more accurate",
                key="vision_template_select",
            )

        if st.button("⬆️ Process Document", type="primary", use_container_width=True, disabled=not api_ok):
            with st.spinner(f"Processing {uploaded_file.name}…"):
                try:
                    mime_type = mimetypes.guess_type(uploaded_file.name)[0] or "application/octet-stream"
                    response  = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, mime_type)},
                        data={"use_llamaparse": str(use_llamaparse), "vision_template": vision_template},
                        timeout=120,
                    )
                    if response.status_code == 200:
                        data = response.json()

                        if "task_id" in data:
                            # ── Async path ────────────────────────────────
                            task_id  = data["task_id"]
                            progress = st.progress(0, text="Processing document…")
                            max_wait, interval, waited = 120, 3, 0
                            done = False
                            while waited < max_wait:
                                time.sleep(interval)
                                waited += interval
                                progress.progress(min(waited / max_wait, 0.9), text=f"Processing… ({waited}s)")
                                try:
                                    status_res  = requests.get(f"{API_URL}/tasks/{task_id}", timeout=5)
                                    if status_res.status_code == 200:
                                        status_data = status_res.json()
                                        if status_data.get("status") == "done":
                                            result = status_data.get("result", {})
                                            progress.progress(1.0, text="Done!")
                                            if "document_id" in result:
                                                load_document(result["document_id"], result["file"])
                                                show_upload_success(result)
                                                done = True
                                                st.rerun()
                                            break
                                        elif status_data.get("status") == "failed":
                                            progress.empty()
                                            st.error(f"Processing failed: {status_data.get('error')}")
                                            break
                                except Exception:
                                    pass
                            if not done:
                                progress.empty()
                                st.warning("Still processing — check back in a moment.")

                        elif "document_id" in data:
                            # ── Sync path ─────────────────────────────────
                            if data.get("error"):
                                st.error(f"⚠️ {data.get('message', data.get('error'))}")
                            else:
                                load_document(data["document_id"], data["file"])
                                show_upload_success(data)
                                st.rerun()
                    else:
                        st.error(f"Upload failed (HTTP {response.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Upload timed out — try a smaller file or check your connection.")
                except Exception as e:
                    st.error(f"Upload error: {e}")

    st.divider()

    # ── URL Ingestion ─────────────────────────────────────────────────────────
    st.markdown("**Or ingest from URL**")
    url_input = st.text_input(
        "Paste a URL",
        placeholder="https://example.com/article",
        label_visibility="collapsed",
    )
    if url_input and st.button("🔗 Ingest URL", use_container_width=True, disabled=not api_ok):
        with st.spinner("Fetching and indexing URL…"):
            try:
                response = requests.post(f"{API_URL}/ingest-url", json={"url": url_input}, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("error"):
                        st.error(f"⚠️ {data.get('message', data.get('error'))}")
                    else:
                        load_document(data["document_id"], data["file"])
                        st.success(f"✅ {data['chunks_stored']} chunks indexed from URL")
                        st.rerun()
                else:
                    st.error("Failed to ingest URL.")
            except requests.exceptions.Timeout:
                st.error("⏱️ URL fetch timed out.")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # ── Document list ─────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**Documents**")
    with col_b:
        st.session_state.multi_mode = st.toggle(
            "Multi", value=st.session_state.multi_mode,
            help="Query across multiple documents at once",
        )

    try:
        docs_res = requests.get(f"{API_URL}/documents", timeout=5)
        if docs_res.status_code == 200:
            docs = docs_res.json()
            if docs:
                for doc in docs:
                    is_active   = doc["id"] == st.session_state.document_id
                    is_selected = doc["id"] in st.session_state.selected_doc_ids
                    doc_type    = doc.get("doc_type") or "general"
                    needs_review = doc.get("requires_review", False)
                    badge_html  = doc_type_badge_html(doc_type)

                    cols = st.columns([5, 1])
                    with cols[0]:
                        if st.session_state.multi_mode:
                            checked = st.checkbox(doc["name"], value=is_selected, key=f"chk_{doc['id']}")
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

                            badge_parts = []
                            if badge_html:
                                badge_parts.append(badge_html)
                            if needs_review:
                                badge_parts.append('<span class="badge-review">⚠️ review</span>')
                            if badge_parts:
                                st.markdown(" ".join(badge_parts), unsafe_allow_html=True)
                            if doc.get("summary_short"):
                                st.caption(doc["summary_short"])

                    with cols[1]:
                        if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                            with st.spinner("Deleting…"):
                                requests.delete(f"{API_URL}/documents/{doc['id']}", timeout=5)
                            if st.session_state.document_id == doc["id"]:
                                for k in ("document_id", "file_name", "messages", "doc_classification"):
                                    st.session_state[k] = [] if k == "messages" else None
                            if doc["id"] in st.session_state.selected_doc_ids:
                                st.session_state.selected_doc_ids.remove(doc["id"])
                            st.rerun()

                if st.session_state.multi_mode and st.session_state.selected_doc_ids:
                    st.info(f"📚 {len(st.session_state.selected_doc_ids)} doc(s) selected")
            else:
                st.markdown("""
                <div style="text-align:center;padding:24px 0;color:#475569;font-size:13px;line-height:1.8">
                    No documents yet.<br>Upload a file above to get started.
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.warning("⚠️ Could not load documents. Is the API running?")

    # ── Classification correction widget ─────────────────────────────────────
    clf = st.session_state.doc_classification
    if (
        clf
        and st.session_state.document_id
        and (clf.get("requires_review") or clf.get("classification_confidence", 1.0) < 0.75)
        and not clf.get("manually_overridden")
    ):
        st.divider()
        confidence_pct = int((clf.get("classification_confidence") or 0) * 100)
        current_type   = clf.get("doc_type", "general")
        st.markdown(f"""
        <div class="correction-box">
            <div class="title">⚠️ Low-confidence classification</div>
            <div class="subtitle">
                Detected as <strong>{current_type.replace("_"," ").title()}</strong>
                ({confidence_pct}%) — please confirm or correct.
            </div>
        </div>
        """, unsafe_allow_html=True)

        default_idx    = ALL_DOC_TYPES.index(current_type) if current_type in ALL_DOC_TYPES else len(ALL_DOC_TYPES) - 1
        corrected_type = st.selectbox(
            "Correct document type",
            options=ALL_DOC_TYPES,
            index=default_idx,
            format_func=lambda x: x.replace("_", " ").title(),
            key="correction_select",
            label_visibility="collapsed",
        )
        if st.button("✅ Confirm classification", use_container_width=True, key="confirm_classification"):
            try:
                res = requests.post(
                    f"{API_URL}/documents/{st.session_state.document_id}/classification",
                    json={"doc_type": corrected_type},
                    timeout=10,
                )
                if res.status_code == 200:
                    st.session_state.doc_classification = res.json().get("classification", clf)
                    st.session_state.doc_classification["manually_overridden"] = True
                    st.success(f"✅ Saved as {corrected_type.replace('_',' ').title()}")
                    st.rerun()
                else:
                    st.error("Could not save correction.")
            except Exception as e:
                st.error(f"Error: {e}")

    # ── Usage stats ───────────────────────────────────────────────────────────
    try:
        usage_res = requests.get(f"{API_URL}/usage", timeout=3)
        if usage_res.status_code == 200:
            usage = usage_res.json()
            if usage["total_calls"] > 0:
                st.divider()
                st.markdown("**Session Usage**")
                c1, c2 = st.columns(2)
                c1.metric("LLM Calls",  usage["total_calls"])
                c2.metric("~Tokens",    f"{usage['total_tokens']:,}")
    except Exception:
        pass

    st.divider()
    st.markdown('<div style="font-size:11px;color:#475569;text-align:center">DocIntel · v0.1 POC</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main area — header
# ══════════════════════════════════════════════════════════════════════════════
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
    clf        = st.session_state.doc_classification
    type_badge = ""
    if clf and clf.get("doc_type") and clf["doc_type"] != "general":
        emoji, _   = DOC_TYPE_DISPLAY.get(clf["doc_type"], ("📄", ""))
        conf_pct   = int((clf.get("classification_confidence") or 0) * 100)
        type_badge = f'<div class="badge">{emoji} {clf["doc_type"].replace("_"," ").title()} · {conf_pct}%</div>'
    st.markdown(f"""
    <div class="page-title">
        <div class="icon">📄</div>
        <div class="name">{st.session_state.file_name}</div>
        {type_badge}
    </div>
    """, unsafe_allow_html=True)

else:
    # ── First-user / empty state ──────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📄</div>
        <div class="title">Welcome to DocIntel</div>
        <div class="desc">
            Upload any document and start asking questions, extracting data,<br>
            and getting structured insights — no setup required.
        </div>
        <div class="steps">
            <strong style="color:#93c5fd">Get started in 3 steps</strong><br>
            1 · Upload a PDF, Word doc, image, or paste a URL in the sidebar<br>
            2 · DocIntel processes and indexes it automatically<br>
            3 · Chat, extract fields, or build charts — all from the tabs above
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Document summary (shown when available) ───────────────────────────────────
if st.session_state.doc_summary:
    summary = st.session_state.doc_summary
    details = summary.get("details", {})
    with st.expander(f"📋 Document Summary — {details.get('document_type', 'Document')}", expanded=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            if details.get("overview"):
                st.markdown(f"**Overview**\n\n{details['overview']}")
            if details.get("key_topics"):
                st.markdown("**Key Topics**")
                for topic in details["key_topics"]:
                    st.markdown(f"- {topic}")
        with c2:
            if details.get("entities"):
                st.markdown("**People & Organisations**")
                st.markdown(", ".join(details["entities"]))
            if details.get("dates"):
                st.markdown("**Key Dates**")
                st.markdown(", ".join(details["dates"]))
            if details.get("amounts"):
                st.markdown("**Key Amounts**")
                st.markdown(", ".join(str(a) for a in details["amounts"]))
        if not any([details.get("overview"), details.get("key_topics")]):
            st.caption(summary.get("summary_short", "No summary available."))


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 Chat", "🗂️ Extract", "🤖 Smart Extract", "📊 Charts", "👤 Review", "⚙️ Settings",
])


# ── Tab 1 · Chat ──────────────────────────────────────────────────────────────
with tab1:
    st.caption("Ask anything about this document in plain English.")

    # ── Toolbar ──────────────────────────────────────────────────────────────
    _, c_pdf, c_word, c_clear = st.columns([4, 1, 1, 1])

    with c_pdf:
        if st.button("📄 PDF", help="Export chat as PDF", use_container_width=True):
            with st.spinner("Generating PDF…"):
                try:
                    res = requests.post(
                        f"{API_URL}/export/pdf",
                        json={
                            "document_id": st.session_state.document_id,
                            "file_name":   st.session_state.file_name,
                            "messages":    st.session_state.messages,
                            "summary":     st.session_state.doc_summary or {},
                        },
                        timeout=30,
                    )
                    if res.status_code == 200:
                        st.download_button(
                            "⬇️ Download PDF",
                            data=res.content,
                            file_name=f"DocIntel_{st.session_state.file_name}.pdf",
                            mime="application/pdf",
                            key="pdf_dl",
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")

    with c_word:
        if st.button("📝 Word", help="Export chat as Word doc", use_container_width=True):
            with st.spinner("Generating Word doc…"):
                try:
                    res = requests.post(
                        f"{API_URL}/export/docx",
                        json={
                            "document_id": st.session_state.document_id,
                            "file_name":   st.session_state.file_name,
                            "messages":    st.session_state.messages,
                            "summary":     st.session_state.doc_summary or {},
                        },
                        timeout=30,
                    )
                    if res.status_code == 200:
                        st.download_button(
                            "⬇️ Download Word",
                            data=res.content,
                            file_name=f"DocIntel_{st.session_state.file_name}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key="docx_dl",
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")

    with c_clear:
        if st.button("🗑️ Clear", help="Clear chat history", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.history_summary = ""
            st.rerun()

    # ── Empty chat hint ───────────────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#475569">
            <div style="font-size:40px;margin-bottom:12px">💬</div>
            <div style="font-size:15px;font-weight:600;color:#94a3b8;margin-bottom:8px">
                Ask anything about this document
            </div>
            <div style="font-size:13px;color:#64748b;line-height:2">
                "Summarise this document" · "What are the key dates?"<br>
                "Extract all amounts" · "Who are the parties involved?"
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources"):
                    render_sources(msg["sources"])

    # ── Chat input ────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about the document…")
    if question:
        if is_multi:
            query_payload = {
                "question":        question,
                "document_ids":    st.session_state.selected_doc_ids,
                "history":         st.session_state.messages[-4:],
                "history_summary": st.session_state.history_summary,
            }
        else:
            query_payload = {
                "question":        question,
                "document_id":     st.session_state.document_id,
                "history":         st.session_state.messages[-4:],
                "history_summary": st.session_state.history_summary,
            }

        save_doc_id = st.session_state.selected_doc_ids[0] if is_multi else st.session_state.document_id
        requests.post(f"{API_URL}/chats/{save_doc_id}", json={"role": "user", "content": question})
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # Two separate elements: one holds the response text (never cleared),
            # one holds transient status messages (safe to overwrite/clear).
            response_placeholder = st.empty()
            status_placeholder   = st.empty()
            response_placeholder.markdown("_Thinking…_")
            full_response = ""

            try:
                with requests.post(
                    f"{API_URL}/query/stream",
                    json=query_payload,
                    stream=True,
                    timeout=60,
                ) as stream_res:
                    if stream_res.status_code != 200:
                        response_placeholder.error(f"⚠️ Query failed (HTTP {stream_res.status_code})")
                    else:
                        for line in stream_res.iter_lines(chunk_size=1):
                            if line:
                                decoded = line.decode("utf-8")
                                if decoded.startswith("data: "):
                                    token = decoded[6:]
                                    if token == "[DONE]":
                                        break
                                    token          = base64.b64decode(token.encode()).decode()
                                    full_response += token
                                    response_placeholder.markdown(full_response + "▌")
            except requests.exceptions.Timeout:
                response_placeholder.error("⏱️ Request timed out. Try a shorter question.")
            except Exception:
                pass

            # Lock in the final response — this element is never touched again
            if full_response:
                response_placeholder.markdown(full_response)
            
            # ── Fetch sources via separate call ───────────────────────────
            # Uses status_placeholder so the response text above is never overwritten
            sources = []
            try:
                status_placeholder.caption("_Fetching sources…_")
                sources_res = requests.post(f"{API_URL}/query", json=query_payload, timeout=30)
                status_placeholder.empty()
                if sources_res.status_code == 200:
                    resp_data   = sources_res.json()
                    sources     = resp_data.get("sources", [])
                    answer_type = resp_data.get("type", "document")
                    if answer_type == "general":
                        st.caption("💬 General answer — not from documents")
                    elif sources:
                        with st.expander("📎 Sources"):
                            render_sources(sources)
            except Exception:
                status_placeholder.empty()

            if full_response:
                requests.post(
                    f"{API_URL}/chats/{save_doc_id}",
                    json={"role": "assistant", "content": full_response, "sources": sources},
                )
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": full_response,
                    "sources": sources,
                })
                maybe_compress_history()


# ── Tab 2 · Extract ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("**Extract structured fields from this document.**")
    st.caption("Choose a template or define your own JSON schema.")

    # ── Load templates ────────────────────────────────────────────────────────
    try:
        templates_res = requests.get(f"{API_URL}/templates", timeout=5)
        templates     = templates_res.json() if templates_res.status_code == 200 else []
    except Exception:
        templates = []

    # ── Auto-select template from classification ──────────────────────────────
    clf             = st.session_state.doc_classification
    auto_tmpl_id    = None
    if clf and clf.get("doc_type") and clf["doc_type"] != "general":
        clf_data     = clf.get("classification_data") or {}
        auto_tmpl_id = clf_data.get("schema_template") or clf.get("doc_type")

    template_options = {"custom": "✏️ Custom schema"}
    template_options.update({t["id"]: t["label"] for t in templates})
    template_keys    = list(template_options.keys())
    default_template = auto_tmpl_id if (auto_tmpl_id and auto_tmpl_id in template_keys) else "custom"

    c1, c2 = st.columns([3, 2])
    with c1:
        if default_template != "custom" and clf:
            conf_pct = int((clf.get("classification_confidence") or 0) * 100)
            st.caption(
                f"🤖 Auto-selected **{template_options[default_template]}** "
                f"based on document type ({conf_pct}% confidence). Change below if needed."
            )
    with c2:
        selected_template = st.selectbox(
            "Template",
            options=template_keys,
            index=template_keys.index(default_template),
            format_func=lambda x: template_options[x],
            label_visibility="collapsed",
            key="extract_template_select",
        )

    # ── Schema editor ─────────────────────────────────────────────────────────
    if selected_template != "custom":
        try:
            tmpl_res = requests.get(f"{API_URL}/templates/{selected_template}", timeout=5)
            if tmpl_res.status_code == 200:
                tmpl_data       = tmpl_res.json()
                template_schema = json.dumps(tmpl_data.get("schema", {}), indent=2)
                st.caption(tmpl_data.get("description", ""))
            else:
                template_schema = "{}"
        except Exception:
            template_schema = "{}"
    else:
        template_schema = json.dumps({
            "field_name":    "description of what to extract",
            "another_field": "description of this field",
        }, indent=2)

    schema_input = st.text_area(
        "Extraction schema (JSON)",
        value=st.session_state.injected_schema or template_schema,
        height=220,
        help="Keys = field names · Values = descriptions to guide extraction",
        key=f"schema_{selected_template}",
    )

    if st.button("🗂️ Extract Fields", type="primary"):
        try:
            fields = json.loads(schema_input)
        except json.JSONDecodeError:
            st.error("⚠️ Invalid JSON in schema editor — fix it and try again.")
            fields = None

        if fields:
            with st.spinner("Extracting fields…"):
                try:
                    res = requests.post(
                        f"{API_URL}/extract",
                        json={"document_id": st.session_state.document_id, "fields": fields},
                        timeout=45,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("error"):
                            st.error(f"⚠️ {data.get('message', data.get('error'))}")
                        else:
                            render_extraction_result(data)
                    else:
                        st.error(f"Extraction failed (HTTP {res.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timed out. The document may be too large.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Tab 3 · Smart Extract ─────────────────────────────────────────────────────
with tab3:
    st.markdown("**Describe what you want to extract in plain English.**")
    st.caption("No schema needed — just tell DocIntel what you need.")

    # ── Example prompts ───────────────────────────────────────────────────────
    examples = [
        "Extract the candidate's name, email, phone, skills, and total experience",
        "Get all financial figures including invoice amount, tax, and payment terms",
        "Extract all dates, parties involved, and key obligations from this contract",
        "Pull out applicant income, loan amount requested, and employment details",
        "Get company name, GSTIN, total tax liability, and filing period",
    ]
    st.markdown("**Try an example:**")
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"💡 {ex[:52]}…", key=f"ex_{i}", use_container_width=True):
                st.session_state.nl_instruction      = ex
                st.session_state.nl_generated_schema = {}

    st.divider()

    instruction = st.text_area(
        "Your extraction instruction",
        value=st.session_state.nl_instruction,
        height=100,
        placeholder="e.g. Extract the candidate's name, email, current company, skills, and years of experience",
        key="nl_input",
    )
    # Sync instruction into session state on change
    if instruction != st.session_state.nl_instruction:
        st.session_state.nl_instruction      = instruction
        st.session_state.nl_generated_schema = {}

    c1, c2, _ = st.columns([2, 2, 3])
    with c1:
        preview_btn   = st.button("👁️ Preview Schema", use_container_width=True)
    with c2:
        extract_nl_btn = st.button("🤖 Extract", type="primary", use_container_width=True)

    # ── Preview schema ────────────────────────────────────────────────────────
    if preview_btn and instruction:
        with st.spinner("Generating schema from instruction…"):
            try:
                res = requests.post(
                    f"{API_URL}/extract/nl",
                    json={"document_id": st.session_state.document_id, "instruction": instruction, "preview_only": True},
                    timeout=20,
                )
                if res.status_code == 200:
                    st.session_state.nl_generated_schema = res.json().get("schema", {})
                else:
                    st.error("Failed to generate schema.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.nl_generated_schema:
        st.markdown("**Generated Schema — edit if needed before extracting:**")
        edited_schema_str = st.text_area(
            "Schema editor",
            value=json.dumps(st.session_state.nl_generated_schema, indent=2),
            height=180,
            key="nl_schema_editor",
            label_visibility="collapsed",
        )
        schema_valid = True
        try:
            edited_schema = json.loads(edited_schema_str)
        except json.JSONDecodeError:
            st.warning("⚠️ Invalid JSON — fix before extracting.")
            schema_valid  = False
            edited_schema = st.session_state.nl_generated_schema

        a1, a2 = st.columns(2)
        with a1:
            run_from_preview = st.button(
                "🤖 Extract with this schema",
                type="primary",
                use_container_width=True,
                disabled=not schema_valid,
                key="nl_extract_from_preview",
            )
        with a2:
            if st.button("📋 Copy to Extract tab", use_container_width=True, key="nl_copy_to_extract"):
                st.session_state.injected_schema = json.dumps(edited_schema, indent=2)
                st.success("✅ Schema copied — switch to the Extract tab.")

        if run_from_preview and schema_valid:
            with st.spinner("Extracting with your schema…"):
                try:
                    res = requests.post(
                        f"{API_URL}/extract",
                        json={"document_id": st.session_state.document_id, "fields": edited_schema},
                        timeout=45,
                    )
                    if res.status_code == 200:
                        render_extraction_result(res.json(), instruction)
                    else:
                        st.error(f"Extraction failed (HTTP {res.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timed out.")
                except Exception as e:
                    st.error(f"Error: {e}")
        st.divider()

    # ── Direct extract ────────────────────────────────────────────────────────
    if extract_nl_btn:
        if not instruction:
            st.warning("Enter an instruction first.")
        else:
            with st.spinner("Understanding instruction and extracting…"):
                try:
                    res = requests.post(
                        f"{API_URL}/extract/nl",
                        json={"document_id": st.session_state.document_id, "instruction": instruction, "preview_only": False},
                        timeout=45,
                    )
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("error"):
                            st.error(f"⚠️ {data['error']}")
                        else:
                            if data.get("schema"):
                                st.session_state.nl_generated_schema = data["schema"]
                            with st.expander("🔍 Generated Schema", expanded=False):
                                st.json(data.get("schema", {}))
                            st.divider()
                            render_extraction_result(data, instruction)
                    else:
                        st.error(f"Extraction failed (HTTP {res.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Timed out. Try a simpler instruction.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Tab 4 · Charts ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("**Tables & Charts extracted from your document.**")
    st.caption("DocIntel auto-detects tables and lets you visualise them instantly.")

    if st.button("🔍 Extract Tables & Charts", type="primary"):
        with st.spinner("Scanning document for tables…"):
            try:
                res = requests.get(f"{API_URL}/tables/{st.session_state.document_id}", timeout=30)
                if res.status_code == 200:
                    st.session_state.tables = res.json().get("tables", [])
                else:
                    st.error("Failed to extract tables.")
            except Exception as e:
                st.error(f"Error: {e}")

    tables = st.session_state.tables
    if not tables:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#475569">
            <div style="font-size:40px;margin-bottom:12px">📊</div>
            <div style="font-size:15px;font-weight:600;color:#94a3b8;margin-bottom:8px">No tables extracted yet</div>
            <div style="font-size:13px;color:#64748b">Click the button above to scan your document.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ Found {len(tables)} table(s)")
        for i, table in enumerate(tables):
            st.divider()
            st.subheader(f"📋 {table.get('title', f'Table {i+1}')}")
            headers    = table.get("headers", [])
            rows       = table.get("rows", [])
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

            c1, c2, c3 = st.columns(3)
            with c1:
                chart_choice = st.selectbox(
                    "Chart type",
                    ["bar", "line", "pie"],
                    index=["bar", "line", "pie"].index(chart_type) if chart_type in ["bar", "line", "pie"] else 0,
                    key=f"chart_type_{i}",
                )
            with c2:
                numeric_cols = [
                    h for h in headers
                    if any(
                        str(r[headers.index(h)]).replace(".", "").replace("-", "").replace(",", "").isdigit()
                        for r in rows if len(r) > headers.index(h)
                    )
                ]
                x_col = st.selectbox("X axis", headers, index=0, key=f"x_{i}")
            with c3:
                y_options = numeric_cols if numeric_cols else headers
                y_col     = st.selectbox("Y axis", y_options, index=0, key=f"y_{i}")

            try:
                df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(",", ""), errors="coerce")
                df        = df.dropna(subset=[y_col])

                if chart_choice == "bar":
                    fig = px.bar(df,  x=x_col, y=y_col, title=table.get("title", ""), color_discrete_sequence=["#2563eb"])
                elif chart_choice == "line":
                    fig = px.line(df, x=x_col, y=y_col, title=table.get("title", ""), markers=True, color_discrete_sequence=["#2563eb"])
                else:
                    fig = px.pie(df,  names=x_col, values=y_col, title=table.get("title", ""))

                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "⬇️ Download CSV",
                    data=df.to_csv(index=False),
                    file_name=f"{table.get('title', f'table_{i+1}')}.csv",
                    mime="text/csv",
                    key=f"dl_{i}",
                )
            except Exception as e:
                st.caption(f"Could not render chart: {e}")


# ── Tab 5 · Review ────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## 👤 Human Review")
    st.caption("Verify extracted fields with evidence. Approve, correct, or reject each field.")

    # ── Guard: no document ────────────────────────────────────────────────────
    if not st.session_state.document_id:
        st.info("Select a document from the sidebar to start reviewing.")
    else:
        # ── Load extraction for review ────────────────────────────────────────
        c1, c2 = st.columns([4, 1])
        with c1:
            review_instruction = st.text_input(
                "What to extract for review",
                placeholder="e.g. extract total amount, vendor name, and invoice date",
                label_visibility="collapsed",
            )
        with c2:
            load_btn = st.button("📥 Load", type="primary", use_container_width=True)

        if load_btn and review_instruction:
            with st.spinner("Extracting for review…"):
                try:
                    res = requests.post(
                        f"{API_URL}/extract/nl",
                        json={"document_id": st.session_state.document_id, "instruction": review_instruction},
                        timeout=30,
                    )
                    if res.status_code == 200:
                        st.session_state.review_data = res.json()
                    else:
                        st.error(f"Extraction failed (HTTP {res.status_code})")
                except Exception as e:
                    st.error(f"Error: {e}")

        data = st.session_state.review_data
        if not data:
            st.info("Enter an extraction instruction above and click Load.")
        else:
            extracted   = data.get("extracted", {})
            validation  = data.get("validation", {})
            business_val = data.get("business_validation", {})
            sources     = data.get("sources", [])

            # ── Business validation banner ────────────────────────────────────
            if business_val and business_val.get("rules_run", 0) > 0:
                if business_val.get("is_valid"):
                    st.success(f"✅ All business rules passed — {business_val['passed']}/{business_val['rules_run']} OK")
                else:
                    st.error(f"❌ {business_val['blocking_failures']} blocking failure(s) — review required")
                    with st.expander("🔍 Rule Details"):
                        for r in business_val.get("results", []):
                            if r["status"] == "FAIL":
                                icon = "🔴" if r["blocking"] else "🟡"
                                st.markdown(f"{icon} **{r['field']}** — {r['message']}")
                                st.caption(f"Rule: `{r['rule_code']}`")

            st.divider()
            st.markdown("**Review each field:**")
            actions_to_submit = {}

            for field_name, value in extracted.items():
                field_val  = validation.get("fields", {}).get(field_name, {})
                confidence = field_val.get("confidence", 0)
                status     = field_val.get("status", "NOT_FOUND")
                val_note   = field_val.get("validation_note", "")
                conf_icon  = "🟢" if status == "FOUND" else "🟡" if status == "LOW_CONFIDENCE" else "🔴"

                with st.expander(f"{conf_icon} **{field_name}** — {int(confidence*100)}% confidence", expanded=(status != "FOUND")):
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown("**Extracted value:**")
                        if isinstance(value, list):
                            st.write(value)
                        else:
                            st.code(str(value) if value else "— not found —")
                        if val_note:
                            st.caption(f"⚠️ Format issue: {val_note}")
                        if sources:
                            st.markdown("**Evidence:**")
                            for s in sources[:2]:
                                chunk_type    = s.get("chunk_type", "text")
                                ev_icon       = "🖼️" if chunk_type == "description" else "📄"
                                evidence_text = s.get("exact_sentence") or s.get("preview", "")
                                if evidence_text:
                                    st.caption(f"{ev_icon} Page {s['page']}: \"{evidence_text[:120]}\"")

                    with c2:
                        st.markdown("**Your decision:**")
                        action = st.radio(
                            "Action",
                            ["✅ Approve", "✏️ Correct", "❌ Reject"],
                            key=f"action_{field_name}",
                            label_visibility="collapsed",
                        )
                        corrected_value = str(value) if value else ""
                        if "Correct" in action:
                            corrected_value = st.text_input(
                                "Corrected value",
                                value=str(value) if value else "",
                                key=f"corrected_{field_name}",
                            )
                        reviewer_note = st.text_input(
                            "Note (optional)",
                            key=f"note_{field_name}",
                            placeholder="Why did you change this?",
                        )
                        actions_to_submit[field_name] = {
                            "action":          "approve" if "Approve" in action else "correct" if "Correct" in action else "reject",
                            "original_value":  str(value) if value else "",
                            "corrected_value": corrected_value,
                            "reviewer_note":   reviewer_note,
                        }

            st.divider()
            if st.button("💾 Submit Review", type="primary"):
                with st.spinner("Saving review…"):
                    review_payload = [
                        {
                            "field":           field,
                            "action":          d["action"],
                            "original_value":  d["original_value"],
                            "corrected_value": d["corrected_value"],
                            "reviewer_note":   d["reviewer_note"],
                        }
                        for field, d in actions_to_submit.items()
                    ]
                    try:
                        res = requests.post(
                            f"{API_URL}/review/{st.session_state.document_id}",
                            json=review_payload,
                            timeout=10,
                        )
                        if res.status_code == 200:
                            approved  = sum(1 for a in review_payload if a["action"] == "approve")
                            corrected = sum(1 for a in review_payload if a["action"] == "correct")
                            rejected  = sum(1 for a in review_payload if a["action"] == "reject")
                            st.success(f"✅ Saved — {approved} approved · {corrected} corrected · {rejected} rejected")
                            final = {
                                field: d["corrected_value"]
                                for field, d in actions_to_submit.items()
                                if d["action"] != "reject"
                            }
                            st.download_button(
                                "⬇️ Download reviewed JSON",
                                data=json.dumps(final, indent=2),
                                file_name="reviewed_extraction.json",
                                mime="application/json",
                            )
                            st.session_state.review_data = {}
                        else:
                            st.error("Failed to save review.")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ── Tab 6 · Settings ──────────────────────────────────────────────────────────
with tab6:
    st.markdown("## ⚙️ Settings")

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.markdown("### 🔑 API Keys")
    st.caption("Generate keys to call the DocIntel API from external systems.")

    c1, c2 = st.columns([3, 1])
    with c1:
        key_name = st.text_input("Key name", placeholder="e.g. Production, Zapier, Client A")
    with c2:
        rate_limit = st.number_input("Calls/day", value=100, min_value=1, max_value=10000)

    if st.button("➕ Generate API Key", type="primary"):
        if key_name:
            try:
                res = requests.post(
                    f"{API_URL}/api-keys",
                    json={"name": key_name, "rate_limit": rate_limit},
                    timeout=10,
                )
                if res.status_code == 200:
                    data = res.json()
                    st.success("✅ Key created — copy it now, it won't be shown again!")
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
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                    c1.markdown(f"**{k['name']}**")
                    c1.caption(f"Prefix: `{k['key_prefix']}`")
                    c2.caption("🟢 Active" if k["is_active"] else "🔴 Revoked")
                    c3.caption(f"{k['calls_today']}/{k['rate_limit']} calls today")
                    with c4:
                        if st.button("🗑️", key=f"revoke_{k['id']}", help="Revoke key"):
                            requests.delete(f"{API_URL}/api-keys/{k['id']}", timeout=5)
                            st.rerun()
    except Exception:
        pass

    st.divider()

    # ── Webhooks ──────────────────────────────────────────────────────────────
    st.markdown("### 🔗 Webhooks")
    st.caption("Push extraction results to any endpoint automatically after processing.")

    c1, c2 = st.columns([3, 1])
    with c1:
        wh_name   = st.text_input("Webhook name", placeholder="e.g. Zapier, CRM, Slack")
        wh_url    = st.text_input("Endpoint URL",  placeholder="https://your-endpoint.com/webhook")
    with c2:
        wh_secret = st.text_input("Secret (optional)", placeholder="for signature verification", type="password")
        wh_events = st.multiselect("Events", ["extraction.complete", "test.ping"], default=["extraction.complete"])

    if st.button("➕ Add Webhook", type="primary"):
        if wh_name and wh_url:
            try:
                res = requests.post(
                    f"{API_URL}/webhooks",
                    json={"name": wh_name, "url": wh_url, "events": wh_events, "secret": wh_secret or None},
                    timeout=10,
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
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
                    c1.markdown(f"**{wh['name']}**")
                    c1.caption(wh["url"][:40] + "…" if len(wh["url"]) > 40 else wh["url"])
                    c2.caption("🟢 Active" if wh["is_active"] else "🔴 Inactive")
                    if wh.get("last_triggered"):
                        c2.caption(f"Last: {wh['last_triggered'][:10]}")
                    c3.caption(f"Fails: {wh.get('fail_count', 0)}")
                    c3.caption(f"Events: {', '.join(wh.get('events', []))}")
                    with c4:
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

    # ── API Reference ─────────────────────────────────────────────────────────
    st.markdown("### 📡 API Reference")
    st.caption("Integrate DocIntel into your own systems using these endpoints.")

    with st.expander("View API endpoints"):
        st.markdown(f"""
**Base URL:** `{API_URL}`

**Authentication:** Add header `X-API-Key: your_key` to any request.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document |
| POST | `/query` | Query a document |
| POST | `/query/stream` | Streaming query |
| POST | `/extract` | Extract structured fields |
| POST | `/extract/nl` | Natural language extraction |
| POST | `/extract/batch` | Batch extraction across documents |
| GET | `/documents` | List all documents |
| GET | `/documents/{{id}}/classification` | Get document classification |
| POST | `/documents/{{id}}/classification` | Override classification |
| GET | `/summary/{{id}}` | Get document summary |
| GET | `/tables/{{id}}` | Extract tables |
| GET | `/health` | System health check |
| POST | `/compress` | Compress chat history |
| GET | `/usage` | Session usage stats |
| POST | `/review/{{id}}` | Submit human review |
| GET | `/tasks/{{id}}` | Poll async task status |
| GET | `/templates` | List extraction templates |
| GET | `/templates/{{id}}` | Get a template schema |

**Example — Natural language extraction:**
```bash
curl -X POST {API_URL}/extract/nl \\
  -H "X-API-Key: your_key" \\
  -H "Content-Type: application/json" \\
  -d '{{"document_id": "abc-123", "instruction": "extract name, email and skills"}}'
```
        """)