import streamlit as st
import requests
import json

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
if "api_error" not in st.session_state:
    st.session_state.api_error = None


def load_document(doc_id: str, doc_name: str):
    st.session_state.document_id = doc_id
    st.session_state.file_name = doc_name
    st.session_state.api_error = None
    try:
        res = requests.get(f"{API_URL}/chats/{doc_id}", timeout=5)
        st.session_state.messages = res.json() if res.status_code == 200 else []
    except Exception:
        st.session_state.messages = []


def check_api():
    try:
        res = requests.get(f"{API_URL}/", timeout=3)
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
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        use_llamaparse = st.toggle("LlamaParse (better for tables/scans)", value=True)
        if st.button("⬆️ Process Document", type="primary", use_container_width=True, disabled=not api_ok):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    response = requests.post(
                        f"{API_URL}/upload",
                        files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                        data={"use_llamaparse": str(use_llamaparse)},
                        timeout=120
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "error" in data:
                            st.error(f"⚠️ {data['error']}")
                        else:
                            load_document(data["document_id"], data["file"])
                            st.success(f"✅ {data['chunks_stored']} chunks · {data.get('parser', 'pypdf')}")
                            st.rerun()
                    else:
                        st.error(f"Upload failed (HTTP {response.status_code})")
                except requests.exceptions.Timeout:
                    st.error("⏱️ Upload timed out. Try a smaller file.")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    st.divider()

    # Documents list
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
                    with cols[1]:
                        if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                            with st.spinner("Deleting..."):
                                requests.delete(f"{API_URL}/documents/{doc['id']}", timeout=5)
                            if st.session_state.document_id == doc["id"]:
                                st.session_state.document_id = None
                                st.session_state.file_name = None
                                st.session_state.messages = []
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
    st.markdown(f"""
    <div class="page-title">
        <div class="icon">📄</div>
        <div class="name">{st.session_state.file_name}</div>
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

tab1, tab2 = st.tabs(["💬 Chat", "🗂️ Extract"])

# --- Tab 1: Chat ---
with tab1:
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
                        file_info = f" · {s['file']}" if s.get("file") else ""
                        st.caption(f"Chunk {s['chunk']} · Page {s['page']}{file_info}")
                        st.text(s["preview"])

    question = st.chat_input("Ask anything about the document...")

    if question:
        if is_multi:
            query_payload = {"question": question, "document_ids": st.session_state.selected_doc_ids}
        else:
            query_payload = {"question": question, "document_id": st.session_state.document_id}

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
                    sources = sources_res.json().get("sources", [])
                    if sources:
                        with st.expander("📎 Sources"):
                            for s in sources:
                                file_info = f" · {s['file']}" if s.get("file") else ""
                                st.caption(f"Chunk {s['chunk']} · Page {s['page']}{file_info}")
                                st.text(s["preview"])
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

# --- Tab 2: Extract ---
with tab2:
    st.markdown("**Define the fields you want to extract from this document.**")
    st.caption("Edit the JSON schema below — keys are field names, values are types or empty strings.")

    default_schema = json.dumps({
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "experience": ""
    }, indent=2)

    schema_input = st.text_area(
        "Extraction schema (JSON)",
        value=default_schema,
        height=220,
        help="Define field names as keys. Use [] for lists, '' for text fields."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        extract_btn = st.button("🗂️ Extract Fields", type="primary", use_container_width=True)

    if extract_btn:
        try:
            schema = json.loads(schema_input)
        except json.JSONDecodeError:
            st.markdown('<div class="error-box">⚠️ Invalid JSON — check your schema format.</div>', unsafe_allow_html=True)
            st.stop()

        with st.spinner("Extracting fields..."):
            try:
                response = requests.post(
                    f"{API_URL}/extract",
                    json={"document_id": st.session_state.document_id, "schema": schema},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    st.markdown('<div class="success-box">✅ Extraction complete</div>', unsafe_allow_html=True)
                    st.json(result["extracted"])
                    download_data = json.dumps(result["extracted"], indent=2)
                    st.download_button(
                        label="⬇️ Download JSON",
                        data=download_data,
                        file_name="extracted.json",
                        mime="application/json"
                    )
                else:
                    st.markdown(f'<div class="error-box">⚠️ Extraction failed (HTTP {response.status_code})</div>', unsafe_allow_html=True)
            except requests.exceptions.Timeout:
                st.markdown('<div class="error-box">⏱️ Extraction timed out. Try a simpler schema.</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="error-box">⚠️ Error: {str(e)}</div>', unsafe_allow_html=True)