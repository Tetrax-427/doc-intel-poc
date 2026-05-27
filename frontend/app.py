import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="DocIntel", page_icon="📄", layout="wide")

st.markdown("""
<style>
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 0;
        background: white;
        padding: 1rem 0;
        z-index: 999;
    }
    .stChatMessageContainer {
        padding-bottom: 80px;
    }
    @media (prefers-color-scheme: dark) {
        .stChatFloatingInputContainer {
            background: #0e1117;
        }
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


def load_document(doc_id: str, doc_name: str):
    st.session_state.document_id = doc_id
    st.session_state.file_name = doc_name
    try:
        res = requests.get(f"{API_URL}/chats/{doc_id}")
        if res.status_code == 200:
            st.session_state.messages = res.json()
        else:
            st.session_state.messages = []
    except Exception:
        st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    st.header("📄 DocIntel")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        use_llamaparse = st.toggle("Use LlamaParse (better for tables/scans)", value=True)
        if st.button("Process Document", type="primary"):
            with st.spinner("Ingesting..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    data={"use_llamaparse": str(use_llamaparse)}
                )
                if response.status_code == 200:
                    data = response.json()
                    load_document(data["document_id"], data["file"])
                    st.success(f"✅ {data['chunks_stored']} chunks stored via {data.get('parser', 'pypdf')}")
                    st.rerun()
                else:
                    st.error("Upload failed. Is the API running?")

    st.divider()
    st.subheader("Documents")
    st.session_state.multi_mode = st.toggle("Multi-doc mode", value=st.session_state.multi_mode)

    try:
        docs_response = requests.get(f"{API_URL}/documents")
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
                        if st.button("🗑️", key=f"del_{doc['id']}"):
                            requests.delete(f"{API_URL}/documents/{doc['id']}")
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
                st.caption("No documents yet.")
    except Exception:
        st.caption("API not reachable.")

# --- Main area ---
if st.session_state.multi_mode and st.session_state.selected_doc_ids:
    st.title(f"📚 Multi-doc mode — {len(st.session_state.selected_doc_ids)} document(s) selected")
elif st.session_state.file_name:
    st.title(f"📄 {st.session_state.file_name}")
else:
    st.title("📄 DocIntel — AI Document Intelligence")
    st.info("Upload a document from the sidebar to get started.")
    st.stop()

tab1, tab2 = st.tabs(["💬 Chat", "🗂️ Extract"])

# --- Tab 1: Chat ---
with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📎 Sources"):
                    for s in msg["sources"]:
                        file_info = f" | {s['file']}" if s.get("file") else ""
                        st.caption(f"Chunk {s['chunk']} | Page {s['page']}{file_info}")
                        st.text(s["preview"])

    question = st.chat_input("Ask anything about the document...")

    if question:
        # Build query payload
        is_multi = st.session_state.multi_mode and len(st.session_state.selected_doc_ids) > 1
        if is_multi:
            query_payload = {
                "question": question,
                "document_ids": st.session_state.selected_doc_ids
            }
        else:
            query_payload = {
                "question": question,
                "document_id": st.session_state.document_id
            }

        save_doc_id = st.session_state.selected_doc_ids[0] if is_multi else st.session_state.document_id

        # Show user message
        requests.post(f"{API_URL}/chats/{save_doc_id}",
                      json={"role": "user", "content": question})
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                with requests.post(
                    f"{API_URL}/query/stream",
                    json=query_payload,
                    stream=True,
                    timeout=60
                ) as stream_res:
                    for line in stream_res.iter_lines(chunk_size=1):
                        if line:
                            decoded = line.decode("utf-8")
                            if decoded.startswith("data: "):
                                token = decoded[6:]
                                if token == "[DONE]":
                                    break
                                full_response += token
                                placeholder.markdown(full_response + "▌")
            except Exception:
                pass

            placeholder.markdown(full_response)

            # Fetch sources
            sources = []
            try:
                sources_res = requests.post(f"{API_URL}/query", json=query_payload)
                if sources_res.status_code == 200:
                    sources = sources_res.json().get("sources", [])
                    with st.expander("📎 Sources"):
                        for s in sources:
                            file_info = f" | {s['file']}" if s.get("file") else ""
                            st.caption(f"Chunk {s['chunk']} | Page {s['page']}{file_info}")
                            st.text(s["preview"])
            except Exception:
                pass

            # Save assistant message
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
    st.subheader("Define fields to extract")
    st.caption("Edit the JSON schema below with the fields you want pulled from the document.")

    default_schema = json.dumps({
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "experience": ""
    }, indent=2)

    schema_input = st.text_area("Extraction schema (JSON)", value=default_schema, height=200)

    if st.button("Extract Fields", type="primary"):
        try:
            schema = json.loads(schema_input)
        except json.JSONDecodeError:
            st.error("Invalid JSON schema.")
            st.stop()

        with st.spinner("Extracting..."):
            response = requests.post(
                f"{API_URL}/extract",
                json={"document_id": st.session_state.document_id, "schema": schema}
            )
            if response.status_code == 200:
                result = response.json()
                st.success("Extraction complete!")
                st.json(result["extracted"])
                download_data = json.dumps(result["extracted"], indent=2)
                st.download_button(label="⬇️ Download JSON", data=download_data, file_name="extracted.json", mime="application/json")
            else:
                st.error("Extraction failed.")