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
    .active-doc {
        background: #1a3a5c;
        border-radius: 8px;
        padding: 6px 10px;
        color: white;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .inactive-doc {
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 13px;
        margin-bottom: 4px;
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

def load_document(doc_id: str, doc_name: str):
    st.session_state.document_id = doc_id
    st.session_state.file_name = doc_name
    # Load chat history from DB
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

    # Upload
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        if st.button("Process Document", type="primary"):
            with st.spinner("Ingesting..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                )
                if response.status_code == 200:
                    data = response.json()
                    load_document(data["document_id"], data["file"])
                    st.success(f"✅ {data['chunks_stored']} chunks stored")
                    st.rerun()
                else:
                    st.error("Upload failed.")

    st.divider()
    st.subheader("Documents")

    try:
        docs_response = requests.get(f"{API_URL}/documents")
        if docs_response.status_code == 200:
            docs = docs_response.json()
            if docs:
                for doc in docs:
                    is_active = doc["id"] == st.session_state.document_id
                    cols = st.columns([5, 1])
                    with cols[0]:
                        label = f"{'🟢 ' if is_active else '📄 '}{doc['name']}"
                        if st.button(label, key=f"load_{doc['id']}", use_container_width=True):
                            load_document(doc["id"], doc["name"])
                            st.rerun()
                    with cols[1]:
                        if st.button("🗑️", key=f"del_{doc['id']}"):
                            requests.delete(f"{API_URL}/documents/{doc['id']}")
                            if st.session_state.document_id == doc["id"]:
                                st.session_state.document_id = None
                                st.session_state.file_name = None
                                st.session_state.messages = []
                            st.rerun()
            else:
                st.caption("No documents yet.")
    except Exception:
        st.caption("API not reachable.")

# --- Main area ---
if st.session_state.file_name:
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
                        st.caption(f"Chunk {s['chunk']} | Page {s['page']}")
                        st.text(s["preview"])

    question = st.chat_input("Ask anything about the document...")
    if question:
        # Save + show user message
        requests.post(f"{API_URL}/chats/{st.session_state.document_id}",
                      json={"role": "user", "content": question})
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Query + show assistant message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "document_id": st.session_state.document_id}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])
                    with st.expander("📎 Sources"):
                        for s in data["sources"]:
                            st.caption(f"Chunk {s['chunk']} | Page {s['page']}")
                            st.text(s["preview"])
                    # Save assistant message
                    requests.post(f"{API_URL}/chats/{st.session_state.document_id}",
                                  json={"role": "assistant", "content": data["answer"], "sources": data["sources"]})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data["sources"]
                    })
                else:
                    st.error("Query failed.")

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