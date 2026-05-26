# DocIntel — AI Document Intelligence POC

A RAG-powered document intelligence API that lets you upload PDFs and ask questions about them — with cited answers and structured extraction.

Built with FastAPI, LlamaIndex, Groq (LLaMA 3.3), HuggingFace embeddings, and Supabase pgvector.

---

## Features

- Upload PDF documents via REST API
- Ask natural language questions about any uploaded document
- Get answers with source citations (chunk + page reference)
- Hybrid search — BM25 keyword + dense vector retrieval with RRF fusion
- Free stack — Groq LLM + HuggingFace embeddings + Supabase free tier

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Vector DB | Supabase pgvector |
| PDF Parsing | pypdf |
| Chunking | LlamaIndex SentenceSplitter |
| Retrieval | Hybrid BM25 + cosine similarity + RRF |
| Backend | FastAPI + Python 3.11 |

---

## Project Structure

```
doc-intel-poc/
├── backend/
│   ├── main.py          # FastAPI app and routes
│   ├── ingestion.py     # PDF parsing, chunking, embedding, storage
│   ├── retrieval.py     # Hybrid search + Groq query engine
│   ├── prompts.py       # LLM prompt templates
│   └── db.py            # Supabase client and helpers
├── frontend/
│   └── app.py           # Streamlit UI (Phase 3)
├── tests/
│   └── sample_docs/     # Test PDFs
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/doc-intel-poc.git
cd doc-intel-poc
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Supabase

Create a free project at [supabase.com](https://supabase.com) and run this in the SQL Editor:

```sql
create extension if not exists vector;

create table documents (
  id uuid primary key default gen_random_uuid(),
  name text,
  created_at timestamp default now()
);

create table chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id),
  content text,
  embedding vector(384),
  metadata jsonb,
  created_at timestamp default now()
);

create index on chunks using ivfflat (embedding vector_cosine_ops);
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 6. Run the server

```bash
cd backend
uvicorn main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`

---

## API Endpoints

### Upload a document
```
POST /upload
Content-Type: multipart/form-data

file: <your PDF>
```

**Response:**
```json
{
  "document_id": "abc-123",
  "file": "invoice.pdf",
  "chunks_stored": 12
}
```

### Query a document
```
POST /query
Content-Type: application/json

{
  "question": "What is the total invoice amount?",
  "document_id": "abc-123"
}
```

**Response:**
```json
{
  "answer": "The total invoice amount is ₹45,000 [1].",
  "sources": [
    {
      "chunk": 1,
      "page": "2",
      "preview": "Total amount due: ₹45,000..."
    }
  ]
}
```

---

## Roadmap

- [x] Phase 1 — PDF ingestion pipeline
- [x] Phase 2 — RAG query engine with hybrid retrieval
- [ ] Phase 3 — Streamlit UI + structured extraction
- [ ] Phase 4 — Auth + multi-tenant support
- [ ] Phase 5 — Deploy to Railway + Streamlit Cloud

---

## License

MIT