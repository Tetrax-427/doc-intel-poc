# DocIntel — AI Document Intelligence Platform

A production-ready RAG-powered document intelligence platform. Upload any document, ask questions, extract structured data, and integrate with external systems via API and webhooks.

Built with FastAPI, LlamaIndex, Groq, HuggingFace embeddings, Cohere reranking, and Supabase pgvector.

---

## Features

### Core Intelligence
- **Multi-format ingestion** — PDF, DOCX, TXT, CSV, XLSX, RTF, MD, PNG, JPG, TIFF
- **URL ingestion** — scrape and index any webpage (Wikipedia, articles, docs)
- **Hybrid RAG** — BM25 + dense vector search + Cohere reranking + RRF fusion
- **Query expansion** — rewrites vague questions before retrieval
- **Streaming responses** — word-by-word output like ChatGPT
- **Multi-doc querying** — ask questions across multiple documents simultaneously
- **Conversation memory** — remembers previous questions with auto-compression

### Extraction
- **Schema-based extraction** — define fields + descriptions, get structured JSON
- **Natural language extraction** — plain English instruction → structured JSON
- **Predefined templates** — CV, Invoice, Loan, GST Return, Contract, Offer Letter, Bank Statement
- **Extraction validation** — confidence scores (0-100%) per field, color-coded results
- **Format validation** — email, phone, date, amount format checking

### Analysis
- **Document summarization** — auto-generates on upload, structured overview
- **Table & chart extraction** — detects tables, renders as bar/line/pie charts
- **General knowledge fallback** — answers non-document questions from LLM knowledge

### Export & Integration
- **Chat export** — download full conversation as PDF or Word report
- **Webhook support** — auto-POST extraction results to any URL
- **API key management** — generate keys for external system access
- **Usage tracking** — token and call count per session

### Infrastructure
- **Centralized LLM engine** — swap providers (Groq/OpenAI/Anthropic) via .env
- **Retry logic** — auto-retry on rate limits with exponential backoff
- **LlamaParse integration** — handles scanned PDFs, images, complex tables

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) / OpenAI / Anthropic |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Reranking | Cohere rerank-english-v3.0 |
| Vector DB | Supabase pgvector |
| PDF/Image Parsing | LlamaParse + pypdf fallback |
| Retrieval | Hybrid BM25 + cosine similarity + RRF |
| Backend | FastAPI + Python 3.11 |
| Frontend | Streamlit |

---

## Project Structure

```
doc-intel-poc/
├── backend/
│   ├── main.py              # FastAPI app and all routes
│   ├── ingestion.py         # Multi-format parsing, chunking, embedding
│   ├── retrieval.py         # RAG pipeline, query engine, extraction
│   ├── prompts.py           # All LLM prompt templates
│   ├── export.py            # PDF and DOCX report generation
│   ├── webhooks.py          # Outbound webhook delivery
│   ├── api_keys.py          # API key generation and validation
│   ├── db.py                # Supabase client and helpers
│   ├── llm/
│   │   ├── engine.py        # Centralized LLM caller, model switching
│   │   └── usage.py         # Token tracking and usage logging
│   └── schemas/
│       ├── templates.py     # Predefined extraction schema templates
│       └── validator.py     # Field confidence scoring and validation
├── frontend/
│   └── app.py               # Streamlit UI
├── tests/
│   └── sample_docs/         # Test documents
├── .env                     # API keys (never commit)
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

### 2. Create virtual environment

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
  summary text,
  summary_short text,
  created_at timestamp default now()
);

create table chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  content text,
  embedding vector(384),
  metadata jsonb,
  created_at timestamp default now()
);

create table chats (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  role text,
  content text,
  sources jsonb,
  created_at timestamp default now()
);

create table usage_logs (
  id uuid primary key default gen_random_uuid(),
  call_type text,
  model text,
  prompt_chars int,
  response_chars int,
  estimated_tokens int,
  latency_ms int,
  created_at timestamp default now()
);

create table api_keys (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  key_hash text not null unique,
  key_prefix text not null,
  is_active boolean default true,
  rate_limit int default 100,
  calls_today int default 0,
  last_reset date default current_date,
  created_at timestamp default now()
);

create table webhooks (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  url text not null,
  events text[] default array['extraction.complete'],
  is_active boolean default true,
  secret text,
  last_triggered timestamp,
  fail_count int default 0,
  created_at timestamp default now()
);

create table webhook_logs (
  id uuid primary key default gen_random_uuid(),
  webhook_id uuid references webhooks(id) on delete cascade,
  event text,
  payload jsonb,
  response_status int,
  success boolean,
  created_at timestamp default now()
);

create index on chunks using ivfflat (embedding vector_cosine_ops);
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```
# LLM Provider — groq | openai | anthropic
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# API Keys
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key        # optional
ANTHROPIC_API_KEY=your_anthropic_key  # optional
COHERE_API_KEY=your_cohere_key
LLAMA_CLOUD_API_KEY=your_llamaparse_key

# Supabase
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### 6. Run the servers

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://localhost:8501`

---

## API Reference

### Authentication
Add header to any request:
```
X-API-Key: your_api_key
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document (multipart/form-data) |
| POST | `/ingest-url` | Ingest from URL `{"url": "..."}` |
| POST | `/query` | Query a document |
| POST | `/query/stream` | Streaming query (SSE) |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{id}` | Delete a document |
| GET | `/summary/{id}` | Get document summary |

### Extraction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/extract` | Schema-based extraction |
| POST | `/extract/nl` | Natural language extraction |
| GET | `/templates` | List extraction templates |
| GET | `/templates/{id}` | Get a specific template |

### Integration Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api-keys` | Create API key |
| GET | `/api-keys` | List API keys |
| DELETE | `/api-keys/{id}` | Revoke API key |
| POST | `/webhooks` | Create webhook |
| GET | `/webhooks` | List webhooks |
| DELETE | `/webhooks/{id}` | Delete webhook |
| POST | `/webhooks/{id}/test` | Test webhook delivery |
| GET | `/webhooks/logs` | View delivery logs |
| GET | `/usage` | Get session usage stats |

### Example — Natural Language Extraction

```bash
curl -X POST http://localhost:8000/extract/nl \
  -H "X-API-Key: di_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc-123",
    "instruction": "extract candidate name, email, skills and years of experience"
  }'
```

### Example — Query with history

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: di_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "abc-123",
    "question": "What is the total invoice amount?",
    "history": []
  }'
```

---

## Switching LLM Providers

Change two lines in `.env` — no code changes needed:

```bash
# Use Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514

# Use GPT-4o
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# Use Groq (default)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
```

---

## Vertical Wrappers (Planned)

This base powers the following vertical products:

| Product | Template | Webhook Target |
|---------|----------|----------------|
| CV Screener | `cv_resume` | ATS / HR system |
| Loan Processor | `loan_application` | CRM / LOS |
| CA Helper | `gst_return`, `invoice` | Accounting software |
| PDF Chat | General RAG | — |

---

## Roadmap

- [x] Multi-format ingestion (PDF, DOCX, CSV, XLSX, images)
- [x] Hybrid RAG with reranking
- [x] Streaming responses
- [x] Multi-doc querying
- [x] Conversation memory + compression
- [x] Structured extraction with validation
- [x] Natural language extraction API
- [x] Predefined schema templates
- [x] Document summarization
- [x] Table and chart extraction
- [x] Chat export (PDF + Word)
- [x] Webhook integration
- [x] API key management
- [x] Centralized LLM engine
- [x] Usage tracking
- [ ] Auth + multi-tenant (Clerk)
- [ ] Deploy (Railway + Streamlit Cloud)
- [ ] Document comparison
- [ ] Batch processing API
- [ ] Vertical wrappers (CV, Loan, CA, PDF Chat)

---

## License

MIT