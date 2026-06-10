# DocIntel — AI Document Intelligence Platform

A production-ready RAG-powered document intelligence platform. Upload any document, ask questions in natural language, extract structured data with confidence scoring, and integrate with external systems via API and webhooks.

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)
**API:** [your-app.up.railway.app](https://your-app.up.railway.app/docs)

---

## Features

### Document Ingestion
- **Multi-format support** — PDF, DOCX, TXT, CSV, XLSX, RTF, Markdown, PNG, JPG, TIFF, WEBP
- **URL ingestion** — scrape and index any webpage
- **Auto-routing** — scanned PDFs automatically routed to LlamaParse OCR
- **Pluggable parser engine** — LlamaParse, pypdf, python-docx, pandas — all return a unified Document model
- **Vision model support** — GPT-4o / Claude Vision for image descriptions (configurable)

### Intelligent Q&A
- **Hybrid RAG** — BM25 keyword + dense vector search + Cohere reranking + RRF fusion
- **Query expansion** — rewrites vague questions before retrieval
- **Streaming responses** — word-by-word output like ChatGPT
- **Multi-doc querying** — ask questions across multiple documents simultaneously
- **Conversation memory** — remembers previous questions with auto-compression
- **General knowledge fallback** — answers non-document questions from LLM knowledge
- **Query classifier** — detects whether to answer from document or general knowledge

### Extraction Engine
- **Schema-based extraction** — define fields with descriptions, get validated JSON
- **Natural language extraction** — plain English instruction → structured JSON
- **Predefined templates** — CV, Invoice, Loan Application, GST Return, Contract, Offer Letter, Bank Statement
- **Extraction validation** — confidence scores (0–100%) per field with colour coding
- **Format validation** — email, phone, date, amount format checking
- **Business logic validation** — sum checks, date ordering, cross-field rules per document type
- **Correction feedback loop** — human corrections improve future extractions automatically

### Document Intelligence
- **Auto-classification** — detects document type on upload (20 types)
- **Auto-summarization** — structured summary on every upload
- **Table extraction** — renders as interactive bar/line/pie charts
- **Human review** — approve, correct, or reject extracted fields with evidence

### Export & Reporting
- **Chat export** — full conversation as PDF or Word report
- **JSON export** — extracted fields with validation summary
- **CSV export** — table data from charts tab

### Integration & API
- **REST API** — full FastAPI backend, Swagger docs at `/docs`
- **API key management** — generate keys with rate limiting
- **Webhook support** — auto-POST extraction results to any URL
- **Webhook signatures** — HMAC-signed payloads for security
- **Batch extraction** — extract fields from multiple documents at once

### Infrastructure
- **Centralized LLM engine** — swap Groq/OpenAI/Anthropic via `.env`
- **Pluggable vision engine** — swap GPT-4o/Claude Vision via `.env`
- **Retry logic** — auto-retry on rate limits with exponential backoff
- **Caching** — embeddings, classifications, vision descriptions cached with TTL
- **Task queue** — heavy operations run async, API never blocks
- **Usage tracking** — token and call count per session
- **Structured logging** — JSON logs with error codes throughout

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | Groq (LLaMA 3.3 70B) / OpenAI / Anthropic |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Reranking | Cohere rerank-english-v3.0 |
| Vector DB | Supabase pgvector |
| PDF/Image Parsing | LlamaParse + pypdf fallback |
| Vision | GPT-4o Vision / Claude Vision (optional) |
| Validation | Custom rules engine (arithmetic, logic, cross-field) |
| Backend | FastAPI + Python 3.11 |
| Frontend | Streamlit |
| Deploy | Railway (backend) + Streamlit Cloud (frontend) |

---

## Project Structure

```
doc-intel/
├── backend/
│   ├── main.py                  # FastAPI app — router registration only
│   ├── ingestion.py             # Orchestrates parsing, chunking, embedding
│   ├── retrieval.py             # RAG pipeline, query engine, extraction
│   ├── prompts.py               # All LLM prompt templates
│   ├── export.py                # PDF and DOCX report generation
│   ├── webhooks.py              # Outbound webhook delivery + retry
│   ├── api_keys.py              # API key generation and validation
│   ├── db.py                    # Supabase client and table helpers
│   ├── routers/
│   │   ├── documents.py         # Upload, list, delete, summary, classification
│   │   ├── query.py             # Query, stream, chat history
│   │   ├── extraction.py        # Extract, NL extract, templates, review
│   │   ├── export.py            # PDF and DOCX export
│   │   ├── integration.py       # API keys, webhooks
│   │   └── system.py            # Health, usage, tasks
│   ├── core/
│   │   ├── config.py            # Centralized config, startup validation
│   │   ├── logger.py            # Structured JSON logging
│   │   ├── errors.py            # Error taxonomy with codes
│   │   ├── document.py          # Document model dataclasses
│   │   ├── cache.py             # TTL cache for embeddings/vision/classification
│   │   ├── queue.py             # Async task queue
│   │   └── auth.py              # Clerk JWT + API key auth
│   ├── parsers/
│   │   ├── base.py              # BaseParser abstract class
│   │   ├── router.py            # AutoRouter — selects best parser
│   │   ├── llamaparse.py        # LlamaParse (PDFs, images, scanned)
│   │   ├── pypdf.py             # pypdf fallback
│   │   ├── docx_parser.py       # DOCX
│   │   ├── csv_parser.py        # CSV and XLSX
│   │   ├── text_parser.py       # TXT and Markdown
│   │   └── url_parser.py        # URL scraping
│   ├── vision/
│   │   ├── base.py              # BaseVisionModel abstract class
│   │   ├── engine.py            # Vision engine orchestrator
│   │   ├── openai.py            # GPT-4o Vision
│   │   ├── anthropic.py         # Claude Vision
│   │   ├── null.py              # NoVision fallback
│   │   └── triggers.py          # Smart trigger logic
│   ├── validation/
│   │   ├── engine.py            # ValidationEngine
│   │   ├── rules/               # Arithmetic, logic, completeness rules
│   │   └── rulesets/            # Per-doc-type rule sets
│   ├── schemas/
│   │   ├── templates.py         # Extraction templates + vision prompts
│   │   └── validator.py         # Field confidence scoring
│   ├── Procfile                 # Railway start command
│   └── railway.json             # Railway deploy config
├── frontend/
│   └── app.py                   # Streamlit UI (6 tabs)
├── tests/                       # 169 tests passing
├── CONTRACTS.md                 # Shared data format contracts
├── .env                         # API keys — never commit
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/doc-intel.git
cd doc-intel
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
    doc_type text,
    classification_confidence float,
    classification_data jsonb,
    requires_review boolean default false,
    user_id text default 'anonymous',
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

create table review_corrections (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    doc_type text,
    field_name text,
    original_value text,
    corrected_value text,
    action text,
    evidence_used text,
    reviewer_note text,
    created_at timestamp default now()
);

-- Indexes
create index on chunks using ivfflat (embedding vector_cosine_ops);
create index on documents(doc_type);
create index on documents(user_id);
create index on review_corrections(doc_type, field_name);

-- RLS Policies (allow all for now — tighten with Clerk auth later)
alter table documents enable row level security;
create policy "allow_all" on documents for all using (true) with check (true);

alter table chunks enable row level security;
create policy "allow_all" on chunks for all using (true) with check (true);

alter table chats enable row level security;
create policy "allow_all" on chats for all using (true) with check (true);

alter table usage_logs enable row level security;
create policy "allow_all" on usage_logs for all using (true) with check (true);

alter table api_keys enable row level security;
create policy "allow_all" on api_keys for all using (true) with check (true);

alter table webhooks enable row level security;
create policy "allow_all" on webhooks for all using (true) with check (true);

alter table webhook_logs enable row level security;
create policy "allow_all" on webhook_logs for all using (true) with check (true);

alter table review_corrections enable row level security;
create policy "allow_all" on review_corrections for all using (true) with check (true);
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
# LLM — groq | openai | anthropic
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Vision (optional — leave empty to disable)
VISION_PROVIDER=
VISION_MODEL=

# API Keys
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
LLAMA_CLOUD_API_KEY=your_llamaparse_key

# Supabase
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key

# System
UPLOAD_DIR=uploads
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=512
CHUNK_OVERLAP=64
COMPRESSION_THRESHOLD=10
VISION_MIN_WORDS=50
CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7

# Auth (Phase 3 — leave empty for now)
CLERK_SECRET_KEY=
CLERK_PUBLISHABLE_KEY=
```

**Minimum required:** `GROQ_API_KEY` + `SUPABASE_URL` + `SUPABASE_KEY`

### 6. Run locally

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Frontend:**
```bash
streamlit run frontend/app.py
```

- UI: `http://localhost:8501`
- API docs: `http://127.0.0.1:8000/docs`

---

## Switching LLM Providers

No code changes needed — just update `.env`:

```env
# Groq (default, free)
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514

# OpenAI GPT-4o
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

---

## Deployment

### Backend — Railway

1. Connect GitHub repo at [railway.app](https://railway.app)
2. Set root directory: `backend`
3. Add all env variables in Railway Variables tab
4. Deploy — auto-deploys on every push to `main`

### Frontend — Streamlit Cloud

1. Connect repo at [share.streamlit.io](https://share.streamlit.io)
2. Set main file: `frontend/app.py`
3. Add secrets in Settings → Secrets:

```toml
API_URL = "https://your-app.up.railway.app"

[auth]
username = "admin"
password = "your_password"
```

---

## API Reference

### Authentication
```
X-API-Key: your_api_key
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/upload` | Upload a document |
| POST | `/ingest-url` | Ingest from URL |
| GET | `/documents` | List all documents |
| DELETE | `/documents/{id}` | Delete a document |
| GET | `/summary/{id}` | Get document summary |
| GET | `/documents/{id}/classification` | Get classification result |
| POST | `/query` | Query a document |
| POST | `/query/stream` | Streaming query (SSE) |
| POST | `/extract` | Schema-based extraction |
| POST | `/extract/nl` | Natural language extraction |
| POST | `/extract/batch` | Extract from multiple documents |
| GET | `/templates` | List extraction templates |
| POST | `/review/{id}` | Submit human review |
| POST | `/export/pdf` | Export chat as PDF |
| POST | `/export/docx` | Export chat as Word |
| POST | `/api-keys` | Create API key |
| POST | `/webhooks` | Create webhook |
| GET | `/usage` | Session usage stats |
| GET | `/tasks/{id}` | Poll async task status |

### Example — Upload and query

```bash
# Upload
curl -X POST https://your-app.up.railway.app/upload \
  -H "X-API-Key: di_your_key" \
  -F "file=@invoice.pdf"

# Query
curl -X POST https://your-app.up.railway.app/query \
  -H "X-API-Key: di_your_key" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "abc-123", "question": "What is the total amount?"}'

# Natural language extraction
curl -X POST https://your-app.up.railway.app/extract/nl \
  -H "X-API-Key: di_your_key" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "abc-123", "instruction": "extract vendor name and total amount"}'
```

---

## Supported File Types

| Type | Extension | Parser |
|---|---|---|
| PDF (text) | `.pdf` | pypdf |
| PDF (scanned) | `.pdf` | LlamaParse (auto-detected) |
| Word | `.docx` | python-docx |
| Text | `.txt`, `.md` | native |
| Spreadsheet | `.csv`, `.xlsx` | pandas |
| Rich Text | `.rtf` | striprtf |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff` | LlamaParse OCR |
| Web pages | URL | httpx + BeautifulSoup |

---

## Extraction Templates

| ID | Description |
|---|---|
| `cv_resume` | Candidate name, email, phone, skills, experience, education |
| `invoice` | Vendor, client, line items, GST, total |
| `loan_application` | Applicant details, income, loan amount |
| `offer_letter` | Candidate, position, CTC, joining date |
| `gst_return` | GSTIN, period, CGST, SGST, IGST, net payable |
| `contract` | Parties, dates, value, obligations |
| `bank_statement` | Account details, period, balances, credits/debits |

---

## UI Tabs

| Tab | Features |
|---|---|
| 💬 Chat | Streaming Q&A, memory, multi-doc mode, export |
| 🗂️ Extract | Template picker, validated extraction, business rules |
| 🤖 Smart Extract | Plain English → structured JSON |
| 📊 Charts | Auto-detected tables as interactive charts |
| 👤 Review | Approve/correct/reject fields with evidence |
| ⚙️ Settings | API keys, webhooks, API reference |

---

## Roadmap

- [x] Hybrid RAG with Cohere reranking
- [x] Streaming responses
- [x] Multi-doc querying
- [x] Conversation memory + compression
- [x] Centralized LLM engine (switchable providers)
- [x] Pluggable parser engine
- [x] Document model (typed pages, tables, entities)
- [x] Auto document classification (20 types)
- [x] Schema + NL extraction with validation
- [x] Business logic validation (sum, date, cross-field)
- [x] Human review with correction feedback loop
- [x] Predefined schema templates (7 types)
- [x] Vision engine (smart triggering, pluggable)
- [x] Caching layer
- [x] Async task queue
- [x] Document summarization
- [x] Table and chart extraction
- [x] Chat export (PDF + Word)
- [x] Webhook integration + API keys
- [x] Usage tracking
- [x] Deployed (Railway + Streamlit Cloud)
- [ ] Full Clerk auth + multi-tenant
- [ ] Workflow engine per vertical
- [ ] Document comparison
- [ ] Vertical wrappers (CA Helper, CV Screener, Loan Processor)

---

## License

MIT