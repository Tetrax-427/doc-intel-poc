# DocIntel — Codebase Guide
## Understanding the repository for new team members

> **Read this before touching any code.**
> This document explains every file in the repo, why it exists,
> what it does, and how it connects to everything else.

---

## How the system works (big picture)

A user uploads a document. The system parses it into structured text,
splits it into chunks, converts each chunk into a vector embedding,
and stores everything in Supabase. When the user asks a question,
the system finds the most relevant chunks using hybrid search (keyword + vector),
reranks them, and passes them to an LLM to generate a cited answer.

On top of that engine sits an **agent layer**: multi-stage agent runs
(e.g. a CV-screening agent) that call the same underlying engine APIs to
complete a full task, plus read-only **chat over a completed agent run**
so a user can ask follow-up questions about a run's result without
re-running it.

```
Upload document
      ↓
AutoRouter picks best parser (LlamaParse / pypdf / docx etc.)
      ↓
Parser returns Document object (pages, tables, entities, images)
      ↓
Ingestion splits into chunks → embeds each chunk → stores in Supabase pgvector
      ↓
User asks question
      ↓
Hybrid search (BM25 + vector) → Cohere reranking → top 5 chunks
      ↓
LLM generates answer with citations
      ↓
Answer streams back to UI word by word
```

Separately, an agent run executes a multi-stage plan against these same
engine APIs and produces a `result` object; once that run is
`status == "completed"`, a user can open a 1:1 chat scoped to that run to
ask questions about its result.

---

## Repository layout

```
doc-intel/
├── backend/                 ← FastAPI server — all business logic
│   └── agents/               ← agent runs (stage machine) + agent chat
├── frontend/                ← Streamlit UI
├── tests/                   ← pytest test suite
├── CONTRACTS.md             ← shared data formats agreed between devs
├── MASTER_PLAN.md           ← full architecture and phase plan
├── README.md                ← setup and deployment guide
├── requirements.txt         ← all Python dependencies
└── .env                     ← API keys (never commit this)
```

---

## Backend — file by file

### `backend/main.py`
**What it is:** The FastAPI application entry point.
**What it does:** Creates the app, registers all routers, adds CORS middleware, and runs startup tasks (warming up the embedding model, starting the task queue).
**Why it exists:** FastAPI requires a single app object. Keeping this file thin (under 80 lines) means all route logic lives in the routers, not here.
**Key thing to know:** If you add a new router, register it here with `app.include_router(...)`.

---

### `backend/ingestion.py`
**What it is:** The document processing pipeline.
**What it does:** Takes a file path, routes it to the correct parser via `AutoRouter`, receives a `Document` object back, splits the text into chunks, generates embeddings for each chunk, and stores everything in Supabase. Also handles URL ingestion.
**Why it exists:** Separates the "get text from a file" step from the "store it for search" step.
**Key thing to know:** Every chunk stored in Supabase has a `metadata` field (JSONB) containing `page`, `file`, `chunk_type` (`text`, `table`, or `description`), and `image_ref`. This metadata drives citation display in the UI.

---

### `backend/retrieval.py`
**What it is:** The intelligence engine — search, extraction, classification, and summarization.
**What it does:** Contains hybrid search (BM25 + cosine similarity + RRF), Cohere reranking, query expansion, document classification, field extraction (flat and nested/dynamic), NL extraction, table extraction, summary generation, history compression, and the full RAG query pipeline.
**Why it exists:** All operations that require reading from the vector DB and calling LLMs live here.
**Key thing to know:** This is the largest and most important file. Most new features will either add a function here or call functions from here. Every LLM call goes through `llm/engine.py`, never directly to Groq/OpenAI.
**Dynamic/nested extraction:** `extract_dynamic_fields(document_id, spec, ...)` mirrors `extract_fields()` but takes a `SchemaSpec` (see `schemas/dynamic.py`) instead of a flat `{field: description}` dict — this is how arbitrarily nested extraction (repeating lists of objects, nested objects) works, for any domain, with no per-domain code. It builds the prompt from `spec_to_field_descriptions(spec)`, gets a recursively-built Pydantic model via `spec_to_model(spec)`, calls the LLM with `structured_max_retries=2`, runs `verify_dynamic_extraction()` (computed-duration verification, see `schemas/validator.py`), then the usual `validate_extraction()` confidence scoring and `ValidationEngine().validate(doc_type="dynamic")`. Flat string fields in the response are wrapped as `{"value": ..., "bbox": ...}` same as `extract_fields()`; list/object fields are returned as raw nested JSON — bbox highlighting doesn't apply to nested structures. `extract_nl()` was rewired to always call `generate_schema_spec()` + `extract_dynamic_fields()` — this single change is what makes `/extract/nl` nested-capable, no new endpoint needed. `nl_to_schema()` (old flat-only NL path) is kept only for external callers still importing it directly; no longer used internally.

---

### `backend/prompts.py`
**What it is:** All LLM prompt templates in one place.
**What it does:** Defines string templates for every type of LLM call — QA, multi-doc QA, general knowledge, classification, query expansion, extraction, NL extraction.
**Why it exists:** Keeping prompts out of code files makes them easy to find, compare, and iterate on without touching business logic.
**Key thing to know:** When an answer quality issue occurs, start here. Prompts use `{variable}` placeholders filled with `.format()`. The `{correction_examples}` block in `EXTRACTION_PROMPT` injects past human corrections as few-shot examples.
- `EXTRACTION_SYSTEM_NESTED` — system prompt used for extraction against nested schemas (via `extract_dynamic_fields()`). Separate from flat `EXTRACTION_SYSTEM` because nested extraction has its own failure modes — merging repeated list items into one, mixing up entities across list items, and collapsing a compound phrase into one field instead of splitting it across siblings (e.g. `"M.Tech, CSE"` going entirely into `course` while `branch` stays null). Includes a rule + before/after example for that last case, written generically rather than tied to any one domain.
- `SCHEMA_GEN_SYSTEM` — system prompt for `schemas.dynamic.generate_schema_spec()`. Instructs the LLM to use `type="list"`/`type="object"` with `properties` for repeating/nested structures, and explicitly **not** to include computed/derived fields (durations, totals, growth rates) — those are always computed in Python after extraction, never by the model.

**Note:** the agent-chat system prompt (`AGENT_CHAT_SYSTEM`) does **not** live here — see `backend/agents/chat_prompts.py` below. It's agent-agnostic (reused across every agent's runs) rather than tied to the document-QA prompts in this file.

---

### `backend/db.py`
**What it is:** All Supabase database operations.
**What it does:** Creates the Supabase client, and provides helper functions for every table — insert, select, update, delete. Also handles user scoping (`user_id` filtering).
**Why it exists:** All DB calls go through one file, making it easy to understand what data operations exist and to swap the DB later if needed.
**Key thing to know:**
- `get_all_documents(user_id)` — filters by user so each user sees only their own docs
- `get_chunks_by_document(document_id)` — fetches chunks for a specific doc (used by retrieval)
- `get_all_chunks(user_id)` — fetches all chunks across all user's docs (used by hybrid search)
- Never call `supabase` directly from outside `db.py` — always add a helper function here

**Agent runs — functions:**
- `create_agent_run(agent_name, task, input_data=None, user_id="anonymous", name=None)` — inserts a new `agent_runs` row with `status="pending"`. Takes an optional `name` (display name set at invoke time, purely cosmetic — see `routers/agents.py` below); `NULL` when not supplied.
- `get_agent_run(run_id, user_id)` — selects `*`, so it already includes `name` and needed no change when the column was added.
- `list_agent_runs(user_id="anonymous", agent_name=None, status=None, limit=50)` — selects an **explicit column list** (`id, name, agent_name, task, status, current_stage, created_at, completed_at`) rather than `*`, so `name` had to be added to that list explicitly for the Past Runs list and Chat tab dropdown to display it.

**Agent chat — functions (new):**
- `save_agent_chat_message(run_id, role, content, user_id)` — inserts one row into `agent_chat_messages`.
- `get_agent_chat_history(run_id, user_id)` — selects `role, content, created_at` for a run, ordered oldest-first, scoped by both `run_id` and `user_id`.
- Both are a direct mirror of the existing `chats` table pattern (`save_message` / `get_chat_history`), keyed by `run_id` instead of `document_id` — agent chat is 1:1 with a single agent run, not shared across runs the way document chat can span a document's history.

---

### `backend/export.py`
**What it is:** Report generation.
**What it does:** Takes a conversation history and generates a formatted PDF or DOCX report containing the document summary, all Q&A pairs, and source citations.
**Why it exists:** Users often need to share their findings with colleagues who aren't using DocIntel.
**Key thing to know:** Uses `reportlab` for PDF and `python-docx` for Word. If the report format needs changing, this is the only file to touch.

---

### `backend/webhooks.py`
**What it is:** Outbound webhook delivery.
**What it does:** After an extraction completes, fires HTTP POST requests to configured webhook URLs. Signs payloads with HMAC-SHA256. Retries up to 3 times on failure. Logs every delivery attempt.
**Why it exists:** Allows external systems (CRMs, ATSs, accounting software) to receive extraction results automatically without polling.
**Key thing to know:** Called from extraction endpoints in `routers/extraction.py` after every extract call. The payload shape is defined in `CONTRACTS.md`.

---

### `backend/api_keys.py`
**What it is:** API key management.
**What it does:** Generates secure API keys (prefixed `di_`), hashes them for storage, validates incoming keys, tracks daily usage counts, and enforces rate limits.
**Why it exists:** External systems that call the API programmatically use API keys instead of user JWTs.
**Key thing to know:** The full key is only shown once at creation — never stored in plain text. Only the SHA-256 hash is in the DB. Rate limit resets daily.

---

## Backend — routers/

Each router file handles a group of related endpoints. They're all registered in `main.py`.

---

### `backend/routers/auth.py`
**Endpoints:** `POST /auth/signup`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`
**What it does:** Handles user account creation and login using Supabase Auth. Returns a JWT on success. Signup is auto-confirmed — no email verification required. The `/me` endpoint is called by the frontend on every page load to check if the session is still valid.
**Why it exists:** Keeps all auth-related routes together, separate from business logic.
**Key thing to know:** Uses `SUPABASE_SERVICE_KEY` for signup (to bypass email confirmation). Uses `SUPABASE_KEY` (anon key) for login. If `SUPABASE_JWT_SECRET` is not set in env, the system runs in dev mode — no login required.

---

### `backend/routers/documents.py`
**Endpoints:** `POST /upload`, `POST /ingest-url`, `GET /documents`, `DELETE /documents/{id}`, `GET /summary/{id}`, `GET /documents/{id}/classification`, `POST /documents/{id}/classification`
**What it does:** Handles everything related to document lifecycle — uploading, listing, deleting, and reading metadata. After upload, triggers classification and summary generation.
**Why it exists:** Document management is a distinct concern from querying or extracting.
**Key thing to know:** The upload flow: save file → `ingest_file()` → classify → generate summary → return result. Classification failure is non-blocking — returns `"general"` as fallback.

---

### `backend/routers/query.py`
**Endpoints:** `POST /query`, `POST /query/stream`, `GET /chats/{document_id}`, `POST /chats/{document_id}`, `POST /compress`
**What it does:** Handles Q&A — standard and streaming. Also manages persistent chat history (save/retrieve per document) and history compression.
**Why it exists:** Chat is the core user interaction — it deserves its own router.
**Key thing to know:** `/query/stream` uses Server-Sent Events (SSE). The frontend consumes it by iterating `response.iter_lines()`. Tokens are base64-encoded in the stream to preserve newlines. Sources are fetched separately via `/query` after streaming completes.

---

### `backend/routers/extraction.py`
**Endpoints:** `POST /extract`, `POST /extract/nl`, `POST /extract/batch`, `GET /templates`, `GET /templates/{id}`, `GET /tables/{id}`, `POST /review/{id}`, `GET /review/{id}/corrections`, `GET /schemas/example`
**What it does:** All extraction operations — schema-based (flat and nested), natural language, batch, and human review. Also serves extraction templates and table data.
**Why it exists:** Extraction is the highest-value feature — grouping all extraction routes makes the API surface clear.
**Key thing to know:** `POST /extract` and `POST /extract/nl` both trigger webhooks after completion. The correction feedback loop is in `retrieval.build_correction_examples()` — past human corrections are injected as few-shot examples into the extraction prompt (currently top-level field names only; sub-fields inside nested list items aren't individually correctable yet).
**Nested/dynamic extraction:** `ExtractRequest` takes an optional inline `nested_schema: dict` alongside the existing flat `fields: dict` — not a separate endpoint, so one route handles both shapes and both return the same response contract (`extracted` / `validation` / `business_validation` / `extraction_id`). Named `nested_schema` rather than `schema` to avoid colliding with Pydantic v1's reserved `BaseModel.schema()`. A validator requires exactly one of `fields` / `nested_schema`. `extract()` branches on which is present: `nested_schema` → validated as a `SchemaSpec` and passed to `retrieval.extract_dynamic_fields()`; otherwise → the existing flat `extract_fields()` path. `NLExtractRequest.preview_only` now returns a full `SchemaSpec` (via `generate_schema_spec`) instead of the old flat `SchemaResult`, so a preview matches what the real call will produce, nested fields included. `GET /schemas/example` returns a worked nested-schema example for API consumers building a schema by hand.

---

### `backend/routers/export.py`
**Endpoints:** `POST /export/pdf`, `POST /export/docx`
**What it does:** Generates and returns report files from chat history.
**Why it exists:** Separated to keep export logic isolated from chat logic.
**Key thing to know:** Returns binary file content directly with appropriate `Content-Disposition` headers. Frontend uses `st.download_button` with the raw bytes.

---

### `backend/routers/integration.py`
**Endpoints:** `POST /api-keys`, `GET /api-keys`, `DELETE /api-keys/{id}`, `POST /webhooks`, `GET /webhooks`, `DELETE /webhooks/{id}`, `POST /webhooks/{id}/test`, `GET /webhooks/logs`
**What it does:** Manages API keys and webhook configurations. The test endpoint sends a ping to verify a webhook URL is reachable.
**Why it exists:** Integration features (keys + webhooks) are for technical users and external systems — kept separate from document and query routes.

---

### `backend/routers/agents.py`
**Endpoints:** `POST /agents/{agent_name}/invoke` (invoke), a resume endpoint for paused runs, `GET /agents/runs` (list, via `list_agent_runs`), `GET /agents/runs/{run_id}`, plus (new) `POST /agents/runs/{run_id}/chat` and `GET /agents/runs/{run_id}/chat`.
**What it does:** Invokes and resumes multi-stage agent runs, lists/reads past runs, and (new) handles read-only Q&A chat scoped to a single completed run.
**Why it exists:** Agent orchestration is a distinct concern from document upload/query/extraction — it composes those engine APIs into multi-stage tasks rather than exposing a single call.
**Key thing to know:**
- All request models (`InvokeAgentRequest`, `ResumeAgentRequest`, and the new `ChatMessageRequest`) use Pydantic v2's `@field_validator` — migrated together from the deprecated `@validator` for consistency rather than leaving a mix of styles in the file.
- `InvokeAgentRequest` gained an optional `name: str | None` field — lets the caller label a run at invoke time (e.g. "Backend Engineer — July batch") instead of only ever seeing a UUID. This is the only place naming is set — no rename-later UI. `invoke_agent()` passes `req.name` straight through to `agent_base.create_and_queue_run(...)`.
- **`POST /runs/{run_id}/chat`** is deliberately **synchronous** — not routed through `task_queue.submit()` like invoke/resume. v1 chat is exactly one `call_llm()` call (see `backend/agents/chat.py`), so there's no multi-stage work to background; the reply comes back in the same response. Delegates to `agents.chat.handle_chat_message()` and returns `400` with code `AGENT_CHAT_UNAVAILABLE` if the run doesn't exist, isn't owned by the caller, or isn't `status == "completed"` yet — all three are "you can't chat here right now" from the caller's perspective, so they're handled uniformly via `agent_chat.AgentChatError` rather than split across 400/404/500.
- **`GET /runs/{run_id}/chat`** returns `{"messages": [...]}`, oldest first, via `get_agent_chat_history`. Returns `404` (not the chat-specific 400) if the run itself doesn't exist/isn't owned — that's a genuine not-found, distinct from "exists but chat isn't available yet."

---

### `backend/routers/system.py`
**Endpoints:** `GET /`, `GET /health`, `GET /usage`, `GET /tasks/{id}`
**What it does:** System status and monitoring. Health check verifies DB connection and embedding model. Usage returns token counts. Tasks endpoint lets the UI poll for async operation completion.
**Why it exists:** Operational endpoints that don't belong to any feature category.
**Key thing to know:** `GET /health` is used by Railway as a healthcheck. If it returns non-200, Railway restarts the container. Always keep this endpoint fast and reliable.

---

## Backend — core/

The `core/` directory contains shared infrastructure used by everything else.

---

### `backend/core/config.py`
**What it is:** Centralized configuration.
**What it does:** Reads all settings from environment variables, validates required keys on startup, and exposes a single `config` object imported everywhere.
**Why it exists:** Before this existed, `os.getenv()` was scattered throughout the codebase. Now there's one place to see every setting and add new ones.
**Key thing to know:** If the server fails to start with a `ValueError`, a required config key is missing. Check `.env`. Never call `os.getenv()` directly outside this file — import `config` instead.

---

### `backend/core/logger.py`
**What it is:** Structured logging.
**What it does:** Provides a `StructuredLogger` class that outputs JSON log lines with consistent fields — timestamp, level, logger name, message, and any keyword arguments passed.
**Why it exists:** JSON logs are searchable and parseable in Railway's log viewer. Plain `print()` statements are not.
**Key thing to know:**
```python
logger = get_logger("ingestion")
logger.info("File parsed", file="invoice.pdf", pages=3, duration_ms=450)
# Outputs: {"timestamp": "...", "level": "INFO", "logger": "ingestion", "message": "File parsed", "file": "invoice.pdf", ...}
```
No `print()` statements anywhere in the codebase. Always use `get_logger()`. `backend/agents/chat.py` uses `get_logger("agents.chat")` and logs only on the empty-reply failure path.

---

### `backend/core/errors.py`
**What it is:** Error taxonomy.
**What it does:** Defines a base `DocIntelError` exception with `code`, `severity`, and `retryable` fields, plus 6 typed subclasses — `ParseError`, `ClassificationError`, `EmbeddingError`, `VisionError`, `ConfigError`, etc.
**Why it exists:** Typed errors with codes make debugging much faster. When a `PARSE_001` appears in logs, you know exactly what failed and where.
**Key thing to know:** Error codes by prefix:
- `PARSE_0xx` — file parsing
- `CLASS_0xx` — classification
- `EMBED_0xx` — embeddings
- `RETRIEV_0xx` — search
- `EXTRACT_0xx` — extraction
- `VISION_0xx` — vision model
- `CONFIG_0xx` — config

Agent chat does **not** use this taxonomy — it raises its own `agents.chat.AgentChatError`, mapped by the router to a single `AGENT_CHAT_UNAVAILABLE` code rather than a `PARSE_0xx`-style prefix, since it's one uniform "not available right now" condition rather than several distinct failure classes.

---

### `backend/core/document.py`
**What it is:** The Document model — the central data contract.
**What it does:** Defines 8 dataclasses that represent a parsed document in a structured way: `Entity`, `TableCell`, `Table`, `LayoutElement`, `ImageElement`, `DocumentPage`, `Classification`, `Document`.
**Why it exists:** Before this, every parser returned a different format — raw text, list of dicts, etc. Now every parser returns a `Document` and everything downstream works the same way regardless of file type.
**Key thing to know:** This is the most important file in `core/`. The `Document` object is the contract between parsers and the rest of the system. Do not change the field names without updating every parser and consumer. When you add a new parser, it must return a `Document`.

---

### `backend/core/auth.py`
**What it is:** JWT verification middleware.
**What it does:** Verifies incoming JWTs using `supabase.auth.get_user(token)` — lets Supabase handle its own token verification regardless of algorithm (ES256 or HS256). Also handles API key auth. Returns a user dict with `user_id` and `email`.
**Why it exists:** Centralizes auth so every router can use `Depends(get_current_user)` without duplicating verification logic.
**Key thing to know:** Dev mode — if `SUPABASE_JWT_SECRET` is not set, returns `{"user_id": "dev_user", "anonymous": True}` for every request. This allows local development without auth setup. The `get_user_id(user)` helper safely extracts `user_id` from any user dict. `routers/agents.py` uses the same `Depends(get_current_user_context)` pattern for the chat routes.

---

### `backend/core/cache.py`
**What it is:** In-memory TTL cache.
**What it does:** Caches expensive repeated operations with automatic expiry. Three domains: embeddings (24h TTL), vision descriptions (7 days TTL), classifications (1h TTL).
**Why it exists:** Generating embeddings and calling vision models costs time and money. If the same text or image is processed twice, the cache prevents redundant API calls.
**Key thing to know:** This is in-process memory — the cache is lost on server restart. That's fine for the current single-server setup. For multi-server production, replace with Redis. Cache keys are MD5 hashes of the input content.

---

### `backend/core/queue.py`
**What it is:** Async task queue for heavy operations.
**What it does:** Runs heavy operations (large file ingestion, vision processing, agent run stages) in background threads. Callers get a `task_id` immediately and poll `GET /tasks/{task_id}` for completion.
**Why it exists:** File ingestion can take 30-90 seconds for large documents. Running it synchronously would block the API server and make the UI appear frozen.
**Key thing to know:** Uses Python's `threading` and `queue.Queue` — no external dependencies. 3 worker threads by default. For multi-server production, replace internals with Celery + Redis while keeping the same `task_queue.submit()` interface. Agent invoke/resume both go through this queue; agent **chat** deliberately does not (see `backend/agents/chat.py` below) — it's a single synchronous LLM call, not multi-stage work.

---

## Backend — agents/

The `agents/` directory holds the multi-stage agent run machinery and,
separately, read-only chat over a completed run's result. It sits above
the document engine (`ingestion.py` / `retrieval.py` / routers) and
composes those APIs rather than duplicating them.

---

### `backend/agents/base.py`
**What it is:** The stage-machine layer that creates and queues agent runs.
**What it does:** `create_and_queue_run(agent_name, task, input_data, user_id="anonymous", name=None)` is the fast, synchronous entry point — it creates the `agent_runs` row via `db.create_agent_run()` and returns `run_id` immediately, **without** executing any stages. Callers (typically a router) are expected to hand `execute_stages()` to `core.queue.task_queue.submit()` right after this returns, so the HTTP response doesn't block on the run.
**Why it exists:** Separates "register that a run exists" (fast, must return quickly to the caller) from "actually run the stages" (slow, backgrounded).
**Key thing to know:** `name` is optional and purely cosmetic — it's threaded straight through to `create_agent_run()` and never read by the stage machine itself. The UI falls back to showing the run's `id` when `name` is `None`.

---

### `backend/agents/chat.py` (new)
**What it is:** The Q&A chat handler for a completed agent run.
**What it does:** `handle_chat_message(run_id, message, user_id)` answers one chat turn grounded in a completed `agent_runs` row's `result`, persists both the user message and the assistant reply on success, and returns `{"role": "assistant", "content": reply}`.
**Why it exists:** Deliberately **not** part of `base.py`'s stage machine — v1 chat is read-only Q&A over a *finished* run and can't mutate it, so it doesn't belong in the pause/resume/stage-loop model at all. No actions or tool-calling in v1; every reply is a single `call_llm()` text call.
**Key thing to know:**
- **Availability check lives here, not in the router** — `AgentChatError` is raised if the run doesn't exist / isn't owned by `user_id`, or `status != "completed"`. Enforcing this in the handler (not the route) means any future caller of `handle_chat_message()` gets the same guarantee, not just the HTTP route.
- **Context source is `result`, not `state`.** `state` carries every sub-agent's raw intermediate output (e.g. full per-candidate CSV data, every scoring pass) and was large enough on its own to blow a fallback model's TPM limit in testing (a single turn requested ~8100 tokens against a 6000–8000 TPM cap and was rejected with a 413 by every provider in the chain). `result` is the fixed, already-compact output contract (`summary` / `findings` / `data` / `extra`, where `extra` already carries things like the evaluation plan and full per-criterion scores), so it's sufficient context on its own — see `_format_run_context()`.
- **`MAX_CONTEXT_CHARS = 6000`** — a hard character cap (`_truncate()`) on the formatted run-context block, as a safety net against `result` growing large on a bigger batch, independent of the `state`-vs-`result` choice above.
- **`MAX_HISTORY_TURNS = 20`** — older turns beyond this are dropped from the *prompt* (not from storage) via `_format_history()`, to bound prompt size on long-running chats. No summarization/compaction in v1.
- **No multi-turn message array.** `llm/engine.call_llm()` only accepts a single `system` + `user` string pair — there's no `messages=[...]` primitive lower in the stack to hand this off to — so prior turns are folded into the `user` string as a plain `"User: ...\nAssistant: ..."` transcript before the new message.
- **Empty-reply handling:** if `call_llm()` returns an empty/whitespace string, `handle_chat_message()` logs the failure and raises `AgentChatError` instead of silently saving a blank assistant message — a failed LLM call surfaces as a clear error, not a blank chat bubble.
- `call_llm(..., call_type="agent_chat", user_id=user_id, session_id=run_id)` — tagged distinctly for usage tracking (`llm/usage.py`) and traced by `run_id` as the session.

---

### `backend/agents/chat_prompts.py` (new)
**What it is:** The system prompt for agent chat.
**What it does:** Defines `AGENT_CHAT_SYSTEM`, the single system prompt used by `agents/chat.py`.
**Why it exists:** Kept in its own file rather than folded into a per-agent prompts file (e.g. `cv_processor/prompts.py`) because chat is **agent-agnostic** — the same prompt is reused for every agent's runs (`cv_processor` today, `ca_helper` later), unlike per-agent stage prompts which are specific to one agent's task.
**Key thing to know:** The prompt explicitly constrains answers to the provided context only (no invented candidates/scores/facts) and tells the model it cannot take actions in v1 — so if a user asks mid-chat for a re-rank or reweighting, the model itself declines and points them to starting a new agent run, rather than the backend silently attempting something unsupported.

---

## Backend — parsers/

The `parsers/` directory implements the pluggable parser engine. Every parser takes a file and returns a `Document`.

---

### `backend/parsers/base.py`
**What it is:** The abstract base class for all parsers.
**What it does:** Defines 4 methods every parser must implement: `can_handle(file_path)`, `parse(file_path, config)`, `get_name()`, `is_available(config)`.
**Why it exists:** The `AutoRouter` only talks to `BaseParser` — it never knows which specific parser is running. This allows adding new parsers without changing any other code.

---

### `backend/parsers/router.py`
**What it is:** The AutoRouter — selects the best available parser for each file.
**What it does:** Given a file path, decides which parser to use based on file extension, whether the PDF is scanned (text extraction test), and which parsers have their API keys configured. Falls back gracefully if the preferred parser is unavailable.
**Routing logic:**
```
Images (.png, .jpg etc.)     → LlamaParse
Scanned PDF (< 50 words/pg)  → LlamaParse
Complex PDF (tables found)   → LlamaParse
Simple text PDF              → pypdf (faster, cheaper)
DOCX                         → DocxParser
CSV / XLSX                   → CsvParser
TXT / MD                     → TextParser
URL                          → UrlParser
No API key available         → pypdf fallback
```

---

### `backend/parsers/llamaparse.py`
**What it is:** LlamaParse cloud parser.
**What it does:** Sends the file to LlamaParse API, receives markdown output, extracts markdown tables into structured `Table` objects, returns a `Document`.
**Why it exists:** LlamaParse handles complex layouts, scanned documents, and images far better than pypdf.
**Key thing to know:** Requires `LLAMA_CLOUD_API_KEY`. Free tier gives 1000 pages/day. Falls back to pypdf if the key is missing or the API fails.

---

### `backend/parsers/pypdf.py`
**What it is:** Local PDF text extractor.
**What it does:** Extracts text page by page using pypdf. No API call, no cost, instant.
**Why it exists:** For simple text-based PDFs, pypdf is faster and free. No need to use LlamaParse for a plain text PDF.
**Key thing to know:** Does not handle scanned PDFs — returns empty or garbled text. The AutoRouter detects scanned PDFs and routes them to LlamaParse automatically.

---

### `backend/parsers/docx_parser.py`
**What it is:** Word document parser.
**What it does:** Extracts paragraphs and tables from DOCX files using `python-docx`. Tables are converted to structured `Table` objects.

### `backend/parsers/csv_parser.py`
**What it is:** Spreadsheet parser.
**What it does:** Reads CSV and XLSX files using pandas. Converts rows to readable text in batches of 50 rows per chunk. Column headers are preserved.

### `backend/parsers/text_parser.py`
**What it is:** Plain text and Markdown parser.
**What it does:** Reads TXT and MD files directly. Splits into 3000-character chunks.

### `backend/parsers/url_parser.py`
**What it is:** Web page ingestion.
**What it does:** Fetches the URL with httpx, strips nav/footer/scripts with BeautifulSoup, extracts main content. Has a special handler for Wikipedia that uses their official API instead of scraping.

---

## Backend — vision/

The `vision/` directory implements pluggable vision model support.

---

### `backend/vision/base.py`
**What it is:** Abstract base class for vision models.
**What it does:** Defines one method: `describe(image_path, prompt) → str`. Must never raise — returns empty string on any failure.

### `backend/vision/null.py`
**What it is:** The no-op fallback.
**What it does:** Returns `""` always. Used when no vision model is configured.
**Why it exists:** The system must work identically whether or not a vision model is configured. `NoVisionModel` makes that true — callers never need to check "is vision configured?" before calling `describe()`.

### `backend/vision/openai.py` and `backend/vision/anthropic.py`
**What they do:** Call GPT-4o Vision or Claude Vision respectively. Base64-encode the image, send with a prompt, return the description. Log timing and usage.

### `backend/vision/triggers.py`
**What it is:** Smart triggering logic.
**What it does:** `should_use_vision(file_path, page_text, is_scanned, doc_type, config)` returns `True` only when vision genuinely adds value — image files, scanned pages, visual doc types, or very low word count pages.
**Why it exists:** Vision API calls cost money. This file ensures they're only made when necessary.

### `backend/vision/engine.py`
**What it is:** Vision orchestrator.
**What it does:** Selects the right vision model from config (or NoVision), checks the trigger conditions, optionally caches results, and provides `describe_image()` and `describe_pdf_page()` as the public API.

---

## Backend — validation/

---

### `backend/validation/engine.py`
**What it is:** The validation orchestrator.
**What it does:** Given extracted fields and a doc type, loads the appropriate ruleset, runs all rules, and returns a structured report (passed, failed, blocking failures, per-rule results).
**Key thing to know:** The response shape matches Contract 4 in `CONTRACTS.md` exactly — this is what Dev 2's UI reads. `RULESET_MAP` includes `"dynamic": "validation.rulesets.generic"` — `retrieval.extract_dynamic_fields()` always passes `doc_type="dynamic"`, since a user-defined schema's field names aren't known in advance and the doc-type-specific rulesets (`invoice.py`, `cv_resume.py`, etc.) can't apply to it.

### `backend/validation/rules/`
**Four rule files:**
- `base.py` — `BaseRule` abstract class and `ValidationResult` dataclass
- `arithmetic.py` — `SumRule` (sum of line items = total), `TotalConsistencyRule` (subtotal + tax = total)
- `logic.py` — `DateOrderRule` (start before end), `ConditionalRequiredRule` (if field A then field B required)
- `completeness.py` — required fields per doc type

### `backend/validation/rulesets/`
**One file per document type** — `invoice.py`, `cv_resume.py`, `gst_return.py`, `contract.py`, `bank_statement.py`, `loan_application.py`. Each exports a `get_rules()` function that returns a list of rule instances. Adding a new rule to invoice validation means editing only `invoice.py`.

**`generic.py`** — the ruleset for `doc_type="dynamic"` (nested/dynamic extraction), schema-agnostic since field names aren't known ahead of time:
- `ListItemConsistencyRule` — flags list items missing sub-fields that other items in the same list have populated (catches partial-extraction degradation on long lists).
- `TopLevelCompletenessRule` — flags entirely-empty top-level fields.
- `DateVerificationMismatchRule` — surfaces items where `_verification.match` is explicitly `false` (a stated duration disagreed with the computed one, see `schemas/validator.py`).

All three exclude the injected `_verification` block from their own checks, so verification metadata never gets mistaken for a missing extracted field. Rules are built against the real `ValidationResult` dataclass in `validation/rules/base.py` (`field`, `rule`, `rule_code`, `status`, `expected`, `actual`, `message`, `severity`, `blocking`) — a mismatched constructor here fails **silently** (`ValidationEngine._load_rules()`'s try/except swallows the `TypeError` per rule), so double-check the signature against `base.py` directly rather than guessing when adding a rule.

---

## Backend — schemas/

---

### `backend/schemas/templates.py`
**What it is:** Extraction schema library.
**What it does:** Contains `TEMPLATES` (7 predefined schemas with field descriptions), `TEMPLATE_MAP` (maps doc type strings to template IDs), `VISION_PROMPTS` (per-doc-type vision prompts), and helper functions `get_template_for_doc_type()`, `get_vision_prompt()`.
**Why it exists:** Templates are data, not code. Keeping them in one file makes adding or editing a template a 10-minute job that doesn't touch any business logic.

### `backend/schemas/dynamic.py`
**What it is:** The recursive meta-schema DSL that powers nested/dynamic extraction.
**What it does:** Describes any extraction schema — flat or arbitrarily nested — as data, not code, so a user-submitted schema and an NL-generated schema go through the exact same extraction path with no per-domain branching.
- `FieldSpec` — recursive Pydantic model: `name`, `type` (`string` / `number` / `integer` / `date` / `boolean` / `list` / `object`), `description`, and optional `properties: list[FieldSpec]` for `list`/`object` types describing the nested shape. Calls `model_rebuild()` since it's self-referencing.
- `SchemaSpec` — `schema_name` + `fields: list[FieldSpec]`. The top-level object both an inline `nested_schema` request and an NL-generated schema conform to.
- `generate_schema_spec(description, user_id)` — calls the LLM (`response_model=SchemaSpec`) to turn a free-text instruction into a `SchemaSpec`, using `prompts.SCHEMA_GEN_SYSTEM`.
- `spec_to_model(spec)` — recursively builds a real Pydantic model via `pydantic.create_model`: a `list` field with `properties` becomes `list[<recursively built item model>]`, an `object` field becomes `Optional[<recursively built model>]`, everything else maps to a primitive type. Results are cached in-process, keyed by an MD5 hash of the spec's JSON, so reusing the same schema doesn't rebuild the model every call.
- `spec_to_field_descriptions(spec)` — renders a `SchemaSpec` as an indented, human-readable field list for the extraction prompt (the nested equivalent of the old flat `fields_with_desc` string).
**Why it exists:** Before this, every "list of structured items" use case needed a hardcoded Pydantic model and a hardcoded prompt. This makes nesting a schema property instead of a code change.
**Key thing to know:** Building the Pydantic model at runtime is what lets Instructor validate arbitrarily deep nesting the same way it validates a flat model — the LLM structured-output guarantee extends to nested shapes for free.

### `backend/schemas/validator.py`
**What it is:** Field-level confidence scoring, plus (new) nested-schema confidence and date-verification logic.
**What it does:** For each extracted field, detects the field type (email, phone, date, amount, text, list), validates the format, scores confidence 0.0–1.0, and returns `FOUND` / `LOW_CONFIDENCE` / `NOT_FOUND` status.
**Why it exists:** Separate from the business logic validation engine — this runs on every extraction regardless of doc type. The business validation in `validation/` runs on top of this.
**Key thing to know:**
- `detect_field_type()` and `detect_verifiable_pairs()` use **token-based** matching (`name.split("_")`, set intersection) rather than substring matching — a substring check once misclassified `candidate_name` as a `date` field because `"date"` is a substring of `"candi-date"`. Don't reintroduce substring matching here.
- **Nested confidence scoring** — `score_nested_list_confidence()` (average non-null-field ratio across list items) and `score_nested_object_confidence()` (non-null ratio of a nested object's sub-fields), dispatched from `score_confidence()` based on value shape, so a half-populated list and a fully-populated list no longer score identically. `validate_extraction()` only runs format validation (email/phone/date/amount/url regex) against flat string values — nested values go straight to shape-based confidence.
- **Verification (computed, not extracted)** — `compute_duration(start, end)` does deterministic date math in Python (never the LLM); missing `end` is treated as "ongoing" against today. `detect_verifiable_pairs(item_fields)` finds a start/end pair (and optional stated-duration field) in a list item by name convention (`start`/`end`/`duration`/`tenure`/`time_spent` tokens) — types `date`/`string`/`integer`/`number` are all eligible, booleans and lists are not. `verify_item(item, pair)` computes the duration and compares it to any stated value. `verify_dynamic_extraction(extracted, spec)` is the orchestrator, called from `retrieval.extract_dynamic_fields()` right after the LLM returns.
- **Backfill** — if a stated-duration field is empty and a duration was computed, the computed value is written into that field and `_verification.backfilled = true` — never overwrites a stated value (a disagreement is surfaced as `"match": false` instead). Verification failures/gaps are silent-safe — they annotate `_verification`, never block or crash extraction.
- Date parsing handles resume-style formats like `"Sept'25"` via an explicit `_MONTH_MAP` + regex matcher in `_parse_date_loose()`, tried before the fixed `_PARSE_FORMATS` list — `dateutil.parser(fuzzy=True)` was tried and rejected because it silently misparses `"Sept'25"` as day=25 of the current year.

---

## Backend — llm/

---

### `backend/llm/engine.py`
**What it is:** The central LLM caller — every LLM call in the system goes through here.
**What it does:** Selects the right client (Groq, OpenAI, Anthropic) from config, calls the API, handles retries with exponential backoff on rate limits, parses JSON responses when `json_mode=True`, and supports streaming.
**Why it exists:** Before this, every function had its own Groq client and its own retry logic. Now there's one place to swap models, add logging, or change retry behaviour.
**Key thing to know:** To switch LLM providers, change `LLM_PROVIDER` in `.env`. No code changes needed. The `call_llm()` function signature is the same regardless of provider. `call_llm()` / `_call_single_provider()` also thread an optional `structured_max_retries` parameter down to `llm/structured.py`'s `call_structured()`. Default is `0` (no behavior change) everywhere except `retrieval.extract_dynamic_fields()`, which passes `2` — nested/list-of-object shapes fail Instructor's validation on the first pass more often than a flat model, so letting Instructor self-correct with the validation error before falling back to the next provider improves reliability without extra app-level retry logic. `call_llm()` accepts a single `system` + `user` string pair only — no `messages=[...]` array — which is why `agents/chat.py` folds prior chat turns into the `user` string as plain text rather than passing structured history.

### `backend/llm/structured.py`
**What it is:** Instructor-based structured-output calling.
**What it does:** `call_structured()` wraps Instructor's `.create()` to get a validated Pydantic object back from an LLM call (used for `SchemaSpec` generation and for dynamic-schema extraction). Takes a `max_retries` param passed straight to Instructor's own retry mechanism, which re-prompts the model with the validation error on failure.

### `backend/llm/usage.py`
**What it is:** Token usage tracking.
**What it does:** Logs every LLM call to an in-memory buffer and writes to the `usage_logs` Supabase table. Provides `get_usage_summary()` for the `/usage` endpoint.
**Why it exists:** You need to know what things cost before you can charge for them.
**Key thing to know:** Agent chat calls are logged with `call_type="agent_chat"` and `session_id=<run_id>`, so usage/cost for chat can be broken out from document Q&A and extraction calls.

---

## Frontend — `frontend/app.py`

**What it is:** The entire Streamlit frontend in one file.
**What it does:** Renders the full application — auth gate, sidebar (upload, document list, usage stats), and 6 tabs (Chat, Extract, Smart Extract, Charts, Review, Settings).
**Why it's one file:** Streamlit doesn't support multi-file apps natively in the same way as React. The tradeoff is a large single file vs complex workarounds.

**Key sections (in order):**
1. **CSS block** — entire dark theme as a `st.markdown` HTML block
2. **Auth functions** — `_auth_headers()`, `_do_login()`, `_do_signup()`, `_do_logout()`, `_verify_session()`, `_check_backend()`, `_render_auth_page()`
3. **Auth gate** — checks session on every load, shows login if needed
4. **Session state init** — all `st.session_state` keys initialized
5. **Helper functions** — `load_document()`, `render_sources()`, `_show_upload_success()`
6. **Sidebar** — brand, upload, URL ingest, document list, usage stats, sign out
7. **Main area** — page title with doc type badge, empty state
8. **Tab 1 — Chat** — message history, chat input, streaming, sources
9. **Tab 2 — Extract** — template picker, schema editor, confidence display, business validation
10. **Tab 3 — Smart Extract** — NL instruction, preview, extract
11. **Tab 4 — Charts** — table detection, chart rendering
12. **Tab 5 — Review** — approve/correct/reject with evidence
13. **Tab 6 — Settings** — API keys, webhooks, API reference

**Key thing to know:** All API calls use `headers=_auth_headers()` to send the JWT. If you add a new API call and forget this, it will work in dev mode but fail in production. UI/API-client changes for the agents and agent-chat feature (Past Runs list, name display, Chat tab) are covered in the companion doc rather than here.

---

## Database — Supabase tables (new/changed this phase)

### `agent_runs.name` (new column)
```sql
alter table agent_runs add column if not exists name text;
```
Nullable, so existing rows are unaffected. Set only at invoke time via `InvokeAgentRequest.name` — no rename-later UI. UI falls back to the run's `id` when `NULL`.

### `agent_chat_messages` (new table)
```sql
create table if not exists agent_chat_messages (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references agent_runs(id) on delete cascade,
    role text not null check (role in ('user', 'assistant')),
    content text not null,
    user_id uuid not null references auth.users(id),
    created_at timestamptz not null default now()
);
```
Keyed by `run_id` (1:1 with a single agent run) rather than `document_id` like the existing `chats` table. RLS mirrors `agent_runs` — single-user ownership, `select`/`insert` policies only (no update/delete — messages are immutable once written, same convention as `chats`). No org/team scoping yet, consistent with the rest of the agents feature.

---

## Tests — `tests/`

```
tests/
├── fixtures/
│   ├── sample_cv.pdf          # Known CV for testing extraction
│   ├── sample_invoice.pdf     # Known invoice for validation tests
│   ├── sample_scanned.pdf     # Scanned doc for parser routing tests
│   ├── sample_table.csv       # CSV for chart extraction tests
│   └── sample_image.jpg       # Image for vision trigger tests
├── conftest.py                # Shared fixtures (sample_document_id etc.)
├── test_config.py             # Config validation tests
├── test_document_model.py     # Document dataclass tests
├── test_parsers.py            # Parser engine and AutoRouter tests
├── test_routes.py             # API endpoint tests
├── test_classification.py     # Classification accuracy tests
├── test_review.py             # Human review endpoint tests
├── test_auth.py               # Auth and user scoping tests
├── test_feedback_loop.py      # Correction feedback loop tests
├── test_vision.py             # Vision trigger and engine tests
├── test_validation.py         # Validation rules and rulesets
├── test_cache.py              # Cache TTL and hit/miss tests
└── test_queue.py              # Task queue tests
```

Run all tests:
```bash
cd tests
pytest -v
# Should show: 169+ passed
```

Run a specific file:
```bash
pytest test_parsers.py -v
```

---

## `CONTRACTS.md`

**What it is:** The agreed data format specifications between backend and frontend.
**Why it exists:** Two developers built this system in parallel. They needed to agree on exact data shapes before building so one side's output matches the other side's expectations.
**What it contains:**
1. Chunk metadata shape
2. Vision chunk format
3. Evidence format in query sources (including `exact_sentence`)
4. Validation result shape
5. Task queue response format
6. Phase coordination notes (batch extraction decision)

**Key thing to know:** If you change any of these shapes in the backend, update `CONTRACTS.md` and update the frontend to match. Breaking a contract silently is the most common source of bugs in this codebase.

---

## Data flow diagrams

### Upload flow
```
POST /upload (with JWT)
    ↓
auth.get_current_user() → user_id
    ↓
save file to uploads/
    ↓
ingestion.ingest_file(file_path, user_id=uid)
    ↓
AutoRouter.route(file_path) → picks best parser
    ↓
parser.parse() → Document object
    ↓
split Document.pages into chunks
    ↓
for each chunk:
    check cache → embed text → store in Supabase chunks table
    ↓
insert into documents table (with user_id)
    ↓
classify_document() → doc_type, confidence
    ↓
generate_summary() → summary text
    ↓
return {document_id, chunks_stored, parser_used, classification}
```

### Query flow
```
POST /query/stream (with JWT + question + document_id)
    ↓
auth.get_current_user() → user_id
    ↓
classify_question() → "document" or "general"
    ↓
if "general": answer_general() → yield tokens → done
    ↓
expand_query() → rewritten question
    ↓
hybrid_search():
    get_chunks_by_document(document_id)
    embed question
    cosine similarity scores (dense)
    BM25 scores (sparse)
    RRF merge → top 10 candidates
    Cohere rerank → top 5 chunks
    ↓
build prompt with chunks + history + question
    ↓
call_llm(prompt, stream=True) → yield tokens to frontend
    ↓
(separately) get_exact_sentence() per chunk → sources
```

### Extraction flow (flat schema — `fields`)
```
POST /extract (with JWT + document_id + fields)
    ↓
get_classification(document_id) → doc_type
    ↓
get_chunks_by_document(document_id) → context
    ↓
build_correction_examples(doc_type, fields) → few-shot examples
    ↓
EXTRACTION_PROMPT.format(examples, fields, context)
    ↓
call_llm(prompt, json_mode=True) → extracted dict
    ↓
validate_extraction(extracted, fields) → confidence per field
    ↓
ValidationEngine.validate(extracted, doc_type) → business rules
    ↓
trigger_webhooks("extraction.complete", payload)
    ↓
return {extracted, validation, business_validation}
```

### Extraction flow (nested schema — `nested_schema`, or any `/extract/nl` call)
```
POST /extract (nested_schema=...) OR POST /extract/nl (instruction=...)
    ↓
[NL only] generate_schema_spec(instruction) → SchemaSpec (LLM call)
    ↓
validate inline dict / generated dict as SchemaSpec
    ↓
retrieval.extract_dynamic_fields(document_id, spec, ...)
    ↓
get_chunks_by_document(document_id) → context (same sandboxing as flat path)
    ↓
spec_to_field_descriptions(spec) → prompt field list
    ↓
spec_to_model(spec) → recursively-built Pydantic model (cached by spec hash)
    ↓
call_llm(prompt, response_model=DynamicModel, structured_max_retries=2)
    ↓
verify_dynamic_extraction(extracted, spec) → computed durations, backfill, match flags
    ↓
shape response: flat fields → {"value","bbox"}, list/object fields → raw nested JSON
    ↓
validate_extraction() confidence scoring + ValidationEngine.validate(doc_type="dynamic")
    ↓
trigger_webhooks("extraction.complete", payload)
    ↓
return {extracted, validation, business_validation}
```

### Agent invoke flow
```
POST /agents/{agent_name}/invoke (with JWT + task + document_ids + optional name)
    ↓
agent_base.create_and_queue_run(agent_name, task, input_data, user_id, name)
    ↓
db.create_agent_run(...) → agent_runs row, status="pending", name stored (or NULL)
    ↓
return run_id immediately (HTTP response does not block)
    ↓
(background, via core.queue.task_queue.submit)
execute_stages() → runs the agent's multi-stage plan → writes result on completion
    ↓
UI polls GET /agents/runs/{run_id} until status == "completed"
```

### Agent chat flow (new)
```
POST /agents/runs/{run_id}/chat (with JWT + message) — synchronous, no task_queue
    ↓
routers/agents.py → agents.chat.handle_chat_message(run_id, message, user_id)
    ↓
db.get_agent_run(run_id, user_id) → 
    not found / not owned → raise AgentChatError → router returns 400 AGENT_CHAT_UNAVAILABLE
    status != "completed" → raise AgentChatError → router returns 400 AGENT_CHAT_UNAVAILABLE
    ↓
db.get_agent_chat_history(run_id, user_id) → prior turns
    ↓
_format_run_context(run)  → run["result"] only (not state), truncated to MAX_CONTEXT_CHARS
_format_history(history)  → last MAX_HISTORY_TURNS folded into plain "User:/Assistant:" text
    ↓
build single user_prompt: run context + conversation so far + new message
    ↓
call_llm(system=AGENT_CHAT_SYSTEM, user=user_prompt, call_type="agent_chat", session_id=run_id)
    ↓
empty reply → raise AgentChatError (400) — no blank message ever saved
    ↓
db.save_agent_chat_message(run_id, "user", message, user_id)
db.save_agent_chat_message(run_id, "assistant", reply, user_id)
    ↓
return {"role": "assistant", "content": reply}
```

---

## Common tasks

**Add a new extraction template:**
→ Edit `backend/schemas/templates.py` — add to `TEMPLATES` dict and `TEMPLATE_MAP`

**Add a new validation rule:**
→ Add rule class to `backend/validation/rules/` then add to relevant ruleset in `backend/validation/rulesets/`

**Add a new file type:**
→ Create `backend/parsers/yourformat.py` extending `BaseParser`, register in `AutoRouter._register_parsers()`

**Change the LLM model:**
→ Update `LLM_PROVIDER` and `LLM_MODEL` in `.env` — no code changes

**Add a new API endpoint:**
→ Add to the appropriate router in `backend/routers/`, register if new router in `main.py`

**Debug a broken upload:**
→ Check Railway logs → look for structured log lines from `ingestion` and `router` loggers → find the error code → trace through this guide

**Add a new vision prompt:**
→ Edit `VISION_PROMPTS` in `backend/schemas/templates.py`

**Use nested/repeating-object extraction for a new domain:**
→ Nothing to add per-domain — send a `nested_schema` (or an NL `instruction`) describing the fields; `schemas/dynamic.py` builds the model at runtime. Only add code if you need a new schema-agnostic validation rule (`validation/rulesets/generic.py`) or need to widen the verification name-convention tokens in `detect_verifiable_pairs()` (`schemas/validator.py`).

**Add a schema-agnostic validation rule (applies to any dynamic/nested extraction):**
→ Add to `backend/validation/rulesets/generic.py`, matching the real `ValidationResult` fields in `validation/rules/base.py` exactly — a constructor mismatch fails silently.

**Debug an agent chat that isn't answering, or is missing context:**
→ Check whether the run's `status` is actually `"completed"` first (chat is blocked otherwise, by design). If it answers but seems to be missing detail that exists in the run, check whether that detail lives in `result.extra` — `agents/chat.py` only ever sees `result`, never the raw `state`, so anything not surfaced into `result` at the end of the agent's stages won't be visible in chat.

**Add a new agent (beyond `cv_processor`):**
→ Build its multi-stage plan on `agents/base.py`'s stage machine as usual. Chat support requires no per-agent code — `agents/chat.py` and `agents/chat_prompts.py` are agent-agnostic and work off `run["result"]` for any agent once its runs reach `status == "completed"`.

---

## What's not built yet (Phase 3)

- **Full workflow engine** — chainable classify → extract → validate → review → webhook per vertical. Agent runs (invoke/resume/list) and read-only agent chat over a completed run now exist; the broader chainable-workflow vision is still open.
- **Actions/tool-calling in agent chat** — v1 chat is read-only Q&A only; it cannot re-run scoring, re-rank candidates, or take any other action on the underlying run.
- **Document comparison** — diff two versions of a document
- **Vertical wrappers** — CA Helper, CV Screener, Loan Processor (workflow configs on top of this engine)
- **Password reset** — currently manual via Supabase Dashboard
- **Session refresh** — JWT expires after 24h, user must log in again
- **Tighter RLS policies** — currently `allow_all` on most tables, should be `user_id = auth.uid()` after Clerk/Supabase Auth is fully wired (agent_runs and agent_chat_messages already use proper `auth.uid()`-scoped RLS, ahead of the rest of the schema)
- **Schema persistence/versioning** — every `nested_schema` is inline per-request; a `/schemas` CRUD API for saving/reusing user-defined schemas across requests was scoped out of the nested-extraction phase
- **Sub-field correction loop** — `build_correction_examples()` only applies to top-level field names on dynamic schemas; sub-fields inside nested list items aren't individually correctable yet

---

*Last updated: 31 July 2026*
*System status: Phases 0, 1, 2 complete — deployed on Railway + Streamlit Cloud. Nested/dynamic schema extraction implemented and tested end-to-end. Agent runs (invoke/resume/list, named runs) and read-only agent chat over a completed run implemented.*