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
(e.g. CV screening, ITR tax filing help) that call the same underlying
engine APIs to complete a full task, plus chat over a completed agent run
so a user can ask follow-up questions — read-only Q&A for agents with no
`chat_tools` registered, or real tool-calling (recalculation, doc lookup)
for agents that register some.

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
`status == "completed"` (or, for some agents, paused at `needs_input`), a
user can open a chat scoped to that run to ask questions about it, and —
for agents that register tools — trigger safe, read-only actions like a
tax recalculation from inside the chat.

---

## Repository layout

```
doc-intel/
├── backend/                 ← FastAPI server — all business logic
│   └── agents/               ← agent runs (stage machine), per-agent packages, chat
├── frontend/                ← Streamlit UI
├── tests/                   ← pytest test suite
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
**Key thing to know:** If you add a new router, register it here with `app.include_router(...)`. Now registers `routers/schema_templates.py` alongside the existing routers (see below).

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
**Consumed by the ITR agent:** `itr_helper/helpers.py` calls `extract_dynamic_fields()` per uploaded document (Form16, FD/interest certificates, stock statements) using the schemas registered in `nested_schema_templates` (see below) — no changes were made to `retrieval.py` itself for this; it's used exactly as extraction already worked.

---

### `backend/prompts.py`
**What it is:** All LLM prompt templates in one place.
**What it does:** Defines string templates for every type of LLM call — QA, multi-doc QA, general knowledge, classification, query expansion, extraction, NL extraction.
**Why it exists:** Keeping prompts out of code files makes them easy to find, compare, and iterate on without touching business logic.
**Key thing to know:** When an answer quality issue occurs, start here. Prompts use `{variable}` placeholders filled with `.format()`. The `{correction_examples}` block in `EXTRACTION_PROMPT` injects past human corrections as few-shot examples.
- `EXTRACTION_SYSTEM_NESTED` — system prompt used for extraction against nested schemas (via `extract_dynamic_fields()`). Separate from flat `EXTRACTION_SYSTEM` because nested extraction has its own failure modes — merging repeated list items into one, mixing up entities across list items, and collapsing a compound phrase into one field instead of splitting it across siblings (e.g. `"M.Tech, CSE"` going entirely into `course` while `branch` stays null). Includes a rule + before/after example for that last case, written generically rather than tied to any one domain.
- `SCHEMA_GEN_SYSTEM` — system prompt for `schemas.dynamic.generate_schema_spec()`. Instructs the LLM to use `type="list"`/`type="object"` with `properties` for repeating/nested structures, and explicitly **not** to include computed/derived fields (durations, totals, growth rates) — those are always computed in Python after extraction, never by the model.

**Note:** agent-specific and agent-agnostic prompts do **not** live here — see `backend/agents/chat_prompts.py` (chat, agent-agnostic), `backend/agents/cv_processor/prompts.py` (CV agent-specific), and the ITR agent's tax-summary prompt inline in `agents/itr_helper/agent.py`. Prompts only live in this file when they belong to the core document-QA/extraction engine, not to a specific agent.

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
- `create_agent_run(agent_name, task, input_data=None, user_id="anonymous", name=None)` — inserts a new `agent_runs` row with `status="pending"`. Takes an optional `name` (display name set at invoke time, purely cosmetic).
- `get_agent_run(run_id, user_id)` — selects `*`, so it already includes `name` and needed no change when the column was added.
- `list_agent_runs(user_id="anonymous", agent_name=None, status=None, limit=50)` — selects an **explicit column list** (`id, name, agent_name, task, status, current_stage, created_at, completed_at`) rather than `*`, so `name` had to be added to that list explicitly for the Past Runs list and Chat tab dropdown to display it.

**Agent chat — functions:**
- `save_agent_chat_message(run_id, role, content, user_id)` — inserts one row into `agent_chat_messages`.
- `get_agent_chat_history(run_id, user_id)` — selects `role, content, created_at` for a run, ordered oldest-first, scoped by both `run_id` and `user_id`.
- Both are a direct mirror of the existing `chats` table pattern (`save_message` / `get_chat_history`), keyed by `run_id` instead of `document_id` — agent chat is scoped to a single run, not shared across a document's whole chat history.

**Agent run flags — functions (new, CV Processor phase):**
- `insert_agent_flag(run_id, flag_type, detail, user_id)` — inserts one row into `agent_run_flags`. `flag_type` is a short code (e.g. `"duplicate_candidate"`, `"ranking_inconsistency"`, `"low_evidence_confidence"`); `detail` is a free-form JSONB payload specific to that flag type.
- `get_flags_for_run(run_id, user_id)` — all flags for a run, ordered by `created_at`. Read by the review stage (to decide whether to re-run ranking) and surfaced to the UI as guardrail warnings.
- Why a separate table instead of stuffing flags into `agent_runs.result`: flags are diagnostic/audit data about *how* a run behaved (bias-audit hits, contradiction detections, ranking swaps), not part of the run's actual output — keeping them separate means `result`'s shape stays exactly the fixed output contract every chat/consumer already relies on (see `agents/chat.py`'s `result`-not-`state` design), and flags can be queried/reported on across runs independently of any one run's result.

**Schema templates — functions (new, Phase B):**
- `create_schema_template(name, spec, scope, owner_id, org_id=None)` — inserts into `nested_schema_templates`. `scope` is one of `personal` / `team` / `org` / `global` (see table definition below).
- `list_schema_templates(user_id, org_id=None)` — returns every template the caller can see: their own `personal` ones, `team`/`org` ones for orgs they belong to, plus all `global` ones. Built as a single query with an `or_()` filter rather than several round trips.
- `get_schema_template(template_id, user_id)` — single template, enforcing the same visibility rule as `list_schema_templates`.
- `delete_schema_template(template_id, user_id)` — owner-only; `global`-scope templates cannot be deleted through the API (seed data only), enforced in `routers/schema_templates.py` rather than here.
- Why this exists: nested extraction schemas were previously always inline-per-request (`nested_schema` in the extract call, or NL-generated each time) — the ITR agent's three schemas (Form16, FD/interest, stocks) are the first case where the same schema needs to be reused across many runs and many users, which is what pushed schema persistence from "not built yet" (see Phase 0-2 guide) into being built now.

**Platform admin — functions (new, Phase B):**
- `is_platform_admin(user_id) -> bool` — checks the `platform_admins` table for a row matching `user_id`. Used by `core/auth.py` (see below), not called directly from routers.

**Crash recovery — function (new):**
- `get_running_agent_runs() -> list[dict]` — a cross-user query (`eq("status", "running")`, no `user_id` filter). **The only helper in `db.py` that isn't scoped to one user** — recovering abandoned runs is inherently an ops-level action across everyone, not something a single user's session can be scoped to. Returns `id`, `agent_name`, `name`, `user_id`, `current_stage`, `created_at`, `started_at` per row; `user_id` is included specifically so each run can be resumed as its actual owner (`resume_stages()` / `get_agent_run()` remain user-scoped internally — this function is the one deliberate exception, not a pattern to copy elsewhere). Used by the new `POST /agents/admin/restart-stuck-runs` endpoint (see `routers/agents.py` below).

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
**Key thing to know:** Called from extraction endpoints in `routers/extraction.py` after every extract call.

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
**Nested/dynamic extraction:** `ExtractRequest` takes an optional inline `nested_schema: dict` alongside the existing flat `fields: dict` — not a separate endpoint, so one route handles both shapes and both return the same response contract (`extracted` / `validation` / `business_validation` / `extraction_id`). Named `nested_schema` rather than `schema` to avoid colliding with Pydantic v1's reserved `BaseModel.schema()`. A validator requires exactly one of `fields` / `nested_schema`. `extract()` branches on which is present: `nested_schema` → validated as a `SchemaSpec` and passed to `retrieval.extract_dynamic_fields()`; otherwise → the existing flat `extract_fields()` path. `NLExtractRequest.preview_only` now returns a full `SchemaSpec` (via `generate_schema_spec`) instead of the old flat `SchemaResult`, so a preview matches what the real call will produce, nested fields included. `GET /schemas/example` returns a worked nested-schema example for API consumers building a schema by hand. This router is unchanged by Phase A/B and the CV Processor work — schema **persistence** (saving/reusing a `nested_schema` across requests) is handled by the new `routers/schema_templates.py` below, not here; this router still only takes schemas inline.

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

### `backend/routers/schema_templates.py` (new, Phase B)
**Endpoints:** `POST /schema-templates`, `GET /schema-templates`, `GET /schema-templates/{id}`, `DELETE /schema-templates/{id}`
**What it does:** CRUD for persisted, reusable nested extraction schemas, at four visibility scopes — `personal` (only the creator), `team`, `org`, and `global` (seeded, read-only via the API). Backed by `db.py`'s schema-template functions and the `nested_schema_templates` table.
**Why it exists:** Pulled forward from "not built yet" specifically to support the ITR agent, whose three schemas (Form16, FD/interest, stocks) are shared across every user and every run rather than being defined inline per request.
**Key thing to know:**
- `POST /schema-templates` validates the inline `spec` as a `SchemaSpec` (same Pydantic model as the extraction router's `nested_schema`), so a template is guaranteed to be usable wherever a `SchemaSpec` is expected.
- Creating a `team`/`org`-scoped template requires the caller to belong to that team/org; creating a `global` one requires `is_platform_admin` (see `core/auth.py` below) — enforced at the router/app level.
- `DELETE` is owner-only for `personal`/`team`/`org` scopes; `global` templates can't be deleted through this API at all (seed data only), even by a platform admin — intentional, to keep the ITR schemas from being accidentally removed by any single admin action.
- **Known gap (documented, not yet fixed):** the table's own `INSERT` RLS policy still checks an `org_admin`-style condition left over from an earlier draft, while the actual app-level gate for `global` inserts checks `platform_admins`. The two don't fully agree yet — flagged in "What's not built yet" below rather than silently left undocumented.

---

### `backend/routers/agents.py`
**Endpoints:** `POST /agents/{agent_name}/invoke`, `POST /agents/runs/{run_id}/resume`, `GET /agents/runs` (list, via `list_agent_runs`), `GET /agents/runs/{run_id}`, `POST /agents/runs/{run_id}/chat`, `GET /agents/runs/{run_id}/chat`, `POST /agents/admin/restart-stuck-runs`.
**What it does:** Invokes and resumes multi-stage agent runs, lists/reads past runs, and handles chat scoped to a single run (read-only Q&A or tool-calling, depending on the agent).
**Why it exists:** Agent orchestration is a distinct concern from document upload/query/extraction — it composes those engine APIs into multi-stage tasks rather than exposing a single call.
**Key thing to know:**
- All request models use Pydantic v2's `@field_validator` (migrated together from the deprecated `@validator` for consistency).
- `InvokeAgentRequest` has an optional `name: str | None` — lets the caller label a run at invoke time (e.g. "Backend Engineer — July batch") instead of only ever seeing a UUID. `invoke_agent()` passes `req.name` straight through to `agent_base.create_and_queue_run(...)`.
- **`ResumeAgentRequest` (Phase A, reshaped)** now accepts `answers` **or** `new_input` — enforced mutually exclusive via a `model_validator`, since they mean two different things: `answers` resumes a `needs_input`-paused run at its `current_stage`; `new_input` feeds new data into an already-`completed` run (e.g. the ITR agent recalculating after another document is uploaded) and has the entry stage decided dynamically. `resume_run()` checks the run's live `status` against which field was supplied *before* calling `agents.base.accept_resume_input()`, returning one of three specific 400 codes on mismatch: `AGENT_RESUME_ANSWERS_REQUIRED`, `AGENT_RESUME_NEW_INPUT_REQUIRED`, `AGENT_RUN_NOT_RESUMABLE`.
- **`POST /runs/{run_id}/chat`** is deliberately **synchronous** — not routed through `task_queue.submit()` like invoke/resume, since one chat turn is at most one tool-calling loop, not multi-stage background work. Looks up the run's `agent_def` from `agents/registry.py` via `agent_name`, resolves `chat_tools` (calling the `chat_tools_factory(run_id)` if the agent registers one — see registry.py below — or falling back to an empty list), and passes them to `agents.chat.handle_chat_message()`. Returns `400` with code `AGENT_CHAT_UNAVAILABLE` if the run doesn't exist, isn't owned by the caller, or isn't in a chat-eligible status.
- **`GET /runs/{run_id}/chat`** returns `{"messages": [...]}`, oldest first, via `get_agent_chat_history`. Returns `404` (not the chat-specific 400) if the run itself doesn't exist/isn't owned.
- **`POST /agents/admin/restart-stuck-runs`** (new) — crash recovery for runs abandoned mid-execution (see the dedicated `agents/base.py` note below for why this is needed at all). Gated on `user.is_platform_admin` — the same gate used for global schema-template creation — returning `403 RESTART_STUCK_RUNS_FORBIDDEN` otherwise. Calls `db.get_running_agent_runs()`, and for each row looks up `get_agent_def(run["agent_name"])`: if that agent type is no longer registered, the run is skipped and reported rather than failing the whole batch; otherwise it submits `agent_base.resume_stages(run_id, agent_def["stages"], user_id=run["user_id"])` to `core.queue.task_queue` — the exact same function the normal `/resume` endpoint already calls, no special-cased restart logic. Returns `{"found": int, "restarted": [run_id, ...], "skipped": [{"run_id", "reason"}, ...]}`. **Manual, admin-triggered only — no automatic heartbeat/staleness sweep** — an operator confirms the old process is actually dead before calling this, rather than the system guessing off a timeout. Works identically for every registered agent (`cv_processor`, `itr_helper`, future ones) since it reuses `resume_stages()` unchanged.

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
No `print()` statements anywhere in the codebase. Always use `get_logger()`. `backend/agents/chat.py` uses `get_logger("agents.chat")`; the CV processor's bias-audit and flag-writing paths log through the same pattern (`get_logger("agents.cv_processor")`).

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

Agent chat/resume errors do **not** use this taxonomy — `agents/chat.py` raises `AgentChatError`, `agents/base.py` raises `AgentRunError`, mapped by the router to a small set of purpose-specific codes (`AGENT_CHAT_UNAVAILABLE`, `AGENT_RESUME_ANSWERS_REQUIRED`, `AGENT_RESUME_NEW_INPUT_REQUIRED`, `AGENT_RUN_NOT_RESUMABLE`) rather than a `PARSE_0xx`-style prefix, since these are agent-orchestration conditions, not document-engine failures.

---

### `backend/core/document.py`
**What it is:** The Document model — the central data contract.
**What it does:** Defines 8 dataclasses that represent a parsed document in a structured way: `Entity`, `TableCell`, `Table`, `LayoutElement`, `ImageElement`, `DocumentPage`, `Classification`, `Document`.
**Why it exists:** Before this, every parser returned a different format — raw text, list of dicts, etc. Now every parser returns a `Document` and everything downstream works the same way regardless of file type.
**Key thing to know:** This is the most important file in `core/`. The `Document` object is the contract between parsers and the rest of the system. Do not change the field names without updating every parser and consumer. When you add a new parser, it must return a `Document`.

---

### `backend/core/auth.py`
**What it is:** JWT verification middleware.
**What it does:** Verifies incoming JWTs using `supabase.auth.get_user(token)` — lets Supabase handle its own token verification regardless of algorithm (ES256 or HS256). Also handles API key auth. Returns a `UserContext` with `user_id` and `email`.
**Why it exists:** Centralizes auth so every router can use `Depends(get_current_user_context)` without duplicating verification logic.
**Key thing to know:** Dev mode — if `SUPABASE_JWT_SECRET` is not set, returns an anonymous dev `UserContext` for every request. This allows local development without auth setup. The `get_user_id(user)` helper safely extracts `user_id` from any user context.
**Platform admin (new, Phase B):** `UserContext` gained an `is_platform_admin: bool` field, populated from `db.is_platform_admin(user_id)` on every request. Two things worth knowing:
- It **fails safe** — if the `is_platform_admin` lookup itself errors (DB hiccup, etc.), the field defaults to `False` rather than `True`, so a transient failure can never accidentally grant admin.
- It is **not free in dev mode** — unlike the rest of dev-mode auth (which skips real checks entirely), the platform-admin lookup still hits the DB even for the anonymous dev user, so admin-gated behavior can be tested locally without a real Supabase JWT.

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
**Key thing to know:** Uses Python's `threading` and `queue.Queue` — no external dependencies. 3 worker threads by default. For multi-server production, replace internals with Celery + Redis while keeping the same `task_queue.submit()` interface. Agent invoke and **both** resume paths (`answers` and `new_input`) go through this queue; agent chat deliberately does not — a chat turn (plain or tool-calling) is bounded work, not a multi-stage run.

---

## Backend — agents/

The `agents/` directory holds the multi-stage agent run machinery, the
per-agent packages (`cv_processor/`, `itr_helper/`), shared agent tooling,
and chat (read-only or tool-calling) over a completed/eligible run. It
sits above the document engine (`ingestion.py` / `retrieval.py` / routers)
and composes those APIs rather than duplicating them.

---

### `backend/agents/base.py`
**What it is:** The stage-machine layer — creates, queues, executes, and resumes agent runs.
**What it does:** `create_and_queue_run(agent_name, task, input_data, user_id="anonymous", name=None)` is the fast, synchronous entry point — it creates the `agent_runs` row via `db.create_agent_run()` and returns `run_id` immediately, **without** executing any stages; callers hand `execute_stages()` to `core.queue.task_queue.submit()` right after, so the HTTP response doesn't block on the run.
**Why it exists:** Separates "register that a run exists" (fast, must return quickly) from "actually run the stages" (slow, backgrounded) from "resume a paused or completed run" (two distinct triggers, see below).
**Key thing to know:**
- `name` is optional and purely cosmetic — threaded through to `create_agent_run()`, never read by the stage machine itself.
- **`accept_resume_input()`** (Phase A — renamed/reshaped from the earlier `accept_resume_answers()`) branches on the run's current `status`:
  - `needs_input` + `answers` supplied → unchanged original behavior, resumes execution at `current_stage`.
  - `completed` + `new_input` supplied → merges `new_input` into `state["input_data"]`, calls `plan_resume_stage()` to pick the re-entry stage, sets `current_stage` to that, and re-queues execution.
  - Any other status, or a payload that doesn't match the run's actual status → raises `AgentRunError`.
- **`plan_resume_stage()`** (Phase A, new) — a single LLM call (Instructor `response_model=_StageSelection`), grounded in the agent's `stage_descriptions` (passed in by the caller from `agents/registry.py`, to avoid a circular import between `base.py` and `registry.py`). Picks the *earliest* stage whose output could change given the new input. Raises `AgentRunError` — deliberately **no** self-correcting retry — if `stage_descriptions` is missing for that agent, or if the model names a stage outside the known set. On financial data (the ITR use case), silently re-running the wrong stage is judged worse than a clear, loud failure.
- `resume_agent_run()` (sync test wrapper) updated to match `accept_resume_input()`'s new signature. `resume_stages()` (the actual re-execution loop) itself is unchanged by Phase A.
- **Output-contract validation (new, CV Processor phase):** `execute_stages()` now validates the final `result` object against a small `REQUIRED_RESULT_KEYS` set (`summary`, `findings`, `data`, `extra`) before marking the run `completed` — if a stage's final-assembly step produces a `result` missing one of these keys, the run is marked `failed` with a descriptive error instead of silently completing with a malformed result. Added because `agents/chat.py`'s entire context model assumes `result` always has this fixed shape (see chat.py below) — a missing key there previously surfaced as a confusing downstream chat failure rather than an upfront one at the source.
- **Crash recovery (new):** the stage loop already persisted progress *between* stages (`current_stage` + `state` written to `agent_runs` after each stage completes) — but nothing detected a process/pod dying mid-run, which left the row stuck at `status="running"` forever with no automatic resume. No changes were made to `base.py` itself to fix this — the fix reuses the existing `resume_stages()` primitive as-is, driven by a new admin endpoint (`POST /agents/admin/restart-stuck-runs`, see `routers/agents.py`) and a new cross-user query (`db.get_running_agent_runs()`). This works identically for every agent because every stage is written to be idempotent given the same input state — re-entering at a stale `current_stage` after a crash just re-runs that one stage from scratch, which is safe. **Accepted caveat:** deployment is currently single-instance, so any `status="running"` run at restart time is guaranteed abandoned; if this becomes multi-instance, triggering a restart while a different, genuinely-still-alive instance is processing that same run would double-execute it — that's left to operator judgment (confirm the old instance is actually dead first), not solved in code.

---

### `backend/agents/registry.py`
**What it is:** The agent registry — maps an agent name to its definition.
**What it does:** A dict-of-dicts (or equivalent) keyed by `agent_name`, each entry describing that agent's stage plan and optional extension points.
**Why it exists:** `agents/base.py`'s stage machine, and `routers/agents.py`, both need to look up "what does this agent's plan look like" and "what optional capabilities does it have" without hardcoding per-agent branches.
**Key thing to know:** Two optional keys were added across Phase A and Phase B, both read defensively (`agent_def.get(key, default)`) so agents that don't set them are completely unaffected:
- **`stage_descriptions: {stage_name: "what this stage does"}`** (Phase A) — used only by `plan_resume_stage()` in `base.py`. `cv_processor`'s entry doesn't set this — it never takes `new_input` on a completed run.
- **`chat_tools` → evolved to `chat_tools_factory`** (Phase B) — originally a static `list[ToolSpec]` (Phase A); the ITR agent's tools need to be scoped to a specific `run_id` (e.g. "recalculate *this* run's tax," not tax in general), which a static list can't express. Changed to a **factory**: `chat_tools_factory: Callable[[str], list[ToolSpec]]` taking `run_id` and returning that run's tools. `routers/agents.py`'s chat route calls `chat_tools_factory(run_id)` if present, or falls back to `[]`. `cv_processor` registers neither key, so it still gets plain read-only chat via `AGENT_CHAT_SYSTEM`.

---

### `backend/agents/chat.py`
**What it is:** The chat handler for an agent run — read-only Q&A, or tool-calling if the agent registers tools.
**What it does:** `handle_chat_message(run_id, message, user_id, chat_tools=None)` answers one chat turn grounded in the run's `result`, persists the user message and the assistant's final reply on success, and returns `{"role": "assistant", "content": reply}`.
**Why it exists:** Deliberately **not** part of `base.py`'s stage machine — chat (in either mode) doesn't mutate the run's own state/plan the way a stage does; even the ITR agent's "recalculate" tool goes through its own dedicated resume path (`new_input`), not through chat mutating the run directly.
**Key thing to know:**
- **Availability check lives here, not in the router** — `AgentChatError` is raised if the run doesn't exist / isn't owned by `user_id`, or isn't in a chat-eligible status. Enforcing this in the handler means any future caller of `handle_chat_message()` gets the same guarantee, not just the HTTP route.
- **Context source is `result`, not `state`** — unchanged from the original design. `state` carries every sub-agent's raw intermediate output and was large enough on its own to blow a fallback model's TPM limit in testing (a single turn once requested ~8100 tokens against a 6000–8000 TPM cap and was rejected with a 413 by every provider in the fallback chain). `result` is the fixed, already-compact output contract (`summary` / `findings` / `data` / `extra`), and — as of the CV Processor phase — `base.py` now actively enforces that every completed run's `result` has this shape (see `REQUIRED_RESULT_KEYS` above), so chat can rely on it unconditionally.
- **`MAX_CONTEXT_CHARS = 6000`** — hard character cap (`_truncate()`) on the formatted run-context block, independent safety net against `result` growing large.
- **`MAX_HISTORY_TURNS = 20`** — older turns dropped from the *prompt* only (not storage), via `_format_history()`.
- **No multi-turn message array** — `llm/engine.call_llm()` only accepts a single `system` + `user` string pair, so prior turns are folded into the `user` string as plain `"User: ...\nAssistant: ..."` text.
- **`chat_tools` parameter (Phase A, new):**
  - Empty/`None` (the default, and every case before Phase A) → **identical to the original behavior** — a single `call_llm()` call with `AGENT_CHAT_SYSTEM`.
  - Non-empty (`cv_processor` never passes any; `itr_helper` does) → routes through `llm/tool_orchestrator.run_tool_loop()` instead, using `AGENT_CHAT_SYSTEM_WITH_TOOLS`. `call_type` is passed as `f"agent_chat_tools:{agent_name}"` for per-agent usage tracing. Only the loop's `final_answer` is persisted to chat history — intermediate tool calls are visible in LLM-call tracing (via `call_type`) but not stored as separate chat messages in v1.
- **Empty-reply handling** — an empty/whitespace final reply (from either the plain or tool-calling path) raises `AgentChatError` instead of silently saving a blank assistant message.

---

### `backend/agents/chat_prompts.py`
**What it is:** The system prompt(s) for agent chat.
**What it does:** Defines two prompts:
- **`AGENT_CHAT_SYSTEM`** (original, unchanged) — used as-is for any agent that registers no `chat_tools_factory` (currently `cv_processor`). Frames chat as strictly read-only and tells the model to decline and point to starting a new run if asked to take an action.
- **`AGENT_CHAT_SYSTEM_WITH_TOOLS`** (Phase A, new) — used when `chat_tools` is non-empty. Drops the "read-only, can't take action" framing from the original prompt, and adds explicit guidance to prefer calling a tool over restating a possibly-stale number pulled from the run's stored `summary` (important for the ITR agent, where a `recalculate_tax` tool exists specifically because the stored summary can go stale the moment new input arrives).
**Why it exists:** Kept separate from any one agent's own prompts file because both prompts are **agent-agnostic** — reused across every agent's runs according to whether that agent registered tools, not written for one agent's task.

---

### `backend/agents/tools/` (new, CV Processor phase)
Shared tool/helper code used by the CV processor's stages — kept in a
`tools/` subpackage rather than inline in `cv_processor/agent.py` since some
of it (candidate-data handling patterns) is expected to be reusable by
future screening-style agents, not just this one.

- **`candidate_data.py`** — sanitization helpers run on every candidate's
  raw extracted data before it's used in any prompt: strips/redacts
  personally-identifying signal that shouldn't influence scoring (the
  specifics feed into the fairness guardrails described in
  `cv_processor/prompts.py` below).
- **`scoring.py`** — the actual per-criterion scoring call site.
  - **Evidence snippets** — each score is now paired with a short quoted
    span from the source document (`evidence_snippet`) plus a **numbered
    citation** back to where in the document it came from, rather than a
    bare number. This is what powers the "why did this candidate score
    this way" explainability requirement.
  - **`run_id` parameter** — scoring calls are now tagged with the owning
    `run_id` (for tracing/debugging a specific run's scoring decisions,
    and so flags written via `db.insert_agent_flag()` can be tied back to
    the run that produced them).
  - **`score_skill_groups()`** (new, performance-optimization pass) —
    replaces the old "one call per raw skill, each call holding every
    candidate" approach for skills specifically (education/experience
    still use the original, unchanged `score_candidates()`). Chunks
    **candidates** into groups sized so each call covers roughly
    `SCORE_SKILLS_PAIRS_PER_CALL` (20, see `optimization_config.py` below)
    candidate × skill-group pairs, and scores a chunk of candidates
    against **every** skill group in one call (a grid) rather than one
    call per skill. `chunk_size = max(1, SCORE_SKILLS_PAIRS_PER_CALL //
    n_skill_groups)`; total calls ≈ `ceil(n_skill_groups × n_candidates /
    SCORE_SKILLS_PAIRS_PER_CALL)`. New structured-output models:
    `SkillGroupCandidateScore` (adds `group_name` to the usual
    score/reasoning/evidence_snippet shape) and `SkillGroupScoreBatch`
    (`scores: list[SkillGroupCandidateScore]`). New prompt builder
    `_skill_groups_system_prompt(skill_groups)` describes every skill
    group once per call and applies the same `_EXCLUSION_CLAUSE`
    independence instruction used elsewhere in this pass (see
    `cv_processor/` below). **Output shape preserved:** still returns
    `{"skill:<group_name>": {document_id: {score, reasoning,
    evidence_snippet}}}`, the same key format the rest of the pipeline
    already expected, so `merge_scores`/`finalize_response`/
    `compute_adjusted_scores` needed no structural changes.
  - **Important tradeoff:** grouping skills does **not always reduce**
    total call count — it trades "few large, risky-size calls" for "more
    small, safe-size calls," and which way it goes depends on how well a
    JD's skills semantically merge (10 skills → 3 groups: `ceil(3×100/20)
    = 15` calls, fewer than the old 10-per-skill approach's risk profile
    but more calls; 10 skills that don't merge at all → `ceil(10×100/20)
    = 50` calls, 5× more than before). The `/20` cap deliberately
    prioritizes bounded, safe prompt size over minimizing raw call count
    — an explicit, discussed choice, not an oversight.

---

### `backend/agents/cv_processor/` (existing agent — significantly expanded, then optimized for scale)
**What it is:** The CV-screening agent's stage plan and prompts.

**What it does — guardrails/explainability pass (adds stages, 6 → 10):**
- **Guardrail stages** — fairness/scope/depth clauses injected into the
  scoring prompts (see `prompts.py` below) plus a **bias-audit log**: every
  scoring decision that trips a fairness heuristic is logged (not
  auto-corrected) via `db.insert_agent_flag()` for later human review.
- **Claim verification** — extracted candidate claims (e.g. "5 years at
  Company X") are checked against source-document evidence before being
  used in scoring; unverifiable claims lead to an **adjusted (lowered)
  confidence score** for that claim rather than being silently trusted or
  silently dropped.
- **Duplicate/contradiction detection** — flags near-duplicate candidate
  entries and internally contradictory claims within a single candidate's
  data (e.g. conflicting employment dates across two source documents),
  written as `agent_run_flags` rows via `db.insert_agent_flag()`. Runs on
  **100% of candidates** and is untouched by the optimization pass below.
- **Ranking-consistency review** — a review pass over the final ranking
  that checks for ordering inconsistencies (e.g. a lower-scored candidate
  ranked above a higher-scored one due to a tie-break quirk). Includes a
  **hard adjacency check in code** (not left to the LLM) — the review
  stage's LLM call can *propose* a re-rank, but the actual swap is only
  applied if a deterministic Python check confirms the proposed swap is
  between adjacent-in-score candidates; a proposed swap that isn't
  adjacent is flagged (`ranking_inconsistency`) rather than silently
  applied, since a large, LLM-proposed jump in ranking is treated as a
  sign the review itself may be wrong, not as ground truth.
- `finalize_response()` assembles the `result` object (summary/findings/
  data/extra, matching `REQUIRED_RESULT_KEYS` in `base.py`) — `extra`
  carries evidence citations, any flags raised during the run, the
  evaluation plan, and full per-criterion scores.

**What it does — performance-optimization pass (same behavior, fewer/safer LLM calls):**
Scope: this pass is **entirely about call volume and prompt size at scale**
(a single run for 5 candidates was taking 10+ minutes from sequential LLM
calls; at 100 candidates several stages were dumping every candidate into
one giant prompt, risking token-limit failures). **No guardrail,
explainability, or analysis-quality behavior was added or removed** — only
*how* and *when* the LLM is called changed. Three stages were redesigned:
`verify_claims`, `score_skills`, `compute_adjusted_scores`.

- **New file — `optimization_config.py`:** every tunable introduced by
  this pass lives here, not hardcoded in pipeline logic — batch sizes,
  multipliers, and hard call-count caps for the three redesigned stages
  (e.g. `VERIFY_CLAIMS_BATCH_SIZE=5`, `VERIFY_CLAIMS_TOP_PERCENT=0.35`,
  `SCORE_SKILLS_PAIRS_PER_CALL=20`, `ADJUSTED_SCORES_BATCH_SIZE=5`,
  `ADJUSTED_SCORES_MAX_CALLS=10`).
- **Pipeline reordered** — `verify_claims` moved from immediately after
  `detect_duplicates_and_issues` (pre-scoring) to immediately after
  `merge_scores` (post-initial-ranking):
  ```
  plan_evaluation → detect_duplicates_and_issues → score_education → score_skills
  → score_experience → merge_scores → verify_claims → compute_adjusted_scores
  → review_ranking → finalize_response
  ```
  **Why:** the redesigned `verify_claims` only checks a *subset* of
  candidates chosen from the ranking, so a ranking has to exist first.
  `merge_scores` now produces its initial ranking from **unverified**
  data — an accepted tradeoff, since verification was never meant to
  change raw per-criterion scores anyway (it only ever fed a separate
  *adjusted comparison score*), so ranking on unverified data doesn't
  change what the ranking itself represents. Every stage's `next_stage`
  value in `agent.py` was updated to match; the transition list was
  manually traced end-to-end after implementation to confirm no stage
  points to a stale name.
- **`verify_claims` — batched and subset-limited:** was 1 LLM call per
  candidate (parallelized, `ThreadPoolExecutor(max_workers=8)`) verifying
  *every* candidate; now `_verify_claims_batch()` sends
  `VERIFY_CLAIMS_BATCH_SIZE` (5) candidates per call (independence
  enforced via a new `_EXCLUSION_CLAUSE` appended to the prompt — prompt-
  only, not mechanically verifiable, an accepted tradeoff), batches (not
  individual candidates) go to the same thread pool, and a new
  `_select_verification_subset()` picks who gets checked at all: top
  `N × 1.5` if a specific top-N was requested, else the top 35% of all
  candidates. **No cascading re-check** — this selection happens once; if
  verification later demotes a borderline verified candidate, whoever
  would replace them in the shortlist is never itself verified. A known,
  accepted, documented limitation, chosen specifically to avoid unbounded
  re-verification loops. Response model changed from one `ClaimVerification`
  per call to `ClaimVerificationBatch` (`verifications: list[ClaimVerification]`).
  Net effect at 100 candidates: ~7 calls (top 35% in batches of 5) versus
  the old design's 100.
- **`score_skills` — skill grouping + grid-style batched scoring:**
  `plan_evaluation`'s structured output gained `skill_group_hints:
  list[SkillGroupHint]` (`group_name` + `member_skills`); its prompt now
  instructs the model to semantically group skills describing the same
  capability at different specificity (e.g. "AI"/"AI agents"/"multi-agent
  systems" → "Agentic AI") without merging genuinely unrelated skills, and
  every raw skill must land in exactly one group. This grouping happens
  **once, at planning time** — not re-inferred per scoring call — so it's
  consistent and auditable (stored in `plan["skill_groups"]`, visible in
  `result.extra.plan`). New `_build_skill_groups(skills, hints)` in
  `agent.py` cross-references the hints against the real skills list and
  **computes each group's `requirement_level` in code**, not trusted from
  the LLM (`"must"` if any member skill was tagged `"must"`), keeping
  must/nice weighting deterministic; any skill the LLM missed becomes its
  own singleton group, guaranteeing full coverage. Actual scoring moved
  from `score_candidates()` (still used for education/experience,
  unchanged) to the new `score_skill_groups()` in `tools/scoring.py` (see
  above) — the `score_skills` stage function itself simplified from a
  per-skill loop to one call into `score_skill_groups()`, which handles
  all chunking internally.
- **`compute_adjusted_scores` — batched, capped:** was one
  `score_candidates()` call per **individual flagged claim**, fully
  sequential. Now each flagged candidate's flagged claims (which may span
  several criteria) are bundled into one request entry, batched
  `ADJUSTED_SCORES_BATCH_SIZE` (5) candidates per call, with a **hard cap**
  of `ADJUSTED_SCORES_MAX_CALLS` (10) total calls — if more than 50
  flagged candidates would need more than 10 calls at batch size 5, the
  batch size grows instead (`batch_size = ceil(n_flagged /
  ADJUSTED_SCORES_MAX_CALLS)`), guaranteeing this stage never exceeds 10
  LLM calls regardless of how many candidates got flagged. New prompt
  `ADJUSTED_SCORES_SYSTEM` (uses both `_EXCLUSION_CLAUSE` and the existing
  fairness clause) and models `AdjustedCriterionScore` /
  `AdjustedScoreBatch`; a new helper `_adjust_batch()` skips and logs a
  failed batch rather than crashing the whole stage. The score-delta math
  itself (summing deltas across flagged claims, proportional shift to
  `overall_score`) is **unchanged** — only how many calls it takes to
  gather the inputs changed. `_resolve_criterion_key()` (maps a flagged
  claim's free-text field to a `criterion_scores` key) now also checks
  skill-group `member_skills`, not just `group_name`, since a flagged
  claim is likely to reference the original raw skill (e.g.
  "Kubernetes") rather than the merged group label (e.g. "Cloud
  Infrastructure"). Net effect at 20 flagged candidates: ~4 calls versus
  the old ~20+.
- **`cv_processor/prompts.py`** — fairness / scope / depth clauses on the
  scoring prompt (guardrails pass) plus, from the optimization pass: the
  new `_EXCLUSION_CLAUSE` (independence instruction, reused across
  `verify_claims`, `score_skill_groups`, and `compute_adjusted_scores`),
  new `ADJUSTED_SCORES_SYSTEM`, `VERIFY_CLAIMS_SYSTEM` rewritten from
  single-candidate to batch framing, `PLAN_EVALUATION_SYSTEM` extended
  with skill-grouping instructions, and `JUDGE_MERGE_SYSTEM` wording
  updated to reference skill groups instead of raw skills.
- **Not touched by the optimization pass:** `tools/candidate_data.py`,
  `agents/base.py`, `db.py`, `agents/registry.py`, `routers/agents.py` —
  none of the guardrail/explainability/duplicate-detection logic needed
  any change; this pass changed calling patterns only.

**Why it exists:** the guardrails/explainability additions are fairness and
correctness guardrails on top of the original scoring pipeline — they don't
change what the agent is fundamentally doing (score and rank candidates
against a task), only how defensible, auditable, and internally consistent
that scoring is. The optimization pass exists purely to make that same
pipeline usable at real candidate-batch scale (tested conceptually against
a 100-candidate scenario) without excessive latency or token-limit risk.

**Known limitations from the optimization pass (documented, not bugs):**
- No cascading re-verification (see `verify_claims` above).
- `score_skills` call count can go **up**, not down, in the worst case
  (skills that don't semantically merge) — an accepted tradeoff of
  prioritizing bounded prompt size over minimizing call count.
- `_resolve_criterion_key()` is still best-effort substring matching (now
  over group names *and* member skills) — same class of limitation as
  other field-name fuzzy matching elsewhere in the codebase (see
  `schemas/validator.py`'s token-based matching note, which this does
  *not* yet follow — worth revisiting together).

**Key thing to know:** `cv_processor` registers **no** `stage_descriptions`
and **no** `chat_tools_factory` in `agents/registry.py` — its chat stays
plain read-only Q&A, unaffected by Phase A's tool-calling chat or Phase B's
ITR-specific tools. None of that changed with the optimization pass either.

---

### `backend/agents/itr_helper/` (new agent, Phase B)
**What it is:** The ITR (Income Tax Return) filing-help agent — extracts
tax-relevant documents, stitches them into a single taxpayer profile,
computes tax liability under both regimes, and answers follow-up questions
with the ability to recalculate on new data.
**Files:**
- **`helpers.py`** — per-document extraction (via `retrieval.extract_dynamic_fields()`
  against the ITR schemas registered in `nested_schema_templates`, see
  above) and stitching into one taxpayer profile. Each document's
  extraction is isolated — one failing document doesn't abort the whole
  run, it's recorded and excluded, mirroring the `extract_batch` tool's
  existing partial-success pattern. Multiple Form16 entries (multiple
  employers in one financial year) are kept as **separate per-employer
  entries**, deliberately **not pre-summed** at extraction/stitch time —
  summing happens only in `calculator.py`, in one place, so there's a
  single source of truth for how totals are derived rather than two
  (extraction-time sum vs. calculation-time sum) that could disagree.
- **`tax_rules.py`** — versioned tax rules for both the old and new
  regimes, scoped to Assessment Year 2026-27. Kept as data (rates, slabs,
  deduction limits) separate from the calculation logic in `calculator.py`,
  the same data-not-code philosophy as `schemas/templates.py` and
  `schemas/dynamic.py` elsewhere in the codebase — a rate/slab change next
  AY should be a data edit here, not a logic change.
- **`calculator.py`** — pure-Python, **zero-LLM** tax computation: applies
  `tax_rules.py`'s rules to the stitched profile, computes liability under
  both regimes so the agent can report which is more favorable, and
  handles capital gains. Deliberately kept LLM-free for the same reason
  `schemas/validator.py`'s duration verification is computed in Python, not
  by the model — tax math must be deterministic and auditable, not a
  plausible-sounding LLM output.
- **`agent.py`** — the 5-stage pipeline: **extract → stitch → validate →
  calculate → summarize**. Pauses at `needs_input` if stitching finds no
  Form16 among the uploaded documents (can't meaningfully compute salary
  tax without it) — the user supplies missing info via the existing
  `answers`-based resume path; a later document upload that changes
  numbers goes through the newer `new_input`-based resume path instead
  (`plan_resume_stage()` in `base.py` picks the right re-entry stage, e.g.
  re-stitch and re-calculate but not re-validate, depending on what
  changed).
- **`chat_tools.py`** — registers two tools via the factory-pattern
  `chat_tools_factory(run_id)` in `agents/registry.py`: **`recalculate_tax`**
  (re-runs `calculator.py` against the current stitched profile — read-only
  in the sense that it doesn't mutate the stored run, it just recomputes
  and reports) and **`lookup_extracted_doc`** (returns a specific
  previously-extracted document's fields on request, e.g. "what did my
  second Form16 say again"). Both tools are doc-ownership-checked — a tool
  call can't be used to read another user's extracted documents even
  indirectly through chat.
**Why it exists:** The first agent built on top of the Phase A shared
infrastructure (tool-calling chat, dual-trigger resume) rather than
extending the original single-purpose `cv_processor` pipeline — validates
that infrastructure is genuinely agent-agnostic.
**Known v1 gaps (documented, deliberate):** no house-property or business
income, no Section 80TTA/80TTB deductions, no STCG/LTCG netting across
transactions, and no RAG-backed chat tool (chat can look up an already-
extracted document or recalculate, but can't search/reason over a
document's full text the way document Q&A chat can).

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
**Key thing to know:** `RULESET_MAP` includes `"dynamic": "validation.rulesets.generic"` — `retrieval.extract_dynamic_fields()` always passes `doc_type="dynamic"`, since a user-defined schema's field names aren't known in advance and the doc-type-specific rulesets (`invoice.py`, `cv_resume.py`, etc.) can't apply to it.

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
**Relationship to `nested_schema_templates` (Phase B):** this file is still the library for the original flat/doc-type-keyed templates; the new `nested_schema_templates` **table** (via `routers/schema_templates.py`) is a separate, user-facing, DB-backed mechanism for saving/reusing arbitrary nested (`SchemaSpec`) schemas across requests. The ITR agent's three schemas live in the DB table, seeded at `global` scope, not in this file.

### `backend/schemas/dynamic.py`
**What it is:** The recursive meta-schema DSL that powers nested/dynamic extraction.
**What it does:** Describes any extraction schema — flat or arbitrarily nested — as data, not code, so a user-submitted schema and an NL-generated schema go through the exact same extraction path with no per-domain branching.
- `FieldSpec` — recursive Pydantic model: `name`, `type` (`string` / `number` / `integer` / `date` / `boolean` / `list` / `object`), `description`, and optional `properties: list[FieldSpec]` for `list`/`object` types describing the nested shape. Calls `model_rebuild()` since it's self-referencing.
- `SchemaSpec` — `schema_name` + `fields: list[FieldSpec]`. The top-level object both an inline `nested_schema` request and a persisted `nested_schema_templates` row's `spec` conform to.
- `generate_schema_spec(description, user_id)` — calls the LLM (`response_model=SchemaSpec`) to turn a free-text instruction into a `SchemaSpec`, using `prompts.SCHEMA_GEN_SYSTEM`.
- `spec_to_model(spec)` — recursively builds a real Pydantic model via `pydantic.create_model`: a `list` field with `properties` becomes `list[<recursively built item model>]`, an `object` field becomes `Optional[<recursively built model>]`, everything else maps to a primitive type. Results are cached in-process, keyed by an MD5 hash of the spec's JSON, so reusing the same schema doesn't rebuild the model every call.
- `spec_to_field_descriptions(spec)` — renders a `SchemaSpec` as an indented, human-readable field list for the extraction prompt (the nested equivalent of the old flat `fields_with_desc` string).
**Why it exists:** Before this, every "list of structured items" use case needed a hardcoded Pydantic model and a hardcoded prompt. This makes nesting a schema property instead of a code change.
**Key thing to know:** Building the Pydantic model at runtime is what lets Instructor validate arbitrarily deep nesting the same way it validates a flat model — the LLM structured-output guarantee extends to nested shapes for free. The in-process model cache (keyed by spec JSON hash) means a `nested_schema_templates` row and an identical inline `nested_schema` request share the same cached compiled model.

### `backend/schemas/validator.py`
**What it is:** Field-level confidence scoring, plus nested-schema confidence and date-verification logic.
**What it does:** For each extracted field, detects the field type (email, phone, date, amount, text, list), validates the format, scores confidence 0.0–1.0, and returns `FOUND` / `LOW_CONFIDENCE` / `NOT_FOUND` status.
**Why it exists:** Separate from the business logic validation engine — this runs on every extraction regardless of doc type. The business validation in `validation/` runs on top of this.
**Key thing to know:**
- `detect_field_type()` and `detect_verifiable_pairs()` use **token-based** matching (`name.split("_")`, set intersection) rather than substring matching — a substring check once misclassified `candidate_name` as a `date` field because `"date"` is a substring of `"candi-date"`. Don't reintroduce substring matching here.
- **Nested confidence scoring** — `score_nested_list_confidence()` (average non-null-field ratio across list items) and `score_nested_object_confidence()` (non-null ratio of a nested object's sub-fields), dispatched from `score_confidence()` based on value shape, so a half-populated list and a fully-populated list no longer score identically. `validate_extraction()` only runs format validation (email/phone/date/amount/url regex) against flat string values — nested values go straight to shape-based confidence.
- **Verification (computed, not extracted)** — `compute_duration(start, end)` does deterministic date math in Python (never the LLM); missing `end` is treated as "ongoing" against today. `detect_verifiable_pairs(item_fields)` finds a start/end pair (and optional stated-duration field) in a list item by name convention (`start`/`end`/`duration`/`tenure`/`time_spent` tokens) — types `date`/`string`/`integer`/`number` are all eligible, booleans and lists are not. `verify_item(item, pair)` computes the duration and compares it to any stated value. `verify_dynamic_extraction(extracted, spec)` is the orchestrator, called from `retrieval.extract_dynamic_fields()` right after the LLM returns.
- **Backfill** — if a stated-duration field is empty and a duration was computed, the computed value is written into that field and `_verification.backfilled = true` — never overwrites a stated value (a disagreement is surfaced as `"match": false` instead). Verification failures/gaps are silent-safe — they annotate `_verification`, never block or crash extraction.
- Date parsing handles resume-style formats like `"Sept'25"` via an explicit `_MONTH_MAP` + regex matcher in `_parse_date_loose()`, tried before the fixed `_PARSE_FORMATS` list — `dateutil.parser(fuzzy=True)` was tried and rejected because it silently misparses `"Sept'25"` as day=25 of the current year. The same deterministic-over-LLM philosophy carries into `itr_helper/calculator.py`'s tax math (see above) — this file was the precedent for "compute it in Python, verify/backfill, never trust the LLM for arithmetic."

---

## Backend — llm/

---

### `backend/llm/engine.py`
**What it is:** The central LLM caller — every LLM call in the system goes through here.
**What it does:** Selects the right client (Groq, OpenAI, Anthropic) from config, calls the API, handles retries with exponential backoff on rate limits, parses JSON responses when `json_mode=True`, and supports streaming.
**Why it exists:** Before this, every function had its own Groq client and its own retry logic. Now there's one place to swap models, add logging, or change retry behaviour.
**Key thing to know:** To switch LLM providers, change `LLM_PROVIDER` in `.env`. No code changes needed. The `call_llm()` function signature is the same regardless of provider. `call_llm()` / `_call_single_provider()` also thread an optional `structured_max_retries` parameter down to `llm/structured.py`'s `call_structured()`. Default is `0` (no behavior change) everywhere except `retrieval.extract_dynamic_fields()` (passes `2`) and `llm/tool_orchestrator.py`'s per-round `ToolCallBatch` parse (passes `1`) — both are cases where a structured/nested shape fails Instructor's validation on the first pass more often than a flat model, so letting Instructor self-correct with the validation error improves reliability without extra app-level retry logic. `call_llm()` accepts a single `system` + `user` string pair only — no `messages=[...]` array — which is why both `agents/chat.py` and `llm/tool_orchestrator.py` fold prior turns/tool results into the `user` string as plain text rather than passing structured history. **`call_llm()` itself was never modified** in a way that changes its signature by Phase A, Phase B, or the CV Processor work — every new capability is a new *caller* of it (or of `call_structured()`), reusing the existing `response_model` (Instructor) path.
**PII masking (new — see `Backend — pii/` below):** on every **non-streaming** call, `call_llm()` now builds a PII mapping from the `system + user` text, masks before sending to the provider, and unmasks the result before returning or caching it. This is internal to `call_llm()` — no caller had to change. **Streaming calls (`stream=True`, `_call_stream()`) are not yet masked** — token-by-token delivery means a placeholder could be split across chunk boundaries, which needs a separate buffering design; this is a known, parked gap (see `pii/` section and "What's not built yet"). Vision calls (`call_vision_llm`) are also not covered — only text prompts are masked, images are sent as-is.

### `backend/llm/tool_orchestrator.py` (new, Phase A)
**What it is:** A custom tool-calling loop layered **on top of** `call_llm()`.
**What it does:**
- **`ToolSpec`** — dataclass describing one callable tool: `name`, `description`, `executor` (the actual callable), and an optional args-schema description. Any agent builds its own list (statically, like Phase A's original design, or via a `chat_tools_factory` as `itr_helper` does — see `agents/registry.py`).
- **`ToolCallBatch`** — the Instructor `response_model` used for each round: `{calls: [...], final_answer}`. An empty `calls` list with `final_answer` set signals the model is done.
- **`run_tool_loop(system, user, tools, call_type, ...)`** — the public entrypoint. Each round: one `call_llm(response_model=ToolCallBatch)` call; every tool call returned that round is executed (**sequentially**, in call order — not concurrently, not via native parallel tool-calling); results are folded back into the prompt as plain text (matching `agents/chat.py`'s existing history-folding style); loops until `final_answer` is set or `max_rounds` is hit.
**Why it exists (why custom, not native provider tool-calling):** reuses the existing Instructor/`response_model` structured-output path that every extraction call site already depends on, so the orchestrator is provider-agnostic for free — exactly the same reason `call_llm()` itself is provider-agnostic.
**Key thing to know:**
- An unknown tool name, or a tool `executor` that raises, does **not** crash the loop — it's fed back to the model as an error string in the next round's prompt so the model can self-correct (e.g. call a different tool, or give up and answer with what it has).
- **Accepted trade-off:** tools called within the *same* round must be independent of each other's output, since none of that round's tool calls have executed yet at the point the model decides to call all of them. A tool needing another tool's result within one turn requires two rounds, not one.
- `call_type` is a **required** parameter with **no shared default** — every caller must supply its own (e.g. `agents/chat.py` passes `f"agent_chat_tools:{agent_name}"`), so usage/cost tracing never blurs different agents' tool-calling activity together.

### `backend/llm/structured.py`
**What it is:** Instructor-based structured-output calling.
**What it does:** `call_structured()` wraps Instructor's `.create()` to get a validated Pydantic object back from an LLM call (used for `SchemaSpec` generation, dynamic-schema extraction, `ToolCallBatch` rounds in `tool_orchestrator.py`, and `plan_resume_stage()`'s `_StageSelection` in `agents/base.py`). Takes a `max_retries` param passed straight to Instructor's own retry mechanism, which re-prompts the model with the validation error on failure.

### `backend/llm/usage.py`
**What it is:** Token usage tracking.
**What it does:** Logs every LLM call to an in-memory buffer and writes to the `usage_logs` Supabase table. Provides `get_usage_summary()` for the `/usage` endpoint.
**Why it exists:** You need to know what things cost before you can charge for them.
**Key thing to know:** Agent chat calls are logged with `call_type="agent_chat"` (plain) or `call_type="agent_chat_tools:{agent_name}"` (tool-calling), and `session_id=<run_id>`, so usage/cost can be broken out per agent and per chat mode, distinct from document Q&A and extraction calls.

---

## Backend — pii/ (new)

PII masking for outbound LLM calls — every prompt is scanned for personal
data (names, PAN, Aadhaar, email, phone, DOB, addresses) and masked with
placeholders before it leaves the system; the LLM only ever sees
placeholders, never real user data; placeholders are swapped back for real
values once the response comes back.

**Why call-scoped, not stored:** agents now combine **multiple documents
into a single LLM call** (e.g. scoring several candidates' resumes
together, or evaluating several tax documents at once). A mapping tied to
one document can't safely cover a multi-document prompt without risking
placeholder collisions between different people's data. Building the
mapping fresh, from the actual prompt content, right before each call,
sidesteps this entirely — masking is always scoped correctly to exactly
what's in that call. No mapping is ever persisted — it's built in memory,
used once, and discarded immediately after that call's response is
unmasked.

**Pipeline (per call):**
```
system + user text
    ↓
regex pass — PAN, Aadhaar, email, phone, numeric DOB (matches blanked before next pass)
    ↓
Presidio (NER) pass — PERSON, ADDRESS, natural-language DOB (e.g. "Oct 2, 1970")
    ↓
name clustering — merges surface forms of the same person (e.g. "Sukanya Verma" / "S. Verma")
    ↓
mask() → LLM call (provider only ever sees placeholders) → unmask()
    ↓
returned to the caller / cached (real data only)
```
Regex runs first and its matches are stripped before Presidio runs, so
Presidio never mis-tags something like a PAN as a name.

### `backend/pii/pii_config.py`
**What it is:** Registry of entity types.
**What it does:** Regex patterns, Presidio entity mappings, and an enable/disable flag per entity type (PAN, Aadhaar, email, phone, numeric DOB, natural-language DOB, person names, address).
**Why it exists:** Config-driven so an entity type can be added, disabled, or reconfigured (pattern, placeholder label) without touching detection or masking logic elsewhere.

### `backend/pii/presidio_setup.py`
**What it is:** Presidio analyzer bootstrap.
**What it does:** Loads the Presidio analyzer once at process startup (`en_core_web_lg` model) and exposes a restricted `analyze_text()` scoped to only the entity types enabled in `pii_config.py`.
**Why it exists:** Loading the NER model is expensive — done once per process, not per call.

### `backend/pii/pii_mapping.py`
**What it is:** Mapping builder.
**What it does:** `build_mapping(text, user_id)` runs the full regex → Presidio → clustering pipeline and returns a list of `{placeholder, real_values, entity_type, source}`.
**Key thing to know — name clustering is deliberately conservative:** only two rules merge two name mentions into one placeholder — an exact full-name match, or a surname match where that surname is unique in the document. No fuzzy/similarity-based merging — wrongly merging two different people is treated as worse than leaving two mentions of the same person unmerged.
**Placeholder format:** `__PII_{user_id}_{ENTITY_TYPE}_{n}__` (e.g. `__PII_user-abc_PERSON_1__`) — deliberately unusual/unique so it's extremely unlikely to collide with real text that happens to appear in a document.

### `backend/pii/pii_masking.py`
**What it is:** The mask/unmask primitives.
**What it does:** `mask(text, mapping)` / `unmask(text, mapping)` — pure string substitution, no side effects, no I/O.

**Failure behavior — fail-open (deliberate):** if detection fails for any reason (a Presidio error, etc.), the call proceeds **unmasked** rather than being blocked — a warning is logged, but the user's request still goes through. Availability was prioritized over blocking a call entirely.

**What masking does not cover (known gaps):** streaming calls (`call_llm(..., stream=True)` / `_call_stream()` — affects `query_document_stream()` and any other streaming call site, parked for a follow-up buffering design) and vision LLM calls (`call_vision_llm` — images sent as-is, only text prompts are masked).

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

**Key thing to know:** All API calls use `headers=_auth_headers()` to send the JWT. If you add a new API call and forget this, it will work in dev mode but fail in production. UI/API-client changes for agents, agent chat (plain and tool-calling), schema templates, and the ITR agent are covered in the companion implementation docs rather than here.

---

## Database — Supabase tables (new/changed across recent phases)

### `agent_runs.name` (column)
```sql
alter table agent_runs add column if not exists name text;
```
Nullable, so existing rows are unaffected. Set only at invoke time via `InvokeAgentRequest.name` — no rename-later UI. UI falls back to the run's `id` when `NULL`.

### `agent_chat_messages` (table)
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
Keyed by `run_id` (scoped to a single agent run) rather than `document_id` like the existing `chats` table. RLS mirrors `agent_runs` — single-user ownership. Originally `select`/`insert` only (messages are immutable once written); a `delete` policy was added in Phase A (see below).

### `agent_run_flags` (new table, CV Processor phase)
Holds diagnostic/audit flags raised during a run — bias-audit hits, duplicate/contradiction detections, ranking-inconsistency flags — written via `db.insert_agent_flag()` and read via `db.get_flags_for_run()`. Scoped by `run_id` and `user_id`, RLS mirrors `agent_runs`. Kept separate from `agent_runs.result` so the result's shape stays exactly the fixed contract chat and other consumers depend on (see `agents/chat.py` and `base.py`'s `REQUIRED_RESULT_KEYS`).

### `nested_schema_templates` (new table, Phase B)
Persisted, reusable `SchemaSpec` schemas at four scopes — `personal`, `team`, `org`, `global` — backed by `routers/schema_templates.py` and `db.py`'s schema-template functions. Seeded with the three ITR schemas (Form16, FD/interest, stocks) at `global` scope. **Known gap:** the table's own `INSERT` RLS policy still checks an `org_admin`-style condition, while the app-level gate for `global`-scope inserts checks `platform_admins` — the two aren't yet reconciled (see `routers/schema_templates.py` above and "What's not built yet" below).

### `platform_admins` (new table, Phase B)
```sql
-- shape: user_id uuid primary key references auth.users(id)
```
Simple allow-list table — presence of a row means that `user_id` is a platform admin. Checked by `db.is_platform_admin()`, consumed by `core/auth.py` to populate `UserContext.is_platform_admin` (fails safe to `False`, checked even in dev mode). Gates `global`-scope schema template creation, among other admin-only actions.

### Delete RLS policies (new migration, Phase A)
`agent_runs` and `agent_chat_messages` already had proper per-user RLS (`INSERT`/`SELECT`/`UPDATE`, each `auth.uid() = user_id`) from when they were first created — the one gap found on inspection was that **neither had a `DELETE` policy**. Added both, matching the existing pattern exactly (`auth.uid() = user_id`, no org/team join). This migration also established the reusable template followed by `agent_run_flags` and other new agent-related tables since: `uuid user_id not null`, four separate policies (`INSERT`/`SELECT`/`UPDATE`/`DELETE`), each a plain `auth.uid() = user_id` check.

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
execute_stages() → runs the agent's multi-stage plan
    ↓
validate final result against REQUIRED_RESULT_KEYS → mark "completed", or "failed" if malformed
    ↓
UI polls GET /agents/runs/{run_id} until status == "completed" (or "needs_input" / "failed")
```

### Agent resume flow (two triggers)
```
POST /agents/runs/{run_id}/resume (with JWT + answers OR new_input — mutually exclusive)
    ↓
routers/agents.py: check run's live status against which field was supplied
    mismatch → 400 (AGENT_RESUME_ANSWERS_REQUIRED / AGENT_RESUME_NEW_INPUT_REQUIRED / AGENT_RUN_NOT_RESUMABLE)
    ↓
agents.base.accept_resume_input(run_id, answers=None, new_input=None)
    ↓
status == "needs_input" + answers  → resume at current_stage (unchanged original path)
status == "completed" + new_input  → merge new_input into state["input_data"]
                                       → plan_resume_stage() [LLM call, grounded in
                                         registry.py's stage_descriptions] → pick re-entry stage
                                       → AgentRunError if stage_descriptions missing or
                                         model picks an unknown stage (no self-correct retry)
    ↓
(background, via core.queue.task_queue.submit) resume_stages() re-executes from the chosen stage
```

### Agent chat flow (plain or tool-calling)
```
POST /agents/runs/{run_id}/chat (with JWT + message) — synchronous, no task_queue
    ↓
routers/agents.py → look up agent_def in agents/registry.py by the run's agent_name
    → chat_tools = agent_def["chat_tools_factory"](run_id) if present, else []
    ↓
agents.chat.handle_chat_message(run_id, message, user_id, chat_tools)
    ↓
db.get_agent_run(run_id, user_id) →
    not found / not owned / not chat-eligible → AgentChatError → router returns 400 AGENT_CHAT_UNAVAILABLE
    ↓
db.get_agent_chat_history(run_id, user_id) → prior turns
    ↓
_format_run_context(run) → run["result"] only (not state), truncated to MAX_CONTEXT_CHARS
_format_history(history) → last MAX_HISTORY_TURNS folded into plain "User:/Assistant:" text
    ↓
if chat_tools empty:
    call_llm(system=AGENT_CHAT_SYSTEM, user=user_prompt, call_type="agent_chat", session_id=run_id)
else:
    tool_orchestrator.run_tool_loop(system=AGENT_CHAT_SYSTEM_WITH_TOOLS, user=user_prompt,
                                     tools=chat_tools, call_type=f"agent_chat_tools:{agent_name}")
    (each round: call_llm(response_model=ToolCallBatch) → execute that round's tool calls
     sequentially → fold results back into prompt as text → repeat until final_answer)
    ↓
empty final reply → AgentChatError (400) — no blank message ever saved
    ↓
db.save_agent_chat_message(run_id, "user", message, user_id)
db.save_agent_chat_message(run_id, "assistant", reply, user_id)  ← only the final_answer, not intermediate tool turns
    ↓
return {"role": "assistant", "content": reply}
```

### ITR agent run flow (illustrative — stage plan, not a new endpoint)
```
POST /agents/itr_helper/invoke (document_ids = uploaded Form16 / FD / stock statements)
    ↓
Stage 1 — extract: per document, retrieval.extract_dynamic_fields() against the
                    matching nested_schema_templates entry (Form16 / FD-interest / stocks)
                    → failures isolated per document, not fatal to the run
    ↓
Stage 2 — stitch: helpers.py combines per-document extractions into one taxpayer
                   profile; multiple Form16 entries kept separate, not pre-summed
    ↓
    no Form16 found → pause at status="needs_input" (answers-based resume to continue)
    ↓
Stage 3 — validate: sanity checks on the stitched profile
    ↓
Stage 4 — calculate: calculator.py (pure Python, zero-LLM) applies tax_rules.py
                      → liability under both regimes + capital gains
    ↓
Stage 5 — summarize: assembles result (summary/findings/data/extra) →
                      base.py validates against REQUIRED_RESULT_KEYS → status="completed"
    ↓
Later: new document uploaded → POST .../resume (new_input) → plan_resume_stage()
       picks re-entry (e.g. back to stitch+calculate, skip re-validate) → re-executes
    ↓
Chat: recalculate_tax / lookup_extracted_doc tools available via chat_tools_factory(run_id)
```

### Crash recovery flow (restart stuck runs)
```
Backend process/pod dies mid-run → agent_runs row left at status="running" forever
    ↓ (operator confirms the old process is actually dead — no automatic detection)
POST /agents/admin/restart-stuck-runs (platform admin JWT)
    ↓
routers/agents.py → check user.is_platform_admin → 403 RESTART_STUCK_RUNS_FORBIDDEN if not
    ↓
db.get_running_agent_runs() → every status="running" row, across all users
    ↓
for each run:
    get_agent_def(run["agent_name"])
        not registered → skip, add to "skipped" list with reason
        registered → task_queue.submit(agent_base.resume_stages(run_id, agent_def["stages"], user_id=run["user_id"]))
    ↓
re-enters _run_stage_loop() at last-persisted current_stage
    → that stage re-runs from scratch (safe — every stage is idempotent given the same input state)
    → pipeline continues normally from there
    ↓
return {"found": N, "restarted": [...], "skipped": [...]}
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
→ Nothing to add per-domain — send a `nested_schema` (or an NL `instruction`) describing the fields; `schemas/dynamic.py` builds the model at runtime. If the schema needs to be reused across requests/users rather than sent inline every time, save it via `routers/schema_templates.py` instead. Only add code if you need a new schema-agnostic validation rule (`validation/rulesets/generic.py`) or need to widen the verification name-convention tokens in `detect_verifiable_pairs()` (`schemas/validator.py`).

**Add a schema-agnostic validation rule (applies to any dynamic/nested extraction):**
→ Add to `backend/validation/rulesets/generic.py`, matching the real `ValidationResult` fields in `validation/rules/base.py` exactly — a constructor mismatch fails silently.

**Debug an agent chat that isn't answering, or is missing context:**
→ Check whether the run's status is actually chat-eligible first. If it answers but seems to be missing detail that exists in the run, check whether that detail lives in `result.extra` — `agents/chat.py` only ever sees `result`, never the raw `state`, so anything not surfaced into `result` at the end of the agent's stages won't be visible in chat. If the agent has `chat_tools`, also check `llm_calls` tracing under `call_type=f"agent_chat_tools:{agent_name}"` to see the round-by-round tool-calling transcript.

**Add a new agent (beyond `cv_processor` / `itr_helper`):**
→ Build its multi-stage plan on `agents/base.py`'s stage machine, and make sure its final `result` always includes `summary` / `findings` / `data` / `extra` (`REQUIRED_RESULT_KEYS`) or the run will be marked `failed` instead of `completed`. Chat support requires no extra work for plain read-only Q&A — `agents/chat.py`/`chat_prompts.py` are agent-agnostic. For tool-calling chat, register a `chat_tools_factory` (not a static `chat_tools` list, if any tool needs to be scoped to `run_id`) in `agents/registry.py`. For a "new input on a completed run" resume path, register `stage_descriptions` in the same registry entry.

**Add or reuse a persisted extraction schema:**
→ `POST /schema-templates` with a `SchemaSpec`-shaped `spec` and a `scope`. `global`-scope creation requires `is_platform_admin`; `team`/`org` requires membership. Fetch via `GET /schema-templates` (returns everything the caller can see) or `GET /schema-templates/{id}`.

**Grant platform-admin access to a user:**
→ Insert a row into `platform_admins` for that `user_id` directly (no self-serve endpoint, by design). `core/auth.py` picks it up on their next request via `db.is_platform_admin()`.

**Recover an agent run stuck at `status="running"` after a crash:**
→ Confirm the old process/pod is actually dead first (no automatic detection). Then, as a platform admin, call `POST /agents/admin/restart-stuck-runs` — it finds every `running` run via `db.get_running_agent_runs()` and resubmits each to `resume_stages()` from its last persisted `current_stage`. Works for any registered agent; unregistered agent types are skipped and reported, not hard-failed.

**Add or reconfigure a PII entity type (masking):**
→ Edit `backend/pii/pii_config.py` — add/adjust a regex pattern or Presidio entity mapping, or flip its enable/disable flag. No changes needed in `pii_mapping.py`, `pii_masking.py`, or `llm/engine.py` for a config-only change.

**Add a new batching/chunking knob for the CV processor's optimization pass:**
→ Add it to `backend/agents/cv_processor/optimization_config.py`, not inline in `agent.py` or `scoring.py` — that's the one place every tunable for `verify_claims`/`score_skills`/`compute_adjusted_scores` batching lives.

---

## What's not built yet

- **Actions/tool-calling beyond what's registered** — chat tool-calling exists (`itr_helper`), but is opt-in per agent via `chat_tools_factory`; `cv_processor` remains read-only Q&A only, by design, not yet by limitation.
- **`nested_schema_templates` RLS/app-gate mismatch** — the table's own `INSERT` RLS still checks an `org_admin`-style condition left from an earlier draft, while the app-level gate for `global` inserts checks `platform_admins`. Documented, not yet reconciled.
- **ITR agent v1 scope gaps** — no house property or business income, no Section 80TTA/80TTB deductions, no STCG/LTCG netting across transactions, and no RAG-backed chat tool for the ITR agent (only recalculate + doc lookup).
- **PII masking doesn't cover streaming or vision calls** — `call_llm(..., stream=True)` (token-by-token delivery risks splitting a placeholder across chunk boundaries — needs a separate buffering design) and `call_vision_llm` (images sent as-is) are both unmasked today. Affects `query_document_stream()` and any other streaming call site.
- **CV processor: no cascading re-verification** — if `verify_claims` demotes a borderline top-N/top-percent candidate post-hoc, whoever would replace them in the shortlist is never itself verified. Accepted tradeoff, not a bug.
- **CV processor: `score_skills` can increase total call count** in the worst case (skills that don't semantically merge), despite reducing prompt-size risk — an explicit, discussed tradeoff, flagged as worth revisiting if it becomes a real cost driver.
- **Crash recovery is single-instance-safe only** — `restart-stuck-runs` assumes any `status="running"` row is abandoned, which is only guaranteed true because deployment is currently single-instance. Multi-instance deployment would need real liveness detection before this stays safe to use as-is.
- **Document comparison** — diff two versions of a document
- **Vertical wrappers beyond CV Screener and ITR Helper** — CA Helper, Loan Processor, etc. (workflow configs on top of the agent engine)
- **Password reset** — currently manual via Supabase Dashboard
- **Session refresh** — JWT expires after 24h, user must log in again
- **Tighter RLS policies on the older, non-agent tables** — most non-agent tables are currently `allow_all`, should move to `user_id = auth.uid()`. `agent_runs`, `agent_chat_messages`, `agent_run_flags` already use proper `auth.uid()`-scoped RLS (including delete policies as of Phase A), ahead of the rest of the schema.
- **Sub-field correction loop** — `build_correction_examples()` only applies to top-level field names on dynamic schemas; sub-fields inside nested list items aren't individually correctable yet.

---

*Last updated: this phase*
*System status: Phases 0, 1, 2 complete — deployed on Railway + Streamlit Cloud. Nested/dynamic schema extraction implemented and tested end-to-end. Agent layer live: `cv_processor` (10-stage pipeline with guardrails, explainability, and ranking review, since re-optimized for call volume/prompt size at scale) and `itr_helper` (5-stage tax filing agent with dual-trigger resume and tool-calling chat), on top of shared agent-agnostic infrastructure (tool-calling orchestrator, resume answers/new_input split, schema template persistence, platform admin, crash recovery for abandoned runs). All outbound non-streaming LLM calls now pass through call-scoped PII masking.*