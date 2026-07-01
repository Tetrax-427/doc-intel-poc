"""
retrieval.py
All retrieval, extraction, classification, and query functions.

Changes in this phase (Security):
  - sandbox_and_check() applied at 4 document-content call sites:
      extract_fields(), generate_summary(), extract_tables(),
      _classify_from_context()
  - sanitize_llm_output() applied to query_document() and
      query_document_stream() responses
  - org_id/team_id threaded through to call_llm() where available
"""

import os
import json
import numpy as np
from typing import Generator
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from db import get_chunks_by_document, get_parent_chunk
from ingestion import get_embed_model
from prompts import (
    QA_SYSTEM, QA_MULTI_SYSTEM,
    GENERAL_SYSTEM,
    CLASSIFIER_SYSTEM,
    QUERY_EXPANSION_SYSTEM,
    EXTRACTION_SYSTEM,
    DOCUMENT_CLASSIFIER_SYSTEM,
    KEYWORD_SIGNALS,
    EMBEDDING_EXEMPLARS,
)
from db import get_corrections_for_doc_type
from llm.engine import call_llm, call_llm_stream
from llm.sanitizer import sandbox_and_check, sanitize_llm_output
from llm.structured import (
    DocumentClassification,
    QueryExpansion,
    DocumentSummary,
    TableItem,
    TableList,
    SchemaResult,
    build_extraction_model,
)
from hyde import (
    generate_hyde_passage,
    generate_query_variants,
    merge_and_dedupe,
    normalise_retrieval_mode,
)
from schemas.validator import validate_extraction
from validation.engine import ValidationEngine

load_dotenv()

from core.logger import get_logger as _get_logger
from schemas.templates import get_template_for_doc_type
from core.config import config as app_config

import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))
_clf_logger = _get_logger("retrieval.classify")
logger = _get_logger("retrieval")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def format_history(messages: list[dict], summary: str = "") -> str:
    return get_history_context(messages, summary)


def get_history_context(messages: list[dict], summary: str = "", window: int = 4) -> str:
    parts = []
    if summary:
        parts.append(f"[Earlier conversation summary]: {summary}")
    recent = messages[-window:] if len(messages) > window else messages
    if recent:
        recent_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in recent
        ])
        parts.append(f"[Recent messages]:\n{recent_text}")
    return "\n\n".join(parts) if parts else "None"


# ---------------------------------------------------------------------------
# Source formatting
# ---------------------------------------------------------------------------

def format_source(chunk: dict) -> dict:
    try:
        chunk_type = chunk.get("chunk_type", "text")
        base = {
            "chunk":          chunk.get("chunk_num"),
            "page":           chunk.get("page"),
            "file":           chunk.get("file"),
            "preview":        chunk.get("content", "")[:150],
            "chunk_type":     chunk_type,
            "exact_sentence": "",
        }
        if chunk_type == "figure":
            base["image_ref"] = chunk.get("image_ref")
            base["caption"]   = chunk.get("caption", "")
            base["bbox"]      = chunk.get("bbox")
        elif chunk_type in ("description", "table"):
            base["image_ref"] = chunk.get("image_ref")
        return base
    except Exception:
        return {
            "chunk":          chunk.get("chunk_num"),
            "page":           chunk.get("page"),
            "file":           chunk.get("file"),
            "preview":        chunk.get("content", "")[:150],
            "chunk_type":     chunk.get("chunk_type", "text"),
            "exact_sentence": "",
        }


# ---------------------------------------------------------------------------
# Classification (question type)
# ---------------------------------------------------------------------------

def classify_question(
    question: str,
    has_document: bool = True,
    user_id: str = "system",
) -> str:
    if not has_document:
        return "general"

    doc_keywords = [
        "this", "the document", "the file", "the letter", "the contract",
        "the invoice", "the report", "the resume", "the agreement", "it says",
        "mentioned", "above", "here", "uploaded", "about"
    ]
    if any(kw in question.lower() for kw in doc_keywords):
        return "document"

    try:
        result = call_llm(
            system=CLASSIFIER_SYSTEM,
            user=question,
            temperature=0.0,
            max_tokens=10,
            call_type="classify",
            user_id=user_id,
        )
        return "document" if "document" in result.strip().lower() else "general"
    except Exception:
        return "document"


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(question: str, user_id: str = "system") -> str:
    try:
        result: QueryExpansion = call_llm(
            system=QUERY_EXPANSION_SYSTEM,
            user=question,
            temperature=0.0,
            max_tokens=100,
            call_type="expand",
            response_model=QueryExpansion,
            user_id=user_id,
        )
        expanded = result.expanded_query.strip()
        return expanded if expanded else question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank_chunks(question: str, chunks: list[dict], top_k: int = None) -> list[dict]:
    effective_top_k = top_k if top_k is not None else app_config.retrieval_top_n
    if not chunks or not os.getenv("COHERE_API_KEY"):
        return chunks[:effective_top_k]
    try:
        results = co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=[c["content"] for c in chunks],
            top_n=effective_top_k,
        )
        reranked = []
        for r in results.results:
            chunk = chunks[r.index].copy()
            chunk["score"] = r.relevance_score
            reranked.append(chunk)
        for i, c in enumerate(reranked):
            c["chunk_num"] = i + 1
        return reranked
    except Exception as e:
        logger.warning("Reranking failed — using original order", error=str(e))
        return chunks[:effective_top_k]


# ---------------------------------------------------------------------------
# Parent context expansion
# ---------------------------------------------------------------------------

def expand_to_parent_context(chunks: list[dict]) -> list[dict]:
    if not app_config.hierarchical_expand_to_parent:
        return chunks

    expanded = []
    for chunk in chunks:
        level     = chunk.get("metadata", {}).get("chunk_level", "flat")
        parent_id = chunk.get("metadata", {}).get("parent_chunk_id")

        if level == "child" and parent_id:
            try:
                parent = get_parent_chunk(parent_id)
                if parent:
                    expanded_chunk = chunk.copy()
                    expanded_chunk["content"] = parent.get("content", chunk["content"])
                    expanded_chunk["_expanded_from_child"] = True
                    expanded.append(expanded_chunk)
                    continue
            except Exception as exc:
                logger.warning(
                    "Parent chunk lookup failed — using child text",
                    parent_id=parent_id, error=str(exc),
                )
        expanded.append(chunk)

    return expanded


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    question: str,
    document_ids: list[str] = None,
    candidate_pool: int | None = None,
    top_n: int | None = None,
    dense_query_override: str | None = None,
    user_id: str = "system",
) -> list[dict]:
    pool_size = candidate_pool if candidate_pool is not None else app_config.retrieval_candidate_pool
    n_results = top_n          if top_n          is not None else app_config.retrieval_top_n

    if n_results > pool_size:
        logger.warning("top_n > candidate_pool — clamping", top_n=n_results, pool=pool_size)
        n_results = pool_size

    embed_model = get_embed_model()

    if document_ids:
        all_chunks = []
        for doc_id in document_ids:
            chunks = get_chunks_by_document(doc_id)
            all_chunks.extend(chunks)
    else:
        return []

    if not all_chunks:
        return []

    searchable = [
        c for c in all_chunks
        if c.get("metadata", {}).get("chunk_level", "flat") != "parent"
        and c.get("embedding") is not None
    ]

    if not searchable:
        return []

    expanded_question  = expand_query(question, user_id=user_id)
    dense_query_text   = dense_query_override if dense_query_override else expanded_question
    question_embedding = embed_model.get_text_embedding(dense_query_text)

    texts        = [c["content"] for c in searchable]
    dense_scores = []
    for chunk in searchable:
        raw_emb = chunk["embedding"]
        if isinstance(raw_emb, str):
            raw_emb = json.loads(raw_emb)
        dense_scores.append(cosine_similarity(question_embedding, raw_emb))

    tokenized   = [t.lower().split() for t in texts]
    bm25        = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(expanded_question.lower().split())

    def rrf_score(rank, k=60):
        return 1 / (k + rank + 1)

    dense_ranked = np.argsort(dense_scores)[::-1]
    bm25_ranked  = np.argsort(bm25_scores)[::-1]

    rrf = {}
    for rank, idx in enumerate(dense_ranked):
        rrf[idx] = rrf.get(idx, 0) + rrf_score(rank)
    for rank, idx in enumerate(bm25_ranked):
        rrf[idx] = rrf.get(idx, 0) + rrf_score(rank)

    top_indices = sorted(rrf, key=rrf.get, reverse=True)[:pool_size]

    candidates = [
        {
            "chunk_num":       i + 1,
            "content":         searchable[idx]["content"],
            "page":            searchable[idx]["metadata"].get("page", "?"),
            "file":            searchable[idx]["metadata"].get("file", "?"),
            "chunk_type":      searchable[idx]["metadata"].get("chunk_type", "text"),
            "image_ref":       searchable[idx]["metadata"].get("image_ref"),
            "chunk_level":     searchable[idx]["metadata"].get("chunk_level", "flat"),
            "parent_chunk_id": searchable[idx]["metadata"].get("parent_chunk_id"),
            "score":           rrf[idx],
            "bbox":            searchable[idx]["metadata"].get("bbox"),
        }
        for i, idx in enumerate(top_indices)
    ]

    return rerank_chunks(question, candidates, top_k=n_results)


# ---------------------------------------------------------------------------
# hybrid_search_with_mode
# ---------------------------------------------------------------------------

def hybrid_search_with_mode(
    question: str,
    document_ids: list[str] = None,
    retrieval_mode: str = "standard",
    candidate_pool: int | None = None,
    top_n: int | None = None,
    doc_type: str = "general",
    user_id: str = "system",
) -> list[dict]:
    mode            = normalise_retrieval_mode(retrieval_mode)
    effective_top_n = top_n if top_n is not None else app_config.retrieval_top_n

    if mode == "hyde":
        passage = generate_hyde_passage(question, doc_type=doc_type, user_id=user_id)
        chunks  = hybrid_search(
            question, document_ids=document_ids,
            candidate_pool=candidate_pool, top_n=effective_top_n,
            dense_query_override=passage, user_id=user_id,
        )
    elif mode == "multiquery":
        queries = generate_query_variants(question, user_id=user_id)
        results_per_query = []
        for q in queries:
            results = hybrid_search(
                q, document_ids=document_ids,
                candidate_pool=candidate_pool, top_n=effective_top_n,
                user_id=user_id,
            )
            results_per_query.append(results)
        chunks = merge_and_dedupe(results_per_query, top_n=effective_top_n)
    else:
        chunks = hybrid_search(
            question, document_ids=document_ids,
            candidate_pool=candidate_pool, top_n=effective_top_n,
            user_id=user_id,
        )

    return expand_to_parent_context(chunks)


# ---------------------------------------------------------------------------
# General answer
# ---------------------------------------------------------------------------

def answer_general(
    question: str,
    history: list[dict] = None,
    history_summary: str = "",
    user_id: str = "system",
) -> str:
    history_text = format_history(history or [], history_summary)
    return call_llm(
        system=GENERAL_SYSTEM,
        user=f"Conversation so far:\n{history_text}\n\nQuestion: {question}",
        temperature=0.5,
        call_type="general",
        user_id=user_id,
    )


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------

def compress_history(messages: list[dict], user_id: str = "system") -> str:
    if not messages:
        return ""
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in messages
    ])
    return call_llm(
        system=(
            "Summarize the following conversation in 3-5 sentences. "
            "Focus on key facts, questions asked, and answers given. "
            "Be concise — this will be used as context for future questions."
        ),
        user=f"Conversation:\n{conversation}\n\nSummary:",
        temperature=0.0,
        max_tokens=300,
        call_type="compress",
        user_id=user_id,
    )


# ---------------------------------------------------------------------------
# Exact evidence extraction
# ---------------------------------------------------------------------------

def get_exact_sentence(
    chunk_content: str,
    question: str,
    user_id: str = "system",
) -> str:
    try:
        if not chunk_content:
            return ""
        sentences = [s.strip() for s in chunk_content.split(".") if len(s.strip()) > 20]
        if not sentences:
            return chunk_content[:200]
        if len(sentences) == 1:
            return sentences[0]
        candidates = sentences[:10]
        result = call_llm(
            system='Return ONLY the single sentence most relevant to the user\'s question. No explanation.',
            user=(
                f'Question: "{question}"\n\n'
                + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(candidates))
            ),
            temperature=0.0,
            max_tokens=150,
            call_type="evidence_extract",
            user_id=user_id,
        )
        extracted = result.strip() if isinstance(result, str) else ""
        return extracted if extracted else sentences[0]
    except Exception:
        return chunk_content[:200] if chunk_content else ""


# ---------------------------------------------------------------------------
# Document query
# ---------------------------------------------------------------------------

def query_document(
    question: str,
    document_id: str = None,
    document_ids: list[str] = None,
    history: list[dict] = None,
    history_summary: str = "",
    provider: str = None,
    model: str = None,
    retrieval_mode: str = "standard",
    user_id: str = "system",
    org_id: str | None = None,
    team_id: str | None = None,
) -> dict:
    has_doc = bool(document_ids or document_id)
    q_type  = classify_question(question, has_document=has_doc, user_id=user_id)

    if q_type == "general":
        answer = answer_general(question, history, history_summary, user_id=user_id)
        return {"answer": sanitize_llm_output(answer), "sources": [], "type": "general"}

    ids      = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    doc_type = _get_doc_type_hint(ids)

    chunks = hybrid_search_with_mode(
        question, document_ids=ids, retrieval_mode=retrieval_mode,
        doc_type=doc_type, user_id=user_id,
    )

    if not chunks:
        answer = answer_general(question, history, history_summary, user_id=user_id)
        return {"answer": sanitize_llm_output(answer), "sources": [], "type": "general"}

    chunks_text  = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)
    system       = QA_MULTI_SYSTEM if is_multi else QA_SYSTEM
    user         = f"Document chunks:\n{chunks_text}\n\nConversation so far:\n{history_text}\n\nQuestion: {question}"

    override_kwargs = {"provider": provider, "model": model} if provider and model else {}
    raw_answer = call_llm(
        system=system,
        user=user,
        temperature=0.2,
        call_type="query",
        user_id=user_id,
        org_id=org_id,
        team_id=team_id,
        **override_kwargs,
    )
    answer = sanitize_llm_output(raw_answer)

    return {
        "answer": answer,
        "sources": [
            {
                "chunk":          c["chunk_num"],
                "page":           c["page"],
                "file":           c["file"],
                "preview":        c["content"][:150],
                "chunk_type":     c.get("chunk_type", "text"),
                "image_ref":      c.get("image_ref"),
                "exact_sentence": get_exact_sentence(c["content"], question, user_id=user_id),
            }
            for c in chunks
        ],
        "type": "document",
    }


# ---------------------------------------------------------------------------
# Streaming query
# ---------------------------------------------------------------------------

def query_document_stream(
    question: str,
    document_id: str = None,
    document_ids: list[str] = None,
    history: list[dict] = None,
    history_summary: str = "",
    provider: str = None,
    model: str = None,
    retrieval_mode: str = "standard",
    user_id: str = "system",
    org_id: str | None = None,
    team_id: str | None = None,
) -> Generator:
    has_doc = bool(document_ids or document_id)
    q_type  = classify_question(question, has_document=has_doc, user_id=user_id)

    if q_type == "general":
        yield sanitize_llm_output(answer_general(question, history, history_summary, user_id=user_id))
        return

    ids      = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    doc_type = _get_doc_type_hint(ids)

    chunks = hybrid_search_with_mode(
        question, document_ids=ids, retrieval_mode=retrieval_mode,
        doc_type=doc_type, user_id=user_id,
    )

    if not chunks:
        yield sanitize_llm_output(answer_general(question, history, history_summary, user_id=user_id))
        return

    chunks_text  = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)
    system       = QA_MULTI_SYSTEM if is_multi else QA_SYSTEM
    user         = f"Document chunks:\n{chunks_text}\n\nConversation so far:\n{history_text}\n\nQuestion: {question}"

    override_kwargs = {"provider": provider, "model": model} if provider and model else {}
    stream_gen = call_llm(
        system=system, user=user,
        call_type="query_stream", stream=True,
        user_id=user_id, org_id=org_id, team_id=team_id,
        **override_kwargs,
    )
    for token in stream_gen:
        yield token


# ---------------------------------------------------------------------------
# Correction feedback loop
# ---------------------------------------------------------------------------

def build_correction_examples(doc_type: str, fields: dict) -> str:
    examples = []
    for field_name in list(fields.keys())[:5]:
        try:
            corrections = get_corrections_for_doc_type(doc_type, field_name, limit=3)
        except Exception:
            continue
        for c in corrections:
            orig = c.get("original_value", "")
            corr = c.get("corrected_value", "")
            if orig and corr and orig != corr:
                examples.append(
                    f"- '{field_name}': was incorrectly extracted as "
                    f"'{orig}', correct value is '{corr}'"
                )
    if not examples:
        return ""
    return "Learn from these past corrections:\n" + "\n".join(examples[:5]) + "\n\n"


# ---------------------------------------------------------------------------
# Bounding box lookup
# ---------------------------------------------------------------------------

def find_field_bbox(field_value: str | None, document_id: str) -> dict | None:
    if not field_value or len(field_value.strip()) < 4:
        return None

    chunks      = get_chunks_by_document(document_id)
    value_lower = field_value.strip().lower()

    matching = [
        c for c in chunks
        if value_lower in c.get("content", "").lower()
        and c.get("metadata", {}).get("bbox") is not None
    ]

    if not matching:
        return None

    def match_position(chunk):
        idx = chunk["content"].lower().find(value_lower)
        return idx if idx >= 0 else 9999

    best = min(matching, key=match_position)
    return best["metadata"]["bbox"]


# ---------------------------------------------------------------------------
# Field extraction — SANDBOXED at document content
# ---------------------------------------------------------------------------

def extract_fields(
    document_id: str,
    fields: dict[str, str],
    doc_type: str = "general",
    user_id: str = "system",
    org_id: str | None = None,
    team_id: str | None = None,
) -> dict:
    if doc_type == "general":
        try:
            from db import get_classification as _get_cls
            cls = _get_cls(document_id)
            if cls:
                detected = cls.get("doc_type", "general")
                if detected and detected != "general":
                    doc_type = detected
        except Exception:
            pass

    all_chunks = get_chunks_by_document(document_id)

    context_chunks = [
        c for c in all_chunks
        if c.get("metadata", {}).get("chunk_level", "flat") in ("parent", "flat")
    ]
    if not context_chunks:
        context_chunks = all_chunks

    context = "\n\n".join([c["content"] for c in context_chunks[:10]])

    fields_with_desc = "\n".join([
        f"- {key}: {value if value and isinstance(value, str) else 'extract from document'}"
        for key, value in fields.items()
    ])

    correction_examples = build_correction_examples(doc_type, fields)
    ExtractionResult    = build_extraction_model(fields)

    # ── SANDBOX: wrap document content before passing to LLM ──
    sandboxed_context = sandbox_and_check(
        context,
        user_id=user_id,
        document_id=document_id,
        label="extract_fields",
    )

    user_content = (
        f"{correction_examples}"
        f"Fields to extract (name: description):\n{fields_with_desc}\n\n"
        f"{sandboxed_context}"
    )

    try:
        result_model = call_llm(
            system=EXTRACTION_SYSTEM,
            user=user_content,
            temperature=0.0,
            call_type="extract",
            response_model=ExtractionResult,
            user_id=user_id,
            document_id=document_id,
            org_id=org_id,
            team_id=team_id,
        )
        raw_extracted = {k: v for k, v in result_model.model_dump().items() if v is not None}
    except Exception as exc:
        return {"extracted": {"error": str(exc)}, "validation": None, "business_validation": {}}

    extracted: dict[str, dict] = {}
    for field_name, value in raw_extracted.items():
        bbox = find_field_bbox(value, document_id)
        extracted[field_name] = {"value": value, "bbox": bbox}

    plain_values = {k: v["value"] for k, v in extracted.items()}
    validation   = validate_extraction(plain_values, fields)

    business_validation = {}
    try:
        engine = ValidationEngine()
        business_validation = engine.validate(plain_values, doc_type)
    except (ImportError, Exception):
        pass

    return {
        "extracted":           extracted,
        "validation":          validation,
        "business_validation": business_validation,
    }


# ---------------------------------------------------------------------------
# Table extraction — SANDBOXED
# ---------------------------------------------------------------------------

def extract_tables(
    document_id: str,
    user_id: str = "system",
    org_id: str | None = None,
    team_id: str | None = None,
) -> list[dict]:
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return []

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    # ── SANDBOX ──
    sandboxed_context = sandbox_and_check(
        context,
        user_id=user_id,
        document_id=document_id,
        label="extract_tables",
    )

    try:
        result: TableList = call_llm(
            system=(
                "Extract ALL tables from the document the user provides. "
                "For each table identify: title (descriptive), headers (column names), "
                "rows (list of string value lists), chart_type ('bar', 'line', or 'pie'). "
                "If no tables are found, return an empty tables list."
            ),
            user=sandboxed_context,
            temperature=0.0,
            call_type="extract_tables",
            response_model=TableList,
            user_id=user_id,
            document_id=document_id,
            org_id=org_id,
            team_id=team_id,
        )
        return [table.model_dump() for table in result.tables]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Summary generation — SANDBOXED
# ---------------------------------------------------------------------------

def generate_summary(
    document_id: str,
    user_id: str = "system",
    org_id: str | None = None,
    team_id: str | None = None,
) -> dict:
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return {"summary": "", "summary_short": ""}

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    # ── SANDBOX ──
    sandboxed_context = sandbox_and_check(
        context,
        user_id=user_id,
        document_id=document_id,
        label="generate_summary",
    )

    try:
        result: DocumentSummary = call_llm(
            system="Analyse the document the user provides and extract a structured summary.",
            user=sandboxed_context,
            temperature=0.0,
            call_type="summarize",
            response_model=DocumentSummary,
            user_id=user_id,
            document_id=document_id,
            org_id=org_id,
            team_id=team_id,
        )
        parsed = result.model_dump()
        return {"summary": json.dumps(parsed), "summary_short": parsed.get("short", "")}
    except Exception:
        return {"summary": "", "summary_short": ""}


# ---------------------------------------------------------------------------
# NL to schema
# ---------------------------------------------------------------------------

def nl_to_schema(instruction: str, user_id: str = "system") -> dict:
    try:
        result: SchemaResult = call_llm(
            system=(
                "Convert the extraction instruction the user provides into a JSON schema. "
                "Return a JSON object where keys are snake_case field names and values are "
                "clear descriptions of what to extract. Include all fields mentioned or "
                "implied. For list fields use values ending in 'as a list'."
            ),
            user=f"Instruction: {instruction}",
            temperature=0.0,
            call_type="nl_to_schema",
            response_model=SchemaResult,
            user_id=user_id,
        )
        schema = result.model_dump()
        if not schema:
            return {"error": "Could not parse instruction into schema"}
        return schema
    except Exception as exc:
        return {"error": "Could not parse instruction into schema", "detail": str(exc)}


def extract_nl(
    document_id: str,
    instruction: str,
    user_id: str = "system",
    org_id: str | None = None,
    team_id: str | None = None,
) -> dict:
    schema = nl_to_schema(instruction, user_id=user_id)
    if "error" in schema:
        return {
            "error": "Could not parse instruction into schema",
            "instruction": instruction,
            "schema": None, "extracted": None,
            "validation": None, "business_validation": {},
        }
    result = extract_fields(document_id, schema, user_id=user_id, org_id=org_id, team_id=team_id)
    return {
        "instruction": instruction, "schema": schema,
        "extracted": result.get("extracted"),
        "validation": result.get("validation"),
        "business_validation": result.get("business_validation"),
    }


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

def _get_confidence_threshold() -> float:
    try:
        return float(app_config.classifier_confidence_threshold)
    except Exception:
        return 0.75


def classify_document(document_id: str, user_id: str = "system") -> dict:
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return {
            "doc_type": "general", "schema_template": "custom",
            "confidence": 0.0, "reasoning": "No document content found.",
            "key_signals": [], "requires_human_review": True, 
            "stage_used": "stage2",
        }
    context = "\n\n".join([c["content"] for c in doc_chunks[:5]])[:2000]
    return classify_document_from_text(context, document_id=document_id, user_id=user_id)


def classify_document_from_text(
    text: str,
    document_id: str = "",
    user_id: str = "system",
) -> dict:
    """
    Two-stage classification pipeline (formerly classification/pipeline.py).

    Stage 1: keyword + embedding scoring (fast, no LLM call).
    Stage 2: LLM classification via _classify_from_context() — only runs
             when Stage 1 confidence falls below the configured threshold,
             or when Stage 1 is disabled via feature flag.

    Output shape is a superset of the original classify_document() shape —
    doc_type, schema_template, confidence, reasoning, key_signals,
    requires_human_review — plus an additive `stage_used` key
    ("stage1" | "stage2").
    """
    if not text or not text.strip():
        return {
            "doc_type": "general", "schema_template": "custom",
            "confidence": 0.0, "reasoning": "No text provided.",
            "key_signals": [], "requires_human_review": True,
            "stage_used": "stage2",
        }

    def _embed(t: str) -> list[float]:
        return get_embed_model().get_text_embedding(t)

    if not app_config.classifier_stage1_enabled:
        # Feature flag off — go straight to LLM (original pre-E1 behaviour)
        result = _classify_from_context(text[:2000], document_id=document_id or "", user_id=user_id)
        result["stage_used"] = "stage2"
        return result

    stage1 = _classify_stage1(text, _embed)
    threshold = _get_confidence_threshold()

    if stage1["confidence"] >= threshold:
        _clf_logger.info(
            "Classification resolved at Stage 1",
            doc_type=stage1["doc_type"], confidence=stage1["confidence"],
            document_id=document_id,
        )
        return _stage1_to_full_result(stage1, threshold)

    _clf_logger.info(
        "Stage 1 confidence below threshold — escalating to LLM",
        stage1_doc_type=stage1["doc_type"], stage1_confidence=stage1["confidence"],
        threshold=threshold, document_id=document_id,
    )

    hint_suffix = ""
    if stage1["doc_type"] and stage1["doc_type"] != "general":
        hint_suffix = (
            f"\n\n(Preliminary keyword analysis suggests this may be a "
            f"'{stage1['doc_type']}' — use your own judgment.)"
        )
    context = text[:2000] + hint_suffix

    result = _classify_from_context(context, document_id=document_id or "", user_id=user_id)
    result["stage_used"] = "stage2"
    return result


# ---------------------------------------------------------------------------
# Stage 1 classifier — keyword + embedding (fast, no LLM call)
# ---------------------------------------------------------------------------
# Folded in from the former classification/stage1.py when the
# classification/ package was merged into retrieval.py. Reuses the existing
# cosine_similarity() defined at the top of this file instead of keeping a
# separate pure-Python implementation.

_exemplar_embeddings: dict[str, list[float]] = {}


def _get_exemplar_embedding(doc_type: str, get_embedding_fn) -> list[float]:
    """
    Return the cached embedding for a doc type's exemplar text (from
    prompts.EMBEDDING_EXEMPLARS). Computed on first access, then cached
    for the process lifetime. Resets automatically on restart.
    """
    if doc_type not in _exemplar_embeddings:
        text = EMBEDDING_EXEMPLARS.get(doc_type, "")
        _exemplar_embeddings[doc_type] = get_embedding_fn(text)
    return _exemplar_embeddings[doc_type]


def _clear_exemplar_cache() -> None:
    """Clear the Stage 1 exemplar embedding cache. Call if the embedding model changes at runtime."""
    _exemplar_embeddings.clear()


def _keyword_score(text_sample: str) -> dict[str, float]:
    """
    Score each doc type by how many keyword signals appear in the first
    500 characters of document text (case-insensitive). Normalised by the
    number of signals defined for that type.
    """
    text_lower = text_sample[:500].lower()
    scores: dict[str, float] = {}
    for doc_type, signals in KEYWORD_SIGNALS.items():
        if not signals:
            scores[doc_type] = 0.0
            continue
        hits = sum(1 for s in signals if s in text_lower)
        scores[doc_type] = hits / len(signals)
    return scores


def _embedding_score(text_sample: str, get_embedding_fn) -> dict[str, float]:
    """
    Embed the first 300 chars of document text and compute cosine similarity
    (via the module-level cosine_similarity()) against each doc type's
    exemplar embedding.
    """
    doc_vec = get_embedding_fn(text_sample[:300])
    scores: dict[str, float] = {}
    for doc_type in EMBEDDING_EXEMPLARS:
        exemplar_vec = _get_exemplar_embedding(doc_type, get_embedding_fn)
        scores[doc_type] = float(cosine_similarity(doc_vec, exemplar_vec))
    return scores


def _classify_stage1(
    full_text: str,
    get_embedding_fn,
    keyword_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> dict:
    """
    Stage 1 classification: combine keyword + embedding scores.

    Confidence is spread-based: (best_score - second_score) / 0.3, capped
    at 1.0. Two nearly-equal top candidates -> low confidence even if both
    have high absolute scores — this correctly escalates ambiguous docs
    to Stage 2.
    """
    if not full_text or not full_text.strip():
        return {"doc_type": "general", "confidence": 0.0, "stage": "stage1",
                "keyword_scores": {}, "embedding_scores": {}}

    kw_scores  = _keyword_score(full_text)
    emb_scores = _embedding_score(full_text, get_embedding_fn)

    all_types = set(kw_scores) | set(emb_scores)
    combined: dict[str, float] = {}
    for doc_type in all_types:
        kw  = kw_scores.get(doc_type, 0.0)
        emb = emb_scores.get(doc_type, 0.0)
        combined[doc_type] = keyword_weight * kw + embedding_weight * emb

    if not combined:
        return {"doc_type": "general", "confidence": 0.0, "stage": "stage1",
                "keyword_scores": kw_scores, "embedding_scores": emb_scores}

    sorted_types = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_types[0]
    second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0.0

    spread     = best_score - second_score
    confidence = min(spread / 0.3, 1.0) if best_score > 0 else 0.0
    confidence = round(confidence, 3)
    resolved_type = best_type if confidence > 0 else "general"

    _clf_logger.info(
        "Stage 1 classification", doc_type=resolved_type, confidence=confidence,
        best_score=round(best_score, 3), spread=round(spread, 3),
    )

    return {
        "doc_type": resolved_type, "confidence": confidence, "stage": "stage1",
        "keyword_scores": kw_scores, "embedding_scores": emb_scores,
    }


def _stage1_to_full_result(stage1: dict, threshold: float) -> dict:
    """Convert a Stage 1 result dict into the full classification shape callers expect."""
    doc_type   = stage1["doc_type"]
    confidence = stage1["confidence"]
    return {
        "doc_type":              doc_type,
        "schema_template":       get_template_for_doc_type(doc_type),
        "confidence":            confidence,
        "reasoning":             f"Keyword/embedding match (confidence {confidence})",
        "key_signals":           [],
        "requires_human_review": confidence < threshold,
        "stage_used":            "stage1",
    }


def _classify_from_context(
    context: str,
    document_id: str = "",
    user_id: str = "system",
) -> dict:
    """
    LLM-only classification
    SANDBOXED: document content is wrapped before passing to LLM.
    Uses DOCUMENT_CLASSIFIER_SYSTEM from prompts.py (not a local constant).
    """
    # ── SANDBOX ──
    sandboxed_context = sandbox_and_check(
        context,
        user_id=user_id,
        document_id=document_id if document_id else None,
        label="classify_document",
    )

    try:
        result: DocumentClassification = call_llm(
            system=DOCUMENT_CLASSIFIER_SYSTEM,
            user=sandboxed_context,
            temperature=0.0,
            call_type="classify_document",
            response_model=DocumentClassification,
            user_id=user_id,
            document_id=document_id if document_id else None,
        )
    except Exception as exc:
        _clf_logger.error("LLM classification failed", document_id=document_id, error=str(exc))
        return {
            "doc_type": "general", "schema_template": "custom",
            "confidence": 0.0, "reasoning": f"Classification error: {exc}",
            "key_signals": [], "requires_human_review": True,
            "stage_used": "stage2",
        }

    doc_type        = result.doc_type.lower().strip()
    confidence      = float(result.confidence)
    schema_template = get_template_for_doc_type(doc_type)
    threshold       = _get_confidence_threshold()

    _clf_logger.info(
        "Document classified (LLM)",
        document_id=document_id, doc_type=doc_type,
        confidence=round(confidence, 3),
        requires_review=confidence < threshold,
    )

    return {
        "doc_type":              doc_type,
        "schema_template":       schema_template,
        "confidence":            confidence,
        "reasoning":             result.reasoning,
        "key_signals":           result.key_signals,
        "requires_human_review": confidence < threshold,
        "stage_used":            "stage2",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_doc_type_hint(document_ids: list[str] | None) -> str:
    if not document_ids:
        return "general"
    try:
        from db import get_classification as _get_cls
        cls = _get_cls(document_ids[0])
        if cls:
            return cls.get("doc_type", "general") or "general"
    except Exception:
        pass
    return "general"