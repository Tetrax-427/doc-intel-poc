# backend/retrieval.py

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
)
from db import get_corrections_for_doc_type
from llm.engine import call_llm, call_llm_stream
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
# Source formatting (B3)
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
    """Returns 'document' or 'general'."""
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
# D3 — Hybrid search
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
            logger.debug("Loaded chunks for doc", doc_id=doc_id, count=len(chunks))
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

    texts = [c["content"] for c in searchable]

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
            question,
            document_ids=document_ids,
            candidate_pool=candidate_pool,
            top_n=effective_top_n,
            dense_query_override=passage,
            user_id=user_id,
        )

    elif mode == "multiquery":
        queries = generate_query_variants(question, user_id=user_id)
        results_per_query = []
        for q in queries:
            results = hybrid_search(
                q,
                document_ids=document_ids,
                candidate_pool=candidate_pool,
                top_n=effective_top_n,
                user_id=user_id,
            )
            results_per_query.append(results)
        chunks = merge_and_dedupe(results_per_query, top_n=effective_top_n)

    else:
        chunks = hybrid_search(
            question,
            document_ids=document_ids,
            candidate_pool=candidate_pool,
            top_n=effective_top_n,
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

def compress_history(
    messages: list[dict],
    user_id: str = "system",
) -> str:
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
) -> dict:
    has_doc = bool(document_ids or document_id)
    q_type  = classify_question(question, has_document=has_doc, user_id=user_id)

    if q_type == "general":
        answer = answer_general(question, history, history_summary, user_id=user_id)
        return {"answer": answer, "sources": [], "type": "general"}

    ids      = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    doc_type = _get_doc_type_hint(ids)

    chunks = hybrid_search_with_mode(
        question,
        document_ids=ids,
        retrieval_mode=retrieval_mode,
        doc_type=doc_type,
        user_id=user_id,
    )

    if not chunks:
        answer = answer_general(question, history, history_summary, user_id=user_id)
        return {"answer": answer, "sources": [], "type": "general"}

    chunks_text  = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)

    system = QA_MULTI_SYSTEM if is_multi else QA_SYSTEM
    user   = f"Document chunks:\n{chunks_text}\n\nConversation so far:\n{history_text}\n\nQuestion: {question}"

    override_kwargs = {"provider": provider, "model": model} if provider and model else {}
    answer = call_llm(
        system=system,
        user=user,
        temperature=0.2,
        call_type="query",
        user_id=user_id,
        **override_kwargs,
    )

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
) -> Generator:
    has_doc = bool(document_ids or document_id)
    q_type  = classify_question(question, has_document=has_doc, user_id=user_id)

    if q_type == "general":
        yield answer_general(question, history, history_summary, user_id=user_id)
        return

    ids      = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    doc_type = _get_doc_type_hint(ids)

    chunks = hybrid_search_with_mode(
        question,
        document_ids=ids,
        retrieval_mode=retrieval_mode,
        doc_type=doc_type,
        user_id=user_id,
    )

    if not chunks:
        yield answer_general(question, history, history_summary, user_id=user_id)
        return

    chunks_text  = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)

    system = QA_MULTI_SYSTEM if is_multi else QA_SYSTEM
    user   = f"Document chunks:\n{chunks_text}\n\nConversation so far:\n{history_text}\n\nQuestion: {question}"

    override_kwargs = {"provider": provider, "model": model} if provider and model else {}
    stream_gen = call_llm(
        system=system,
        user=user,
        call_type="query_stream",
        stream=True,
        user_id=user_id,
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
# E2 — Bounding box lookup for extracted field values
# ---------------------------------------------------------------------------

def find_field_bbox(field_value: str | None, document_id: str) -> dict | None:
    if not field_value or len(field_value.strip()) < 4:
        return None

    chunks = get_chunks_by_document(document_id)
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
# Field extraction — E2 enriched with bbox
# ---------------------------------------------------------------------------

def extract_fields(
    document_id: str,
    fields: dict[str, str],
    doc_type: str = "general",
    user_id: str = "system",
) -> dict:
    """
    Extract fields from a document using Instructor structured output.
    Returns {extracted: {field: {value, bbox}}, validation, business_validation}.
    """
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

    user_content = (
        f"{correction_examples}"
        f"Fields to extract (name: description):\n{fields_with_desc}\n\n"
        f"Document:\n{context}"
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
        )
        raw_extracted = {k: v for k, v in result_model.model_dump().items() if v is not None}
    except Exception as exc:
        return {"extracted": {"error": str(exc)}, "validation": None, "business_validation": {}}

    # E2 — attach bbox per field
    extracted: dict[str, dict] = {}
    for field_name, value in raw_extracted.items():
        bbox = find_field_bbox(value, document_id)
        extracted[field_name] = {"value": value, "bbox": bbox}

    plain_values = {k: v["value"] for k, v in extracted.items()}

    validation = validate_extraction(plain_values, fields)

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
# Table extraction
# ---------------------------------------------------------------------------

def extract_tables(document_id: str, user_id: str = "system") -> list[dict]:
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return []

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    try:
        result: TableList = call_llm(
            system=(
                "Extract ALL tables from the document the user provides. "
                "For each table identify: title (descriptive), headers (column names), "
                "rows (list of string value lists), chart_type ('bar', 'line', or 'pie'). "
                "If no tables are found, return an empty tables list."
            ),
            user=f"Document:\n{context}",
            temperature=0.0,
            call_type="extract_tables",
            response_model=TableList,
            user_id=user_id,
            document_id=document_id,
        )
        return [table.model_dump() for table in result.tables]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(document_id: str, user_id: str = "system") -> dict:
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return {"summary": "", "summary_short": ""}

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    try:
        result: DocumentSummary = call_llm(
            system="Analyse the document the user provides and extract a structured summary.",
            user=f"Document:\n{context}",
            temperature=0.0,
            call_type="summarize",
            response_model=DocumentSummary,
            user_id=user_id,
            document_id=document_id,
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
) -> dict:
    schema = nl_to_schema(instruction, user_id=user_id)
    if "error" in schema:
        return {
            "error": "Could not parse instruction into schema",
            "instruction": instruction,
            "schema": None, "extracted": None,
            "validation": None, "business_validation": {},
        }
    result = extract_fields(document_id, schema, user_id=user_id)
    return {
        "instruction": instruction, "schema": schema,
        "extracted": result.get("extracted"),
        "validation": result.get("validation"),
        "business_validation": result.get("business_validation"),
    }


# ---------------------------------------------------------------------------
# Document classification — E1 two-stage pipeline
# ---------------------------------------------------------------------------

DOCUMENT_CLASSIFIER_SYSTEM = (
    "You are a document classification expert. "
    "Analyse the document text and classify it. "
    "Valid doc_type values: invoice, receipt, resume, cv, contract, agreement, nda, report, "
    "research paper, financial statement, balance sheet, income statement, medical record, "
    "prescription, legal document, court filing, article, email, letter, general, "
    "gst return, gstr-1, gstr-3b, offer letter, loan application, bank statement"
)

TEMPLATE_MAP = {
    "invoice":              "invoice",
    "receipt":              "invoice",
    "resume":               "cv_resume",
    "cv":                   "cv_resume",
    "curriculum vitae":     "cv_resume",
    "contract":             "contract",
    "agreement":            "contract",
    "nda":                  "contract",
    "report":               "report",
    "research paper":       "report",
    "financial statement":  "financial",
    "balance sheet":        "financial",
    "income statement":     "financial",
    "medical record":       "medical",
    "prescription":         "medical",
    "legal document":       "legal",
    "court filing":         "legal",
    "article":              "general",
    "email":                "general",
    "letter":               "general",
    "general":              "general",
}


def _get_confidence_threshold() -> float:
    try:
        return float(app_config.classification_confidence_threshold)
    except Exception:
        return 0.75


def classify_document(document_id: str, user_id: str = "system") -> dict:
    """
    Classify a document using chunks already stored in the DB.
    E1: routes through the two-stage pipeline.
    """
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return {
            "doc_type": "general", "schema_template": "custom",
            "confidence": 0.0, "reasoning": "No document content found.",
            "key_signals": [], "requires_human_review": True,
        }

    context = "\n\n".join([c["content"] for c in doc_chunks[:5]])[:2000]
    return classify_document_from_text(context, document_id=document_id, user_id=user_id)


def classify_document_from_text(
    text: str,
    document_id: str = "",
    user_id: str = "system",
) -> dict:
    """
    E1: tries Stage 1 (keyword + embedding) first; falls back to LLM only
    when Stage 1 confidence is below threshold.
    """
    if not text or not text.strip():
        return {
            "doc_type": "general", "schema_template": "custom",
            "confidence": 0.0, "reasoning": "No text provided.",
            "key_signals": [], "requires_human_review": True,
            "stage_used": "stage2",
        }

    from classification.pipeline import classify as _classify_pipeline

    def _embed(t: str) -> list[float]:
        return get_embed_model().get_text_embedding(t)

    return _classify_pipeline(
        full_text=text,
        get_embedding_fn=_embed,
        document_id=document_id or None,
        user_id=user_id,
    )


def _classify_from_context(
    context: str,
    document_id: str = "",
    user_id: str = "system",
) -> dict:
    """
    LLM-only classification — called by pipeline.py Stage 2.
    No longer uses the old in-memory classification cache (core/cache.py).
    The new unified llm_cache handles caching automatically via call_llm().
    """
    try:
        result: DocumentClassification = call_llm(
            system=DOCUMENT_CLASSIFIER_SYSTEM,
            user=f"Document (first ~2000 characters):\n{context}",
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