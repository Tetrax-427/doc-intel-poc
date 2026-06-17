# backend/retrieval.py

import os
import json
import numpy as np
from typing import Generator
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from db import get_chunks_by_document
from ingestion import get_embed_model
from prompts import (
    QA_PROMPT, QA_PROMPT_MULTI, GENERAL_PROMPT,
    CLASSIFIER_PROMPT, QUERY_EXPANSION_PROMPT, EXTRACTION_PROMPT
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
from schemas.validator import validate_extraction
from validation.engine import ValidationEngine

load_dotenv()

from core.logger import get_logger as _get_logger
from schemas.templates import get_template_for_doc_type
from core.config import config as app_config
from core.cache import get_classification, set_classification, make_text_hash
from db import get_chunks_by_document

import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))
_clf_logger = _get_logger("retrieval.classify")

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
    """
    Build a standardised source entry from a retrieval chunk.

    Handles four chunk_type values:
      "text"        — standard text source
      "table"       — table source
      "description" — vision description (image-level, no specific figure ref)
      "figure"      — B3: figure/chart with caption + bbox in metadata

    The returned dict is the contract between retrieval and the frontend.
    Fields present on ALL types:
      chunk, page, file, preview, chunk_type, exact_sentence

    Additional fields for "figure":
      image_ref, caption, bbox

    Additional fields for "description":
      image_ref

    Never raises — falls back to a basic source dict on any error.
    """
    try:
        chunk_type = chunk.get("chunk_type", "text")
        base = {
            "chunk":          chunk.get("chunk_num"),
            "page":           chunk.get("page"),
            "file":           chunk.get("file"),
            "preview":        chunk.get("content", "")[:150],
            "chunk_type":     chunk_type,
            "exact_sentence": "",   # filled by caller after get_exact_sentence()
        }

        if chunk_type == "figure":
            base["image_ref"] = chunk.get("image_ref")
            base["caption"]   = chunk.get("caption", "")
            base["bbox"]      = chunk.get("bbox")          # B1 — normalized coords

        elif chunk_type in ("description", "table"):
            base["image_ref"] = chunk.get("image_ref")

        return base

    except Exception:
        # Safe fallback — never let source formatting break a query response
        return {
            "chunk":          chunk.get("chunk_num"),
            "page":           chunk.get("page"),
            "file":           chunk.get("file"),
            "preview":        chunk.get("content", "")[:150],
            "chunk_type":     chunk.get("chunk_type", "text"),
            "exact_sentence": "",
        }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_question(question: str, has_document: bool = True) -> str:
    """Returns 'document' or 'general'"""
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
            CLASSIFIER_PROMPT.format(question=question),
            temperature=0.0,
            max_tokens=10,
            call_type="classify"
        )
        return "document" if "document" in result.strip().lower() else "general"
    except Exception:
        return "document"


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def expand_query(question: str) -> str:
    """
    Rewrite vague questions to improve retrieval.
    Uses Instructor structured output to guarantee a clean string response.
    Falls back to original question on any error.
    """
    try:
        result: QueryExpansion = call_llm(
            QUERY_EXPANSION_PROMPT.format(question=question),
            temperature=0.0,
            max_tokens=100,
            call_type="expand",
            response_model=QueryExpansion,
        )
        expanded = result.expanded_query.strip()
        return expanded if expanded else question
    except Exception:
        return question


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank_chunks(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Use Cohere to rerank retrieved chunks."""
    if not chunks or not os.getenv("COHERE_API_KEY"):
        return chunks[:top_k]
    try:
        results = co.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=[c["content"] for c in chunks],
            top_n=top_k
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
        print(f"Reranking failed: {e} — using original order")
        return chunks[:top_k]


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(question, document_ids=None, top_k=10):
    embed_model = get_embed_model()
    if document_ids:
        all_chunks = []
        for doc_id in document_ids:
            chunks = get_chunks_by_document(doc_id)
            print(f"[DEBUG] doc_id={doc_id} chunks={len(chunks)}")
            all_chunks.extend(chunks)
    else:
        all_chunks = []

    if not all_chunks:
        return []

    expanded_question = expand_query(question)
    print(f"Expanded query: {expanded_question}")

    question_embedding = embed_model.get_text_embedding(expanded_question)
    texts = [c["content"] for c in all_chunks]

    # Dense search
    dense_scores = []
    for chunk in all_chunks:
        raw_emb = chunk["embedding"]
        if isinstance(raw_emb, str):
            raw_emb = json.loads(raw_emb)
        dense_scores.append(cosine_similarity(question_embedding, raw_emb))

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(expanded_question.lower().split())

    # RRF fusion
    def rrf_score(rank, k=60):
        return 1 / (k + rank + 1)

    dense_ranked = np.argsort(dense_scores)[::-1]
    bm25_ranked  = np.argsort(bm25_scores)[::-1]

    rrf = {}
    for rank, idx in enumerate(dense_ranked):
        rrf[idx] = rrf.get(idx, 0) + rrf_score(rank)
    for rank, idx in enumerate(bm25_ranked):
        rrf[idx] = rrf.get(idx, 0) + rrf_score(rank)

    top_indices = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    candidates = [
        {
            "chunk_num":    i + 1,
            "content":      all_chunks[idx]["content"],
            "page":         all_chunks[idx]["metadata"].get("page", "?"),
            "file":         all_chunks[idx]["metadata"].get("file", "?"),
            "chunk_type":   all_chunks[idx]["metadata"].get("chunk_type", "text"),
            "image_ref":    all_chunks[idx]["metadata"].get("image_ref"),
            "caption":      all_chunks[idx]["metadata"].get("caption", ""),   # B3
            "bbox":         all_chunks[idx]["metadata"].get("bbox"),           # B1
            "score":        rrf[idx],
        }
        for i, idx in enumerate(top_indices)
    ]

    return rerank_chunks(question, candidates, top_k=5)


# ---------------------------------------------------------------------------
# General answer
# ---------------------------------------------------------------------------

def answer_general(
    question: str,
    history: list[dict] = None,
    history_summary: str = ""
) -> str:
    history_text = format_history(history or [], history_summary)
    return call_llm(
        GENERAL_PROMPT.format(history=history_text, question=question),
        temperature=0.5,
        call_type="general"
    )


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------

def compress_history(messages: list[dict]) -> str:
    if not messages:
        return ""
    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in messages
    ])
    return call_llm(
        f"""Summarize the following conversation in 3-5 sentences.
Focus on key facts, questions asked, and answers given.
Be concise — this will be used as context for future questions.

Conversation:
{conversation}

Summary:""",
        temperature=0.0,
        max_tokens=300,
        call_type="compress"
    )


# ---------------------------------------------------------------------------
# Exact evidence extraction
# ---------------------------------------------------------------------------

def get_exact_sentence(chunk_content: str, question: str) -> str:
    """
    Find the single most relevant sentence from a chunk for a given question.
    Never raises.
    """
    try:
        if not chunk_content:
            return ""

        sentences = [
            s.strip()
            for s in chunk_content.split(".")
            if len(s.strip()) > 20
        ]

        if not sentences:
            return chunk_content[:200]

        if len(sentences) == 1:
            return sentences[0]

        candidates = sentences[:10]

        result = call_llm(
            f"""From the sentences below, return ONLY the single sentence most relevant to: "{question}"

{chr(10).join(f"{i + 1}. {s}" for i, s in enumerate(candidates))}

Return ONLY the sentence text, nothing else. No numbering, no explanation.""",
            temperature=0.0,
            max_tokens=150,
            call_type="evidence_extract"
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
) -> dict:
    has_doc = bool(document_ids or document_id)
    q_type  = classify_question(question, has_document=has_doc)

    if q_type == "general":
        answer = answer_general(question, history, history_summary)
        return {"answer": answer, "sources": [], "type": "general"}

    ids      = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    chunks   = hybrid_search(question, document_ids=ids)

    if not chunks:
        answer = answer_general(question, history, history_summary)
        return {"answer": answer, "sources": [], "type": "general"}

    chunks_text  = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)
    prompt = (QA_PROMPT_MULTI if is_multi else QA_PROMPT).format(
        chunks=chunks_text,
        history=history_text,
        question=question
    )

    # pass override through only when both are supplied
    override_kwargs = {"provider": provider, "model": model} if provider and model else {}
    answer = call_llm(prompt, temperature=0.2, call_type="query", **override_kwargs)

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
                "exact_sentence": get_exact_sentence(c["content"], question),
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
) -> Generator:
    has_doc = bool(document_ids or document_id)
    q_type  = classify_question(question, has_document=has_doc)

    if q_type == "general":
        yield answer_general(question, history, history_summary)
        return

    ids      = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    chunks   = hybrid_search(question, document_ids=ids)

    if not chunks:
        yield answer_general(question, history, history_summary)
        return

    chunks_text  = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)
    prompt = (QA_PROMPT_MULTI if is_multi else QA_PROMPT).format(
        chunks=chunks_text,
        history=history_text,
        question=question
    )

    # forward override to stream path when supplied
    override_kwargs = {"provider": provider, "model": model} if provider and model else {}
    stream_gen = call_llm(prompt, call_type="query_stream", stream=True, **override_kwargs)
    for token in stream_gen:
        yield token


# ---------------------------------------------------------------------------
# Correction feedback loop
# ---------------------------------------------------------------------------

def build_correction_examples(doc_type: str, fields: dict) -> str:
    """
    Build few-shot correction examples from past human review decisions.
    Returns empty string when no corrections exist.
    Never raises.
    """
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

    lines = examples[:5]
    return "Learn from these past corrections:\n" + "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def extract_fields(
    document_id: str,
    fields: dict[str, str],
    doc_type: str = "general",
) -> dict:
    """
    Extract fields from a document using Instructor structured output.

    Pipeline:
    1. Auto-detect doc_type from stored classification if not passed explicitly
    2. Build few-shot correction examples from past human review decisions
    3. Build a dynamic Pydantic model from the fields dict
    4. Run the extraction prompt through Instructor (response_model=ExtractionResult)
    5. Run confidence scoring (Phase 1 behaviour)
    6. Run business logic validation (Phase 2 — skipped silently if not available)

    Return shape is identical to the old implementation:
        {
            "extracted":           dict of extracted values,
            "validation":          confidence scoring result,
            "business_validation": Contract 4 shaped result, or {},
        }
    """
    # 1. Auto-detect doc_type
    if doc_type == "general":
        try:
            from db import get_classification
            cls = get_classification(document_id)
            if cls:
                detected = cls.get("doc_type", "general")
                if detected and detected != "general":
                    doc_type = detected
        except Exception:
            pass

    doc_chunks = get_chunks_by_document(document_id)
    context    = "\n\n".join([c["content"] for c in doc_chunks[:10]])

    fields_with_desc = "\n".join([
        f"- {key}: {value if value and isinstance(value, str) else 'extract from document'}"
        for key, value in fields.items()
    ])

    # 2. Build correction examples
    correction_examples = build_correction_examples(doc_type, fields)

    # 3. Build dynamic Pydantic model from fields dict
    ExtractionResult = build_extraction_model(fields)

    # 4. Prompt + Instructor call
    prompt = EXTRACTION_PROMPT.format(
        correction_examples=correction_examples,
        fields_with_descriptions=fields_with_desc,
        context=context,
    )

    try:
        result_model = call_llm(
            prompt,
            temperature=0.0,
            call_type="extract",
            response_model=ExtractionResult,
        )
        # Convert to plain dict, dropping None values for missing fields
        extracted = {
            k: v for k, v in result_model.model_dump().items()
            if v is not None
        }
    except Exception as exc:
        return {
            "extracted":           {"error": str(exc)},
            "validation":          None,
            "business_validation": {},
        }

    # 5. Confidence scoring (Phase 1 — unchanged)
    validation = validate_extraction(extracted, fields)

    # 6. Business logic validation — guarded import
    business_validation = {}
    try:
        engine = ValidationEngine()
        business_validation = engine.validate(extracted, doc_type)
    except ImportError:
        pass
    except Exception:
        pass

    return {
        "extracted":           extracted,
        "validation":          validation,
        "business_validation": business_validation,
    }


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------

def extract_tables(document_id: str) -> list[dict]:
    """
    Extract all tables from a document using Instructor structured output.
    Returns a list of dicts (same shape as before) for caller compatibility.
    """
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return []

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    try:
        result: TableList = call_llm(
            f"""Extract ALL tables from the document below.
For each table identify:
- title: descriptive title for the table
- headers: list of column names
- rows: list of rows, each row is a list of string values
- chart_type: suggest "bar", "line", or "pie" based on the data

If no tables are found, return an empty tables list.

Document:
{context}""",
            temperature=0.0,
            call_type="extract_tables",
            response_model=TableList,
        )
        # Convert to list of plain dicts — preserves existing return shape
        return [table.model_dump() for table in result.tables]

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(document_id: str) -> dict:
    """
    Generate a structured summary using Instructor.
    Return shape is identical to the old implementation:
        {"summary": "<json string>", "summary_short": "<one sentence>"}
    """
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        return {"summary": "", "summary_short": ""}

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    try:
        result: DocumentSummary = call_llm(
            f"""Analyse the document below and extract a structured summary.
             "short": one sentence (max 20 words) describing what this document is
- "overview": 2-3 sentence overview of the document
- "key_topics": list of 3-5 main topics covered
- "entities": list of important names, companies, or organizations mentioned
- "dates": list of important dates mentioned (empty list if none)
- "amounts": list of important numbers, amounts, or figures mentioned (empty list if none)
- "document_type": type of document (e.g. "Resume", "Invoice", "Contract", "Report", "Article")


Document:
{context}""",
            temperature=0.0,
            call_type="summarize",
            response_model=DocumentSummary,
        )
        parsed = result.model_dump()
        return {
            "summary":       json.dumps(parsed),
            "summary_short": parsed.get("short", ""),
        }
    except Exception:
        return {"summary": "", "summary_short": ""}


# ---------------------------------------------------------------------------
# NL to schema
# ---------------------------------------------------------------------------

def nl_to_schema(instruction: str) -> dict:
    """
    Convert plain English instruction to extraction schema using Instructor.
    Uses SchemaResult (extra="allow") to handle dynamic field names.
    Falls back to error dict on failure (preserves old behaviour).
    """
    try:
        result: SchemaResult = call_llm(
            f"""Convert the following extraction instruction into a JSON schema for document extraction.

Rules:
- Return a JSON object where keys are snake_case field names
- Values are clear descriptions of what to extract for that field
- Include all fields mentioned or implied by the instruction
- For list fields (skills, items, etc.) use descriptive values ending in "as a list"
- Be specific about which entity the field belongs to

Instruction: {instruction}""",
            temperature=0.0,
            call_type="nl_to_schema",
            response_model=SchemaResult,
        )
        # model_dump() on extra="allow" returns all dynamic fields
        schema = result.model_dump()
        if not schema:
            return {"error": "Could not parse instruction into schema"}
        return schema

    except Exception as exc:
        return {"error": "Could not parse instruction into schema", "detail": str(exc)}


def extract_nl(document_id: str, instruction: str) -> dict:
    """
    Natural language extraction pipeline:
    1. Convert instruction to schema
    2. Run extraction with feedback loop + validation
    3. Return schema + extracted + validation + business_validation
    """
    schema = nl_to_schema(instruction)

    if "error" in schema:
        return {
            "error":               "Could not parse instruction into schema",
            "instruction":         instruction,
            "schema":              None,
            "extracted":           None,
            "validation":          None,
            "business_validation": {},
        }

    result = extract_fields(document_id, schema)

    return {
        "instruction":         instruction,
        "schema":              schema,
        "extracted":           result.get("extracted"),
        "validation":          result.get("validation"),
        "business_validation": result.get("business_validation"),
    }


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------

DOCUMENT_CLASSIFIER_PROMPT = """You are a document classification expert.

Analyse the document text below and classify it.

Valid doc_type values: invoice, receipt, resume, cv, contract, agreement, nda, report,
research paper, financial statement, balance sheet, income statement, medical record,
prescription, legal document, court filing, article, email, letter, general,
gst return, gstr-1, gstr-3b, offer letter, loan application, bank statement

Document (first ~2000 characters):
{context}"""

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


def classify_document(document_id: str) -> dict:
    """
    Classify a document into a known type using Instructor structured output.
    Checks classification cache first — LLM only called on cache miss.
    Never raises — always returns a safe default on failure.
    Return shape is identical to the old implementation.
    """
    doc_chunks = get_chunks_by_document(document_id)
    if not doc_chunks:
        _clf_logger.warning("No chunks found for classification", document_id=document_id)
        return {
            "doc_type":              "general",
            "schema_template":       "custom",
            "confidence":            0.0,
            "reasoning":             "No document content found.",
            "key_signals":           [],
            "requires_human_review": True,
        }

    context = "\n\n".join([c["content"] for c in doc_chunks[:5]])[:2000]

    # Cache check
    try:
        text_hash = make_text_hash(context)
        cached = get_classification(text_hash)
        if cached is not None:
            _clf_logger.info("Classification cache hit", document_id=document_id)
            return cached
    except Exception:
        text_hash = None

    # LLM call via Instructor
    try:
        result: DocumentClassification = call_llm(
            DOCUMENT_CLASSIFIER_PROMPT.format(context=context),
            temperature=0.0,
            call_type="classify_document",
            response_model=DocumentClassification,
        )
    except Exception as exc:
        _clf_logger.error("LLM call failed during classification",
                          document_id=document_id, error=str(exc))
        return {
            "doc_type":              "general",
            "schema_template":       "custom",
            "confidence":            0.0,
            "reasoning":             f"Classification error: {exc}",
            "key_signals":           [],
            "requires_human_review": True,
        }

    doc_type        = result.doc_type.lower().strip()
    confidence      = float(result.confidence)
    schema_template = get_template_for_doc_type(doc_type)
    threshold       = _get_confidence_threshold()

    _clf_logger.info(
        "Document classified",
        document_id=document_id,
        doc_type=doc_type,
        confidence=round(confidence, 3),
        schema_template=schema_template,
        requires_review=confidence < threshold,
    )

    classification_result = {
        "doc_type":              doc_type,
        "schema_template":       schema_template,
        "confidence":            confidence,
        "reasoning":             result.reasoning,
        "key_signals":           result.key_signals,
        "requires_human_review": confidence < threshold,
    }

    # Cache write
    try:
        if text_hash:
            set_classification(text_hash, classification_result)
    except Exception:
        pass

    return classification_result