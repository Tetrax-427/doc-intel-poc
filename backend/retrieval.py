import os
import json
import numpy as np
from typing import Generator
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from db import get_all_chunks
from ingestion import get_embed_model
from prompts import (
    QA_PROMPT, QA_PROMPT_MULTI, GENERAL_PROMPT,
    CLASSIFIER_PROMPT, QUERY_EXPANSION_PROMPT, EXTRACTION_PROMPT
)
from llm.engine import call_llm, call_llm_stream

load_dotenv()

import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))


# --- Helpers ---

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


# --- Classification ---

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


# --- Query expansion ---

def expand_query(question: str) -> str:
    """Rewrite vague questions to improve retrieval"""
    try:
        expanded = call_llm(
            QUERY_EXPANSION_PROMPT.format(question=question),
            temperature=0.0,
            max_tokens=100,
            call_type="expand"
        )
        return expanded.strip() if expanded.strip() else question
    except Exception:
        return question


# --- Reranking ---

def rerank_chunks(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Use Cohere to rerank retrieved chunks"""
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


# --- Hybrid search ---

def hybrid_search(question: str, document_ids: list[str] = None, top_k: int = 10) -> list[dict]:
    embed_model = get_embed_model()
    all_chunks = get_all_chunks()

    if document_ids:
        all_chunks = [c for c in all_chunks if c["document_id"] in document_ids]

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

    # RRF
    def rrf_score(rank, k=60):
        return 1 / (k + rank + 1)

    dense_ranked = np.argsort(dense_scores)[::-1]
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    rrf = {}
    for rank, idx in enumerate(dense_ranked):
        rrf[idx] = rrf.get(idx, 0) + rrf_score(rank)
    for rank, idx in enumerate(bm25_ranked):
        rrf[idx] = rrf.get(idx, 0) + rrf_score(rank)

    top_indices = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    candidates = [
        {
            "chunk_num": i + 1,
            "content": all_chunks[idx]["content"],
            "page": all_chunks[idx]["metadata"].get("page", "?"),
            "file": all_chunks[idx]["metadata"].get("file", "?"),
            "chunk_type": all_chunks[idx]["metadata"].get("chunk_type", "text"),
            "image_ref": all_chunks[idx]["metadata"].get("image_ref"),
            "score": rrf[idx]
        }
        for i, idx in enumerate(top_indices)
    ]

    return rerank_chunks(question, candidates, top_k=5)


# --- General answer ---

def answer_general(question: str, history: list[dict] = None, history_summary: str = "") -> str:
    history_text = format_history(history or [], history_summary)
    return call_llm(
        GENERAL_PROMPT.format(history=history_text, question=question),
        temperature=0.5,
        call_type="general"
    )


# --- History compression ---

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


# --- Document query ---

def query_document(
    question: str,
    document_id: str = None,
    document_ids: list[str] = None,
    history: list[dict] = None,
    history_summary: str = ""
) -> dict:
    has_doc = bool(document_ids or document_id)
    q_type = classify_question(question, has_document=has_doc)

    if q_type == "general":
        answer = answer_general(question, history, history_summary)
        return {"answer": answer, "sources": [], "type": "general"}

    ids = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    chunks = hybrid_search(question, document_ids=ids)

    if not chunks:
        answer = answer_general(question, history, history_summary)
        return {"answer": answer, "sources": [], "type": "general"}

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)
    prompt = (QA_PROMPT_MULTI if is_multi else QA_PROMPT).format(
        chunks=chunks_text,
        history=history_text,
        question=question
    )

    answer = call_llm(prompt, temperature=0.2, call_type="query")

    return {
        "answer": answer,
        "sources": [
            {
                "chunk": c["chunk_num"],
                "page": c["page"],
                "file": c["file"],
                "preview": c["content"][:150],
                "chunk_type": c.get("chunk_type", "text"),
                "image_ref": c.get("image_ref")
            }
            for c in chunks
        ],
        "type": "document"
    }


# --- Streaming query ---

def query_document_stream(
    question: str,
    document_id: str = None,
    document_ids: list[str] = None,
    history: list[dict] = None,
    history_summary: str = ""
) -> Generator:
    has_doc = bool(document_ids or document_id)
    q_type = classify_question(question, has_document=has_doc)

    if q_type == "general":
        yield answer_general(question, history, history_summary)
        return

    ids = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1
    chunks = hybrid_search(question, document_ids=ids)

    if not chunks:
        yield answer_general(question, history, history_summary)
        return

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])
    history_text = format_history(history or [], history_summary)
    prompt = (QA_PROMPT_MULTI if is_multi else QA_PROMPT).format(
        chunks=chunks_text,
        history=history_text,
        question=question
    )

    stream_gen = call_llm_stream(prompt, call_type="query_stream")
    for token in stream_gen:
        yield token


# --- Extraction ---
def extract_fields(document_id: str, fields: dict) -> dict:
    from schemas.validator import validate_extraction

    all_chunks = get_all_chunks()
    doc_chunks = [c for c in all_chunks if c["document_id"] == document_id]
    context = "\n\n".join([c["content"] for c in doc_chunks[:10]])

    fields_with_desc = "\n".join([
        f"- {key}: {value if value and isinstance(value, str) else 'extract from document'}"
        for key, value in fields.items()
    ])

    prompt = EXTRACTION_PROMPT.format(
        fields_with_descriptions=fields_with_desc,
        context=context
    )

    extracted = call_llm(
        prompt,
        temperature=0.0,
        json_mode=True,
        call_type="extract"
    )

    if "error" in extracted:
        return {"extracted": extracted, "validation": None}

    validation = validate_extraction(extracted, fields)

    return {
        "extracted": extracted,
        "validation": validation
    }

# --- Table extraction ---

def extract_tables(document_id: str) -> list[dict]:
    all_chunks = get_all_chunks()
    doc_chunks = [c for c in all_chunks if c["document_id"] == document_id]
    if not doc_chunks:
        return []

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    result = call_llm(
        f"""Extract ALL tables from the document below.
For each table return a JSON object with:
- "title": descriptive title for the table
- "headers": list of column names
- "rows": list of rows, each row is a list of values
- "chart_type": suggest "bar", "line", or "pie" based on the data

Return ONLY a JSON array of tables. No explanation, no markdown fences.
If no tables found, return [].

Document:
{context}

JSON array:""",
        temperature=0.0,
        json_mode=True,
        call_type="extract_tables"
    )

    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "error" in result:
        return []
    return []


# --- Summary generation ---

def generate_summary(document_id: str) -> dict:
    all_chunks = get_all_chunks()
    doc_chunks = [c for c in all_chunks if c["document_id"] == document_id]
    if not doc_chunks:
        return {"summary": "", "summary_short": ""}

    context = "\n\n".join([c["content"] for c in doc_chunks[:15]])

    parsed = call_llm(
        f"""Analyze the document below and return a JSON object with exactly these keys:
- "short": one sentence (max 20 words) describing what this document is
- "overview": 2-3 sentence overview of the document
- "key_topics": list of 3-5 main topics covered
- "entities": list of important names, companies, or organizations mentioned
- "dates": list of important dates mentioned (empty list if none)
- "amounts": list of important numbers, amounts, or figures mentioned (empty list if none)
- "document_type": type of document (e.g. "Resume", "Invoice", "Contract", "Report", "Article")

Return ONLY valid JSON, no explanation, no markdown fences.

Document:
{context}

JSON:""",
        temperature=0.0,
        json_mode=True,
        call_type="summarize"
    )

    if isinstance(parsed, dict) and "error" not in parsed:
        return {
            "summary": json.dumps(parsed),
            "summary_short": parsed.get("short", "")
        }
    return {"summary": "", "summary_short": ""}

def nl_to_schema(instruction: str) -> dict:
    """Convert plain English instruction to extraction schema"""
    result = call_llm(
        f"""Convert the following extraction instruction into a JSON schema for document extraction.

Rules:
- Return a JSON object where keys are snake_case field names
- Values are clear descriptions of what to extract for that field
- Include all fields mentioned or implied by the instruction
- For list fields (skills, items, etc.) use descriptive values ending in "as a list"
- Be specific about which entity the field belongs to

Instruction: {instruction}

Return ONLY valid JSON, no explanation, no markdown.

JSON schema:""",
        temperature=0.0,
        json_mode=True,
        call_type="nl_to_schema"
    )
    return result


def extract_nl(document_id: str, instruction: str) -> dict:
    """
    Natural language extraction pipeline:
    1. Convert instruction to schema
    2. Run extraction with validation
    3. Return schema + extracted + validation
    """
    # Step 1 — Convert instruction to schema
    schema = nl_to_schema(instruction)

    if "error" in schema:
        return {
            "error": "Could not parse instruction into schema",
            "instruction": instruction,
            "schema": None,
            "extracted": None,
            "validation": None
        }

    # Step 2 — Run extraction using existing pipeline
    result = extract_fields(document_id, schema)

    return {
        "instruction": instruction,
        "schema": schema,
        "extracted": result.get("extracted"),
        "validation": result.get("validation")
    }
    
"""
APPEND THIS BLOCK TO THE BOTTOM OF retrieval.py
"""

# ── Document classification ───────────────────────────────────────────────────

# Maps LLM-returned doc_type strings to the matching extraction template ID.
# Template IDs must match keys returned by schemas/templates.py::list_templates().
TEMPLATE_MAP = {
    "invoice":              "invoice",
    "receipt":              "invoice",        # close enough — reuse invoice template
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

CONFIDENCE_THRESHOLD = 0.75  # below this → requires_human_review = True

DOCUMENT_CLASSIFIER_PROMPT = """You are a document classification expert.

Analyse the document text below and return ONLY a valid JSON object with:
- "doc_type": one of: invoice, receipt, resume, cv, contract, agreement, nda, report,
  research paper, financial statement, balance sheet, income statement, medical record,
  prescription, legal document, court filing, article, email, letter, general
- "confidence": float 0.0–1.0 (how confident you are)
- "reasoning": one sentence explaining your classification
- "key_signals": list of 2–4 short phrases from the document that led to this classification

Return ONLY valid JSON — no explanation, no markdown fences.

Document (first ~2000 characters):
{context}

JSON:"""


from core.logger import get_logger as _get_logger
from schemas.templates import get_template_for_doc_type
 
_clf_logger = _get_logger("retrieval.classify")
 
DOCUMENT_CLASSIFIER_PROMPT = """You are a document classification expert.
 
Analyse the document text below and return ONLY a valid JSON object with:
- "doc_type": one of: invoice, receipt, resume, cv, contract, agreement, nda, report,
  research paper, financial statement, balance sheet, income statement, medical record,
  prescription, legal document, court filing, article, email, letter, general,
  gst return, gstr-1, gstr-3b, offer letter, loan application, bank statement
- "confidence": float 0.0–1.0 (how confident you are)
- "reasoning": one sentence explaining your classification
- "key_signals": list of 2–4 short phrases from the document that led to this classification
 
Return ONLY valid JSON — no explanation, no markdown fences.
 
Document (first ~2000 characters):
{context}
 
JSON:"""
 
 
def _get_confidence_threshold() -> float:
    """
    Read the confidence threshold from app config.
    Falls back to 0.75 if config is not available (e.g. during tests).
    """
    try:
        from core.config import config as app_config
        return float(app_config.classification_confidence_threshold)
    except Exception:
        return 0.75
 
 
def classify_document(document_id: str) -> dict:
    """
    Classify a document into a known type using LLM + confidence scoring.
 
    Returns a dict with:
        doc_type, schema_template, confidence,
        reasoning, key_signals, requires_human_review
 
    Never raises — always returns a safe default on failure.
    """
    all_chunks = get_all_chunks()
    doc_chunks = [c for c in all_chunks if c["document_id"] == document_id]
 
    if not doc_chunks:
        _clf_logger.warning("No chunks found for classification", document_id=document_id)
        return {
            "doc_type": "general",
            "schema_template": "custom",
            "confidence": 0.0,
            "reasoning": "No document content found.",
            "key_signals": [],
            "requires_human_review": True,
        }
 
    # Use first ~2000 chars — enough signal, cheap to run
    context = "\n\n".join([c["content"] for c in doc_chunks[:5]])[:2000]
 
    try:
        result = call_llm(
            DOCUMENT_CLASSIFIER_PROMPT.format(context=context),
            temperature=0.0,
            json_mode=True,
            call_type="classify_document",
        )
    except Exception as exc:
        _clf_logger.error("LLM call failed during classification", document_id=document_id, error=str(exc))
        return {
            "doc_type": "general",
            "schema_template": "custom",
            "confidence": 0.0,
            "reasoning": f"Classification error: {exc}",
            "key_signals": [],
            "requires_human_review": True,
        }
 
    if not isinstance(result, dict) or "doc_type" not in result:
        _clf_logger.error("LLM returned unexpected format", document_id=document_id, result_type=type(result).__name__)
        return {
            "doc_type": "general",
            "schema_template": "custom",
            "confidence": 0.0,
            "reasoning": "LLM returned unexpected format.",
            "key_signals": [],
            "requires_human_review": True,
        }
 
    doc_type = result.get("doc_type", "general").lower().strip()
    confidence = float(result.get("confidence", 0.0))
    schema_template = get_template_for_doc_type(doc_type)
    threshold = _get_confidence_threshold()
 
    _clf_logger.info(
        "Document classified",
        document_id=document_id,
        doc_type=doc_type,
        confidence=round(confidence, 3),
        schema_template=schema_template,
        requires_review=confidence < threshold,
    )
 
    return {
        "doc_type": doc_type,
        "schema_template": schema_template,
        "confidence": confidence,
        "reasoning": result.get("reasoning", ""),
        "key_signals": result.get("key_signals", []),
        "requires_human_review": confidence < threshold,
    }