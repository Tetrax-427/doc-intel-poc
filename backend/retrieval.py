import os
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from db import get_all_chunks
from ingestion import get_embed_model
from prompts import QA_PROMPT, QA_PROMPT_MULTI, GENERAL_PROMPT, CLASSIFIER_PROMPT, QUERY_EXPANSION_PROMPT
import numpy as np
import json
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

import cohere
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def classify_question(question: str, has_document: bool = True) -> str:
    """Returns 'document' or 'general'"""
    if not has_document:
        return "general"

    # Keywords that strongly suggest document intent
    doc_keywords = [
        "this", "the document", "the file", "the letter", "the contract",
        "the invoice", "the report", "the resume", "the agreement", "it says",
        "mentioned", "above", "here", "uploaded", "about"
    ]
    q_lower = question.lower()
    if any(kw in q_lower for kw in doc_keywords):
        return "document"

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": CLASSIFIER_PROMPT.format(question=question)}],
            temperature=0.0,
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().lower()
        return "document" if "document" in result else "general"
    except Exception:
        return "document"

def expand_query(question: str) -> str:
    """Rewrite vague questions to improve retrieval"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(question=question)}],
            temperature=0.0,
            max_tokens=100
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else question
    except Exception:
        return question


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
        # Re-number chunks after reranking
        for i, c in enumerate(reranked):
            c["chunk_num"] = i + 1
        return reranked
    except Exception as e:
        print(f"Reranking failed: {e} — using original order")
        return chunks[:top_k]


def answer_general(question: str, history: list[dict] = None, history_summary: str = "") -> str:
    """Answer a general question from Groq's knowledge"""
    history_text = format_history(history or [], history_summary)
    prompt = GENERAL_PROMPT.format(history=history_text, question=question)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

def compress_history(messages: list[dict]) -> str:
    """Summarize a long conversation into a compact context string"""
    if not messages:
        return ""

    conversation = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}"
        for m in messages
    ])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"""Summarize the following conversation in 3-5 sentences. 
Focus on key facts, questions asked, and answers given. 
Be concise — this will be used as context for future questions.

Conversation:
{conversation}

Summary:"""
        }],
        temperature=0.0,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


def get_history_context(messages: list[dict], summary: str = "", window: int = 4) -> str:
    """Build history context from summary + recent messages"""
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


def format_history(messages: list[dict], summary: str = "") -> str:
    return get_history_context(messages, summary)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_search(question: str, document_ids: list[str] = None, top_k: int = 10) -> list[dict]:
    embed_model = get_embed_model()
    all_chunks = get_all_chunks()

    if document_ids:
        all_chunks = [c for c in all_chunks if c["document_id"] in document_ids]

    if not all_chunks:
        return []

    # Expand query before retrieval
    expanded_question = expand_query(question)
    print(f"Expanded query: {expanded_question}")

    texts = [c["content"] for c in all_chunks]
    question_embedding = embed_model.get_text_embedding(expanded_question)

    # Dense search
    dense_scores = []
    for chunk in all_chunks:
        raw_emb = chunk["embedding"]
        if isinstance(raw_emb, str):
            raw_emb = json.loads(raw_emb)
        score = cosine_similarity(question_embedding, raw_emb)
        dense_scores.append(score)

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
            "score": rrf[idx]
        }
        for i, idx in enumerate(top_indices)
    ]

    # Rerank with Cohere
    return rerank_chunks(question, candidates, top_k=5)

def extract_fields(document_id: str, schema: dict) -> dict:
    # Get all chunks for this document
    all_chunks = get_all_chunks()
    doc_chunks = [c for c in all_chunks if c["document_id"] == document_id]

    # Use first 10 chunks as context (enough for most docs)
    context = "\n\n".join([c["content"] for c in doc_chunks[:10]])

    prompt = f"""Extract the following fields from the document below.
Return ONLY a valid JSON object with these exact keys: {list(schema.keys())}
If a field is not found, use null for strings or [] for lists.
Do not include any explanation or markdown, just the JSON.

Document:
{context}

Fields to extract: {json.dumps(schema, indent=2)}

JSON output:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    raw = response.choices[0].message.content.strip()

    # Clean markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = {"error": "Could not parse extraction output", "raw": raw}

    return {"extracted": extracted}

from typing import Generator

def query_document(question: str, document_id: str = None, document_ids: list[str] = None, history: list[dict] = None, history_summary: str = "") -> dict:
    # Classify question
    
    has_doc = bool(document_ids or document_id)
    q_type = classify_question(question, has_document=has_doc)

    if q_type == "general":
        answer = answer_general(question, history, history_summary)
        return {"answer": answer, "sources": [], "type": "general"}

    ids = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1

    chunks = hybrid_search(question, document_ids=ids)

    if not chunks:
        # Fallback to general if no chunks found
        answer = answer_general(question, history, history_summary)
        return {"answer": answer, "sources": [], "type": "general"}

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])

    history_text = format_history(history or [], history_summary)
    prompt_template = QA_PROMPT_MULTI if is_multi else QA_PROMPT
    prompt = prompt_template.format(chunks=chunks_text, history=history_text, question=question)

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"chunk": c["chunk_num"], "page": c["page"], "file": c["file"], "preview": c["content"][:150]}
            for c in chunks
        ],
        "type": "document"
    }

def query_document_stream(question: str, document_id: str = None, document_ids: list[str] = None, history: list[dict] = None, history_summary: str = "") -> Generator:
    # Classify question
    has_doc = bool(document_ids or document_id)
    q_type = classify_question(question, has_document=has_doc)
    if q_type == "general":
        answer = answer_general(question, history, history_summary)
        yield answer
        return

    ids = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1

    chunks = hybrid_search(question, document_ids=ids)

    if not chunks:
        answer = answer_general(question, history, history_summary)
        yield answer
        return

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])

    history_text = format_history(history or [], history_summary)
    prompt_template = QA_PROMPT_MULTI if is_multi else QA_PROMPT
    prompt = prompt_template.format(chunks=chunks_text, history=history_text, question=question)

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta