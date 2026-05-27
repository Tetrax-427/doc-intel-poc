import os
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from db import get_all_chunks
from ingestion import get_embed_model
from prompts import QA_PROMPT, QA_PROMPT_MULTI
import numpy as np
import json
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_search(question: str, document_ids: list[str] = None, top_k: int = 5) -> list[dict]:
    embed_model = get_embed_model()
    all_chunks = get_all_chunks()

    # Filter by documents if specified
    if document_ids:
        all_chunks = [c for c in all_chunks if c["document_id"] in document_ids]

    if not all_chunks:
        return []

    texts = [c["content"] for c in all_chunks]
    question_embedding = embed_model.get_text_embedding(question)

    # Dense search
    dense_scores = []
    for chunk in all_chunks:
        raw_emb = chunk["embedding"]
        if isinstance(raw_emb, str):
            import json
            raw_emb = json.loads(raw_emb)
        score = cosine_similarity(question_embedding, raw_emb)
        dense_scores.append(score)

    # BM25
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.lower().split())

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

    return [
        {
            "chunk_num": i + 1,
            "content": all_chunks[idx]["content"],
            "page": all_chunks[idx]["metadata"].get("page", "?"),
            "file": all_chunks[idx]["metadata"].get("file", "?"),
            "score": rrf[idx]
        }
        for i, idx in enumerate(top_indices)
    ]


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

def query_document(question: str, document_id: str = None, document_ids: list[str] = None) -> dict:
    # Build id list
    ids = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1

    chunks = hybrid_search(question, document_ids=ids)

    if not chunks:
        return {"answer": "No documents found to query.", "sources": []}

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])

    prompt_template = QA_PROMPT_MULTI if is_multi else QA_PROMPT
    prompt = prompt_template.format(chunks=chunks_text, question=question)

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
        ]
    }


def query_document_stream(question: str, document_id: str = None, document_ids: list[str] = None) -> Generator:
    ids = document_ids if document_ids else ([document_id] if document_id else None)
    is_multi = ids and len(ids) > 1

    chunks = hybrid_search(question, document_ids=ids)

    if not chunks:
        yield "No documents found to query."
        return

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Doc: {c['file']}, Page {c['page']}): {c['content']}"
        for c in chunks
    ])

    prompt_template = QA_PROMPT_MULTI if is_multi else QA_PROMPT
    prompt = prompt_template.format(chunks=chunks_text, question=question)

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
    chunks = hybrid_search(question, document_id)

    if not chunks:
        yield "No document found to query."
        return

    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Page {c['page']}): {c['content']}"
        for c in chunks
    ])

    prompt = QA_PROMPT.format(chunks=chunks_text, question=question)

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