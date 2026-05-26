import os
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from db import get_all_chunks
from ingestion import get_embed_model
from prompts import QA_PROMPT
import numpy as np

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_search(question: str, document_id: str = None, top_k: int = 5) -> list[dict]:
    embed_model = get_embed_model()
    all_chunks = get_all_chunks()

    # Filter by document if specified
    if document_id:
        all_chunks = [c for c in all_chunks if c["document_id"] == document_id]

    if not all_chunks:
        return []

    texts = [c["content"] for c in all_chunks]
    question_embedding = embed_model.get_text_embedding(question)

    # --- Dense search (vector similarity) ---
    dense_scores = []
    for chunk in all_chunks:
        # Convert embedding from string to list of floats if needed
        raw_emb = chunk["embedding"]
        if isinstance(raw_emb, str):
            import json
            raw_emb = json.loads(raw_emb)
        score = cosine_similarity(question_embedding, raw_emb)
        dense_scores.append(score)

    # --- Sparse search (BM25 keyword) ---
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.lower().split())

    # --- Reciprocal Rank Fusion ---
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
            "score": rrf[idx]
        }
        for i, idx in enumerate(top_indices)
    ]

def query_document(question: str, document_id: str = None) -> dict:
    # Retrieve top chunks
    chunks = hybrid_search(question, document_id)

    if not chunks:
        return {"answer": "No document found to query.", "sources": []}

    # Format chunks for prompt
    chunks_text = "\n\n".join([
        f"[{c['chunk_num']}] (Page {c['page']}): {c['content']}"
        for c in chunks
    ])

    # Call Groq
    prompt = QA_PROMPT.format(chunks=chunks_text, question=question)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {"chunk": c["chunk_num"], "page": c["page"], "preview": c["content"][:150]}
            for c in chunks
        ]
    }