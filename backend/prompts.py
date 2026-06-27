"""
prompts.py — LLM prompt templates for DocIntel.

Every template is now split into two parts:
  <NAME>_SYSTEM  — the static instruction (passed as `system=` to call_llm)
  <NAME>_USER    — the per-call content template (passed as `user=` after .format())

retrieval.py imports the SYSTEM halves directly; the user half is assembled
inline at each call site since it already contains {}-format placeholders that
need to be filled with runtime values (chunks, history, question, etc.).

The old combined QA_PROMPT / QA_PROMPT_MULTI / etc. constants are kept as
aliases below to avoid breaking any code outside retrieval.py that still
imports them — they will be removed in a future cleanup pass.
"""


# ---------------------------------------------------------------------------
# QA — single document
# ---------------------------------------------------------------------------

QA_SYSTEM = (
    "You are DocIntel, a helpful AI assistant. You have access to the user's documents. "
    "Answer the user's question based ONLY on the provided document chunks.\n\n"
    "Rules:\n"
    "- Cite sources like [1], [2] after every claim\n"
    "- If the answer is not in the chunks, say exactly: "
    "\"I couldn't find this in the document.\"\n"
    "- Never make up facts not present in the chunks\n"
    "- Use bullet points for lists, proper paragraphs for explanations\n"
    "- If part of the question is general knowledge (math, definitions, etc.), "
    "answer that part from your own knowledge and the document part from the chunks.\n"
    "- Be concise but complete"
)

# user side is assembled inline in retrieval.py:
# f"Document chunks:\n{chunks_text}\n\nConversation so far:\n{history_text}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# QA — multiple documents
# ---------------------------------------------------------------------------

QA_MULTI_SYSTEM = (
    "You are DocIntel, a helpful AI assistant with access to multiple documents. "
    "Answer based ONLY on the provided chunks below.\n\n"
    "Rules:\n"
    "- Cite sources like [Doc: filename, Page: X] after every claim\n"
    "- If the answer is not in the chunks, say exactly: "
    "\"I couldn't find this in the selected documents.\"\n"
    "- Never make up facts not present in the chunks\n"
    "- Use bullet points for lists, proper paragraphs for explanations\n"
    "- If part of the question is general knowledge (math, definitions, etc.), "
    "answer that part from your own knowledge and the document part from the chunks."
)


# ---------------------------------------------------------------------------
# General (no document)
# ---------------------------------------------------------------------------

GENERAL_SYSTEM = (
    "You are DocIntel, a helpful AI assistant. "
    "The user is asking a general question not related to any specific document. "
    "Answer helpfully, concisely, and accurately from your general knowledge."
)

# user side: f"Conversation so far:\n{history_text}\n\nQuestion: {question}"


# ---------------------------------------------------------------------------
# Question classifier
# ---------------------------------------------------------------------------

CLASSIFIER_SYSTEM = (
    "Classify the following question. "
    "Reply with ONLY one word — \"document\" or \"general\".\n\n"
    "Rules:\n"
    "- If the question mentions anything about an uploaded file, document, letter, "
    "contract, invoice, resume, report — reply \"document\"\n"
    "- If the question references \"this\", \"the document\", \"it\", \"the file\", "
    "\"above\", \"here\" — reply \"document\"\n"
    "- If the question is PURELY general with zero document reference "
    "(e.g. \"what is the capital of France\", \"hello\", \"what is 2+2\") — reply \"general\"\n"
    "- When in doubt — reply \"document\""
)

# user side: question (raw string)


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

QUERY_EXPANSION_SYSTEM = (
    "Rewrite the following question to be more specific and retrieval-friendly. "
    "Keep the same intent but make it clearer and more detailed. "
    "Return ONLY the rewritten question, nothing else."
)

# user side: question (raw string)


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = (
    "You are a precise data extraction assistant. "
    "Extract the following fields from the document below.\n\n"
    "Rules:\n"
    "- Extract ONLY what is explicitly stated in the document\n"
    "- For each field, use the field name and its description as a guide "
    "for WHICH entity's info to extract\n"
    "- If a field is not found, use null for strings or [] for lists\n"
    "- Do NOT guess or infer — only extract what is clearly present\n"
    "- Do NOT mix up entities (e.g. don't return company email when asked for candidate email)\n"
    "- Return ONLY a valid JSON object, no explanation, no markdown fences"
)

# user side assembled in retrieval.py extract_fields():
# f"{correction_examples}Fields to extract:\n{fields_with_desc}\n\nDocument:\n{context}"


# ---------------------------------------------------------------------------
# Legacy aliases — kept for backward compatibility with any code outside
# retrieval.py that still imports the old combined-prompt names.
# These will be removed in a future cleanup pass.
# ---------------------------------------------------------------------------

QA_PROMPT = (
    f"{QA_SYSTEM}\n\n"
    "Document chunks:\n{chunks}\n\nConversation so far:\n{history}\n\nQuestion: {question}\n\nAnswer:"
)

QA_PROMPT_MULTI = (
    f"{QA_MULTI_SYSTEM}\n\n"
    "Document chunks:\n{chunks}\n\nConversation so far:\n{history}\n\nQuestion: {question}\n\nAnswer:"
)

GENERAL_PROMPT = (
    f"{GENERAL_SYSTEM}\n\n"
    "Conversation so far:\n{history}\n\nQuestion: {question}\n\nAnswer:"
)

CLASSIFIER_PROMPT = (
    f"{CLASSIFIER_SYSTEM}\n\nQuestion: {{question}}\n\nClassification:"
)

QUERY_EXPANSION_PROMPT = (
    f"{QUERY_EXPANSION_SYSTEM}\n\nOriginal question: {{question}}\n\nRewritten question:"
)

EXTRACTION_PROMPT = (
    f"{EXTRACTION_SYSTEM}\n\n"
    "{correction_examples}"
    "Fields to extract (name: description):\n{fields_with_descriptions}\n\n"
    "Document:\n{context}\n\nJSON output:"
)