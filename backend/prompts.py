QA_PROMPT = """You are a helpful document assistant. Answer the user's question based ONLY on the provided document chunks below.

For each point in your answer, cite the source chunk number like [1], [2] etc.
If the answer is not found in the chunks, say "I couldn't find this in the document."

Document chunks:
{chunks}

Question: {question}

Answer:"""


QA_PROMPT_MULTI = """You are a helpful document assistant. Answer the user's question based ONLY on the provided chunks below, which come from multiple documents.

For each point in your answer, cite the source like [Doc: filename, Page: X].
If the answer is not found in the chunks, say "I couldn't find this in the selected documents."

Document chunks:
{chunks}

Question: {question}

Answer:"""