QA_PROMPT = """You are a helpful document assistant. Answer the user's question based ONLY on the provided document chunks below.

For each point in your answer, cite the source chunk number like [1], [2] etc.
If the answer is not found in the chunks, say "I couldn't find this in the document."

Document chunks:
{chunks}

Question: {question}

Answer:"""