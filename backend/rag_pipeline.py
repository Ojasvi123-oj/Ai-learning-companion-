from embeddings import embed_query
from vector_store import search
from documents_store import get_documents
from config import GROQ_API_KEY, LLM_MODEL
from groq import Groq

client = Groq(api_key=GROQ_API_KEY)

chat_memory = []


def clear_memory():
    global chat_memory
    chat_memory = []


def extract_topics(text):

    prompt = f"""
Extract the main topics covered in this document.

{text[:5000]}
"""

    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return r.choices[0].message.content.strip()


# ================= GENERAL AI =================
def general_llm_answer(query):

    global chat_memory

    history = ""
    for m in chat_memory[-5:]:
        history += f"{m['role']}: {m['content']}\n"

    prompt = f"""
You are a helpful AI assistant.

Conversation History:
{history}

User Question:
{query}

Give a clear and helpful answer.
"""

    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = r.choices[0].message.content.strip()

    chat_memory.append({"role": "user", "content": query})
    chat_memory.append({"role": "assistant", "content": answer})

    return answer


# ================= MAIN RAG =================
def ask_question(query):

    global chat_memory

    q = query.lower()

    # ===== TOPIC MODE =====
    if "topic" in q or "topics" in q:

        docs = get_documents()

        if not docs:
            return {"response": "No documents uploaded.", "fallback": False}

        out = ""

        for d in docs.values():
            out += f"{d['name']}:\n{d['topics']}\n\n"

        return {"response": out, "fallback": False}


    # ===== QUERY EXPANSION (IMPORTANT) =====
    query_expanded = f"""
Explain in detail: {query}.
Include definition, explanation, and examples if available.
"""

    vec = embed_query(query_expanded)

    docs = search(vec, 10)

    # ===== SORT ALWAYS =====
    docs = sorted(docs, key=lambda x: x["score"])

    # ===== ALWAYS KEEP TOP 3 =====
    docs = docs[:3]

    # ===== SOFT FILTER (NOT STRICT) =====
    docs = [d for d in docs if d["score"] < 1.5]

    # ===== DEBUG LOG =====
    print("\n--- RETRIEVAL DEBUG ---")
    for d in docs:
        print(f"Score: {d['score']}")
        print(d["text"][:200])
        print("------")

    # ===== IF STILL NOTHING =====
    if not docs:
        return {
            "response": "The uploaded documents do not contain this information.",
            "fallback": True
        }

    context = "\n\n".join([d["text"] for d in docs])

    history = ""
    for m in chat_memory[-5:]:
        history += f"{m['role']}: {m['content']}\n"

    prompt = f"""
You are an AI assistant answering questions from uploaded documents.

Instructions:
- Use the context as the main source.
- If partial information exists, still answer.
- Do NOT say "no information" unless completely irrelevant.
- Do NOT use markdown symbols like ** or *.

Conversation History:
{history}

Context:
{context}

Question:
{query}

Give a clear, well-explained answer.
"""

    r = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = r.choices[0].message.content.strip()

    chat_memory.append({"role": "user", "content": query})
    chat_memory.append({"role": "assistant", "content": answer})

    return {"response": answer, "fallback": False}