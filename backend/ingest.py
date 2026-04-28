from pypdf import PdfReader
from docx import Document
import uuid
import os

from embeddings import embed_texts
from vector_store import add_documents
from documents_store import save_document
from rag_pipeline import extract_topics


def chunk(text, size=500, overlap=100):

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        part = " ".join(words[start:start + size])
        chunks.append(part)
        start += size - overlap

    return chunks


def extract_text(path):

    ext = path.lower().split(".")[-1]

    # ===== PDF =====
    if ext == "pdf":
        try:
            reader = PdfReader(path)

            pages = []

            for i, page in enumerate(reader.pages):
                t = page.extract_text()
                if t:
                    pages.append((i, t))

            return pages

        except Exception as e:
            print("PDF ERROR:", e)
            return []

    # ===== DOCX =====
    elif ext == "docx":
        try:
            doc = Document(path)

            full_text = "\n".join([p.text for p in doc.paragraphs])

            return [(0, full_text)]

        except Exception as e:
            print("DOCX ERROR:", e)
            return []

    # ===== TXT =====
    elif ext == "txt":
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        return [(0, text)]

    else:
        print("Unsupported file type:", ext)
        return []


# ================= MAIN INGEST =================
def ingest_pdf(path):

    doc_id = str(uuid.uuid4())
    file_name = os.path.basename(path)

    pages = extract_text(path)

    if not pages:
        print("No text extracted from file.")
        return None

    full_text = ""
    texts = []

    for page_num, page_text in pages:

        if not page_text.strip():
            continue

        full_text += page_text + "\n"

        chunks = chunk(page_text)

        for c in chunks:
            texts.append({
                "text": c,
                "doc_id": doc_id,
                "file_name": file_name,
                "page": page_num
            })

    if not texts:
        print("No usable chunks created.")
        return None

    # ===== EMBEDDINGS =====
    vectors = embed_texts([t["text"] for t in texts])

    add_documents(texts, vectors)

    # ===== TOPIC EXTRACTION =====
    topics = extract_topics(full_text)

    save_document(doc_id, file_name, topics)

    return doc_id