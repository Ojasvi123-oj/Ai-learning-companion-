import faiss
import pickle
import os
import numpy as np
from config import INDEX_PATH, META_PATH

index = None
metadata = []


def load_index():

    global index, metadata

    if os.path.exists(INDEX_PATH):

        index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)

    else:

        index = faiss.IndexFlatL2(384)
        metadata = []

def save():
    import os

    os.makedirs("data/index", exist_ok=True)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def rebuild_index(vectors):

    global index

    dim = len(vectors[0])

    index = faiss.IndexFlatL2(dim)

    index.add(np.array(vectors).astype("float32"))


def add_documents(texts, vectors):

    global metadata

    if index.ntotal == 0:
        index.add(np.array(vectors).astype("float32"))
    else:
        index.add(np.array(vectors).astype("float32"))

    metadata.extend(texts)

    save()

def search(vec, top_k=10):

    if index.ntotal == 0:
        return []

    D, I = index.search(np.array([vec]).astype("float32"), top_k)

    results = []

    for dist, idx in zip(D[0], I[0]):

        if idx < len(metadata):

            item = metadata[idx].copy()
            item["score"] = float(dist)

            results.append(item)

    return results

def delete_doc(doc_id):

    global index, metadata

    if index.ntotal == 0:
        return

    new_metadata = []
    new_vectors = []

    for i, m in enumerate(metadata):

        if m["doc_id"] != doc_id:
            new_metadata.append(m)

            try:
                vec = index.reconstruct(i)
                new_vectors.append(vec)
            except:
                pass

    # If nothing left → reset index
    if len(new_vectors) == 0:
        index = faiss.IndexFlatL2(384)
        metadata = []
        save()
        return

    # Rebuild index with remaining vectors
    dim = len(new_vectors[0])
    index = faiss.IndexFlatL2(dim)

    index.add(np.array(new_vectors).astype("float32"))

    metadata = new_metadata

    save()