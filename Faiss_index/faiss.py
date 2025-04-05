import faiss
import numpy as np
import os

def build_faiss_index(passages, model, index_path="models/faiss_index_optimized.bin"):
    """Build and save FAISS index"""
    index_path = "models/faiss_index_optimized.bin"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    embeddings = model.encode(passages, convert_to_numpy=True, show_progress_bar=True)
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, 32)
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    print("✅ FAISS HNSW index created & saved!")
    return index

def load_or_create_faiss_index(passages, model, index_path="models/faiss_index_optimized.bin"):
    """Load existing FAISS index or create a new one"""
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        print("✅ FAISS index loaded!")
        return index
    return build_faiss_index(passages, model)


def retrieve_context(question, faiss_index, passages, model, top_k=1):
    """Retrieve most relevant context for the question"""
    question_embedding = model.encode([question], convert_to_numpy=True)
    D, I = faiss_index.search(question_embedding, top_k)

    retrieved_texts = [passages[i] for i in I[0] if i < len(passages)]
    return " ".join(retrieved_texts) if retrieved_texts else None
