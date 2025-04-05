import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from Faiss_index.faiss import retrieve_context

def load_models():
    """Load SentenceTransformer for embeddings & fine-tuned BERT for QA"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", device=device)
    qa_pipeline = pipeline("question-answering", model="Dhruv-2902/fine-tuned-bert-squad", device=0 if device == "cuda" else -1)
    
    print("✅ Models loaded successfully!")
    return embedding_model, qa_pipeline

def answer_question(question, faiss_index, passages, embedding_model, qa_pipeline):
    """Retrieve context from FAISS and answer using fine-tuned BERT"""
    context = retrieve_context(question, faiss_index, passages, embedding_model)
    if not context:
        return "No relevant context found."

    result = qa_pipeline(question=question, context=context)
    return result['answer']
