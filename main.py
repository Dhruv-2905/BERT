import streamlit as st
from model.models import load_models, answer_question
from Faiss_index.faiss import load_or_create_faiss_index
from util.utils import load_json_data, extract_passages
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


st.title("📚 AI-Powered QA System")

json_data = load_json_data("data/output1.json")
passages = extract_passages(json_data)

embedding_model, qa_pipeline = load_models()

faiss_index = load_or_create_faiss_index(passages, embedding_model)


question = st.text_input("🔍 Enter your question:")
if st.button("Get Answer"):
    if question:
        answer = answer_question(question, faiss_index, passages, embedding_model, qa_pipeline)
        st.write(f"📝 **Answer:** {answer}")
    else:
        st.warning("⚠️ Please enter a question!")
