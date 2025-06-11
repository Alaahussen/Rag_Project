import os
import re
import spacy
import warnings
import shutil
from pathlib import Path
from collections import Counter

import streamlit as st
from dotenv import load_dotenv

from loaders import (
    load_documents,
    load_document,
)

from splitter import (
    split_by_chunk_size,
)

from vectorstore import (
    create_vector_store,
    load_vector_store,
)

from qa_chain import (
    create_qa_chain,
)

from query import (
    rag_query,
)

from gemini_generate import (
    generate_response,
)

from utils import (
    extract_name_spacy,
    extract_matched_names_from_query,
)

# Setup
warnings.filterwarnings("ignore")
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

# UI config
st.set_page_config(page_title="CV Ranker", layout="wide")

# CSS styling (unchanged from your original)
st.markdown("""
    <style>
    /* Your existing CSS styles */
    </style>
""", unsafe_allow_html=True)

# Directory for uploads
upload_dir = "uploaded_cvs"

def clear_upload_directory():
    """Empty the upload directory completely"""
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    os.makedirs(upload_dir, exist_ok=True)

# Clear uploads at start
clear_upload_directory()

# Sidebar Navigation
page = st.sidebar.selectbox("🔍 Navigate", ["🏆 Rank Candidates", "💬 Candidate Chatbot"])

# Session State Initialization
for key in ["out", "first_name_dict", "full_name_dict"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# ================= Page 1: Ranking ==================
if page == "🏆 Rank Candidates":
    st.markdown('<div class="title-center">🏆 Rank Top Candidates Based on Job Description</div>', unsafe_allow_html=True)
    
    # Clear previous uploads when page loads
    clear_upload_directory()
    
    uploaded_files = st.file_uploader(
        "📂 Upload CVs (PDF format)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(upload_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

    criteria = st.text_area("📋 Job Description / Criteria", height=150,
                          placeholder="Enter the job requirements, skills needed, and other relevant criteria...")
    ranking_key = "ranking_result"

    if st.button("🌟 Rank Candidates", type="primary"):
        if os.path.isdir(upload_dir) and len(os.listdir(upload_dir)) > 0:
            if not criteria.strip():
                st.error("Please enter job criteria before ranking")
                st.stop()
                
            with st.spinner("🔍 Analyzing candidates..."):
                try:
                    documents, candidate_names = load_documents(upload_dir)
                    chunks = split_by_chunk_size(documents)
                    vector_store = create_vector_store(chunks)

                    all_docs = vector_store.similarity_search(criteria, k=30)
                    candidate_counter = Counter()

                    for doc in all_docs:
                        candidate = doc.metadata.get("candidate_name")
                        if candidate in candidate_names:
                            candidate_counter[candidate] += 1

                    top_candidates = [name for name, _ in candidate_counter.most_common(5)]
                    evaluations = []

                    st.session_state.out = {}
                    st.session_state.first_name_dict = {}
                    st.session_state.full_name_dict = {}

                    for candidate in top_candidates:
                        cv_path = os.path.join(upload_dir, candidate + ".pdf")
                        st.session_state.out[candidate] = cv_path

                        name_parts = extract_name_spacy(candidate).lower().split()
                        if name_parts:
                            st.session_state.first_name_dict[name_parts[0]] = cv_path
                            if len(name_parts) > 1:
                                full_name = " ".join(name_parts[:2])
                                st.session_state.full_name_dict[full_name] = cv_path

                        candidate_docs = [doc for doc in all_docs if doc.metadata.get("candidate_name") == candidate][:5]
                        context = "\n\n".join([doc.page_content for doc in candidate_docs])

                        prompt = f"""You are an expert HR assistant evaluating a candidate for a position.

Given the resume information below for candidate {candidate}, evaluate the candidate specifically on this criteria:
{criteria}

---
📄 Resume Information:
{context}
---

Provide a score from 1 to 10 and a justification.
Format:
CANDIDATE: {candidate}
SCORE: [1-10]
JUSTIFICATION: [Detailed explanation]"""
                        evaluation = generate_response(prompt)
                        evaluations.append(evaluation)

                    all_evaluations = "\n\n".join(evaluations)
                    ranking_prompt = f"""You are an expert HR assistant summarizing the top 5 candidates for a position requiring:
{criteria}

Based on the evaluations below, summarize *why* these candidates are the best fit and rank them.

Evaluations:
{all_evaluations}

Format your response as:
1. Candidate Name (Score): Summary of why ranked here
2. ..."""
                    ranking = generate_response(ranking_prompt)
                    st.session_state[ranking_key] = ranking
                    
                    st.subheader("🏅 Top 5 Candidates")
                    st.markdown(ranking)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please upload CVs first")
    else:
        if ranking_key in st.session_state:
            st.subheader("🏅 Top 5 Candidates")
            st.markdown(st.session_state[ranking_key])

# ================= Page 2: Chatbot ==================
elif page == "💬 Candidate Chatbot":
    st.markdown('<div class="title-center">💬 Candidate Chatbot</div>', unsafe_allow_html=True)
    
    if not st.session_state.out:
        st.warning("⚠️ Please upload CVs and rank candidates on the 'Rank Candidates' page first")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
        st.session_state.last_candidate_key = None

    query = st.chat_input("Ask about a candidate ")

    if query:
        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if query.strip().lower() in greetings or re.match(r"^(hi|hello|hey)[.!\s]*$", query.strip(), re.IGNORECASE):
            st.session_state.chat_history.append((query, "👋 Hello! How can I assist you with candidate information today?"))
        else:
            candidate_names = extract_matched_names_from_query(
                query,
                st.session_state.full_name_dict,
                st.session_state.first_name_dict
            )

            candidate_key = None
            if candidate_names:
                candidate_key = candidate_names[0].lower()
                if candidate_key != st.session_state.last_candidate_key:
                    doc_path = st.session_state.full_name_dict.get(candidate_key) or \
                               st.session_state.first_name_dict.get(candidate_key)
                    if doc_path:
                        with st.spinner(f"📖 Loading {candidate_key}'s CV..."):
                            document = load_document(doc_path)
                            chunks = split_by_chunk_size(document)
                            vector_store = create_vector_store(chunks)
                            st.session_state.qa_chain = create_qa_chain(vector_store)
                            st.session_state.last_candidate_key = candidate_key
                    else:
                        st.warning(f"⚠️ Couldn't find CV for {candidate_key}")
                        st.stop()

            if st.session_state.qa_chain:
                with st.spinner("💭 Analyzing CV..."):
                    try:
                        result = rag_query(query, st.session_state.qa_chain)
                        answer = result.get("answer", "⚠️ Couldn't generate an answer")
                        st.session_state.chat_history.append((query, answer))
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("🔍 Please mention a candidate name in your question")

    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f'<div class="user-message">🧑 <strong>You:</strong><br>{q}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="assistant-message">🤖 <strong>Assistant:</strong><br>{a}</div>', unsafe_allow_html=True)
            st.markdown('<div class="clearfix"></div>', unsafe_allow_html=True)
