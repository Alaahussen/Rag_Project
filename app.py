import os
import re
import spacy
import warnings
from pathlib import Path
from collections import Counter

import streamlit as st
from dotenv import load_dotenv

from preprocessing import (
    load_documents,
    load_document,
    split_by_chunk_size,
    create_vector_store,
    create_qa_chain,
    rag_query,
    generate_response,
    extract_name_spacy,
    extract_matched_names_from_query,
)

# Setup
warnings.filterwarnings("ignore")
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyBhY4ePpJttU3PybscdOmbHghDcWmng9Ik"

nlp = spacy.load("en_core_web_sm")

# UI config
st.set_page_config(page_title="CV Ranker", layout="wide")

# CSS styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .chat-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: #f1f1f1;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox("🔍 Navigate", ["🏆 Rank Candidates", "🤖 Candidate Chatbot"])

# Session State Initialization
for key in ["out", "first_name_dict", "full_name_dict"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# Directory for uploads
upload_dir = "uploaded_cvs"
os.makedirs(upload_dir, exist_ok=True)

# ================= Page 1: Ranking ==================
if page == "🏆 Rank Candidates":
    st.title("🏆 Rank Top Candidates Based on Job Description")
    uploaded_files = st.file_uploader("📁 Upload multiple CV PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(upload_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

    criteria = st.text_area("📝 Enter Job Description / Criteria", height=200)
    ranking_key = "ranking_result"

    if st.button("🚀 Rank Candidates"):
        if os.path.isdir(upload_dir):
            with st.spinner("Processing CVs..."):
                documents, candidate_names = load_documents(upload_dir)
                chunks = split_by_chunk_size(documents)
                vector_store = create_vector_store(chunks)

                all_docs = vector_store.similarity_search(criteria, k=30)
                candidate_counter = Counter()

                for doc in all_docs:
                    candidate = doc.metadata.get("candidate_name")
                    if candidate in candidate_names:
                        candidate_counter[candidate] += 1

                top_candidates = [name for name, _ in candidate_counter.most_common(3)]
                evaluations = []

                # Clear and rebuild the out dictionary
                st.session_state.out = {}
                st.session_state.first_name_dict = {}
                st.session_state.full_name_dict = {}

                for candidate in top_candidates:
                    cv_path = os.path.join(upload_dir, candidate + ".pdf")
                    st.session_state.out[candidate] = cv_path

                    # Update candidate name dictionaries
                    name_parts = extract_name_spacy(candidate).lower().split()
                    if name_parts:
                        st.session_state.first_name_dict[name_parts[0]] = cv_path
                        if len(name_parts) > 1:
                            full_name = " ".join(name_parts[:2])
                            st.session_state.full_name_dict[full_name] = cv_path

                    candidate_docs = [doc for doc in all_docs if doc.metadata.get("candidate_name") == candidate][:3]
                    context = "\n\n".join([doc.page_content for doc in candidate_docs])

                    prompt = f"""
You are an expert HR assistant evaluating a candidate for a position.

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
JUSTIFICATION: [Detailed explanation]
"""
                    evaluation = generate_response(prompt)
                    evaluations.append(evaluation)

                all_evaluations = "\n\n".join(evaluations)
                ranking_prompt = f"""
You are an expert HR assistant summarizing the top 3 candidates for a position requiring:
{criteria}

Based on the evaluations below, summarize *why* these candidates are the best fit and rank them.

Evaluations:
{all_evaluations}

Format your response as:
1. Candidate Name (Score): Summary of why ranked here
2. ...
"""
                ranking = generate_response(ranking_prompt)
                st.session_state[ranking_key] = ranking
                st.subheader("✅ Top 3 Candidates")
                st.markdown(ranking)
        else:
            st.error("Please upload CVs first.")
    else:
        if ranking_key in st.session_state:
            st.subheader("✅ Top 3 Candidates")
            st.markdown(st.session_state[ranking_key])

# ================= Page 2: Chatbot ==================
elif page == "🤖 Candidate Chatbot":
    st.title("🤖 Candidate Chatbot (Ask Questions About a CV)")

    if not st.session_state.out:
        st.warning("Please upload CVs and rank candidates on the '🏆 Rank Candidates' page first.")
        st.stop()

    if not st.session_state.get("first_name_dict") or not st.session_state.get("full_name_dict"):
        st.session_state.first_name_dict = {}
        st.session_state.full_name_dict = {}
        for k, v in st.session_state.out.items():
            name_parts = extract_name_spacy(k).lower().split()
            if name_parts:
                st.session_state.first_name_dict[name_parts[0]] = v
                if len(name_parts) > 1:
                    full_name = " ".join(name_parts[:2])
                    st.session_state.full_name_dict[full_name] = v

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
        st.session_state.last_candidate_key = None

    query = st.chat_input("Ask a question about a candidate...")

    if query:
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
                    document = load_document(doc_path)
                    chunks = split_by_chunk_size(document)
                    vector_store = create_vector_store(chunks)
                    st.session_state.qa_chain = create_qa_chain(vector_store)
                    st.session_state.last_candidate_key = candidate_key
                else:
                    st.warning("⚠️ Candidate found but CV file is missing.")
                    st.stop()

        if st.session_state.qa_chain:
            with st.spinner("💬 Thinking..."):
                result = rag_query(query, st.session_state.qa_chain)
                answer = result.get("answer", "⚠️ No answer generated.")
                sources = result.get("sources", [])
            st.session_state.chat_history.append((query, answer, sources))
        else:
            st.warning("❌ Please include a known candidate name to begin.")

    if st.session_state.chat_history:
        for i, (q, a, sources) in enumerate(st.session_state.chat_history):
            st.markdown(f"**🧑‍💼 You:** {q}")
            st.markdown(f"**🤖 Answer:** {a}")
            st.markdown("---")
