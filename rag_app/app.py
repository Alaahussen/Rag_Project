import os
import re
import spacy
import shutil
import warnings
from pathlib import Path
from collections import Counter
import tempfile
from collections import Counter
import streamlit as st
from dotenv import load_dotenv
import os
import difflib

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
api_key = os.getenv("API_KEY")
nlp = spacy.load("en_core_web_sm")

# UI config
st.set_page_config(page_title="CV Ranker", layout="wide")


def query_all_resumes(question,vector_store, k=100):
    """Query resumes and return only matching candidates with clean justification for Streamlit display."""
    
    all_docs = vector_store.similarity_search(question, k=k)

    # Group documents by candidate
    candidate_chunks = {}
    for doc in all_docs:
        candidate = doc.metadata.get("candidate_name", "Unknown")
        candidate_chunks.setdefault(candidate, []).append(doc.page_content)

    evaluations = []

    for candidate, pages in candidate_chunks.items():
        context = "\n\n".join(pages[:3])  # Limit to top 3 chunks

        prompt = f"""
You are an expert HR assistant helping with candidate screening.

Given the resume information below for candidate {candidate}, answer the HR professional's question.

---
📄 Resume Information:
{context}
---

❓ HR Question:
{question}

---
🎯 Your Task:
- Say whether this candidate matches the requirement.
- If yes, explain *why* using the resume.
- If no, say clearly: "This candidate is not a match based on the resume."
- Only use information found in the resume.
- Use this format:

Candidate: {candidate}  
Justification: [Your decision and explanation]
"""

        response = generate_response(prompt)

        # Filter out non-matching candidates
        if "not a match" not in response.lower():
            # Extract only candidate name and cleaned justification
            name_match = re.search(r"Candidate:\s*(.+)", response)
            just_match = re.search(r"Justification:\s*(.*)", response, re.DOTALL)

            if name_match and just_match:
                name = name_match.group(1).strip()
                justification = just_match.group(1).strip()

                # Optional: Remove "Yes, this candidate matches the requirement" boilerplate
                justification = re.sub(r"^yes,? this candidate matches the requirement\.?", "", justification, flags=re.IGNORECASE).strip()

                # Combine and format nicely
                evaluations.append(f"{name}\n\n{justification}")

    return "\n\n------------------------------------------------------------------------------------------------------------------------\n\n".join(evaluations)
    
def extract_matches_and_paths(evaluation_output, upload_dir, cutoff=0.6):
    """
    Extract candidate names from evaluation output and match them to PDF files using fuzzy matching.
    Assumes the candidate name is the first line of each block.
    Returns: {candidate_name: full_pdf_path}
    """
    matched = {}
    
    # Get all PDF filenames and their lowercase base names
    pdf_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]
    pdf_basenames = [os.path.splitext(f)[0].lower() for f in pdf_files]

    # Split the evaluation into blocks per candidate
    for block in evaluation_output.strip().split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Assume the first line is the candidate name
        candidate_name = lines[0].strip().lower()

        # Fuzzy match with PDF filenames
        closest = difflib.get_close_matches(candidate_name, pdf_basenames, n=1, cutoff=cutoff)
        if closest:
            match_idx = pdf_basenames.index(closest[0])
            matched[candidate_name] = os.path.join(upload_dir, pdf_files[match_idx])

    return matched

    
# CSS styling
st.markdown("""
    <style>
    /* Button styling */
    .stButton > button {
        background-color: #4CAF50;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stButton > button:active {
        transform: translateY(0);
        color: white !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div > button {
        color: #2c3e50 !important;
        background-color: white !important;
        border: 1px solid #d9d9d9 !important;
    }
    .stFileUploader > div > div > div > button:hover {
        border-color: #4CAF50 !important;
        color: #2c3e50 !important;
    }
    .stFileUploader > div > div > div > button:active {
        color: #2c3e50 !important;
        background-color: white !important;
    }
    
    /* Chat message styling */
    .user-message {
        background-color: #e3f2fd;
        border-radius: 12px 12px 0 12px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-radius: 12px 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 70%;
        float: right;
        clear: both;
    }
    
    /* General styling */
    .title-center {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #2c3e50;
        margin: 20px 0;
    }
    .stTextArea textarea {
        border-radius: 8px;
        padding: 12px;
    }
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox("🔍 Navigate", ["🏆 Rank Candidates", "💬 Candidate Chatbot"])

# Session State Initialization
for key in ["out", "first_name_dict", "full_name_dict"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# Directory for uploads
upload_dir = tempfile.mkdtemp()
#upload_dir = "uploaded_cvs"
#os.makedirs(upload_dir, exist_ok=True)  
# ================= Page 1: Ranking ==================
# ... (keep all previous imports and setup code the same)

# ================= Page 1: Ranking ==================
if page == "🏆 Rank Candidates":
    st.markdown('<div class="title-center">🏆 Rank Top Candidates Based on Job Description</div>', unsafe_allow_html=True)
    
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

                    # Initialize session state dictionaries
                    st.session_state.out = {}
                    st.session_state.first_name_dict = {}
                    st.session_state.full_name_dict = {}

                    # Assume `ranking` is the evaluation output (text)
                    ranking = query_all_resumes(criteria,vector_store)

                    # Store the raw ranking output for display or debugging
                    st.session_state[ranking_key] = ranking

                    # Extract matched candidate names and PDF paths
                    matched_candidates = extract_matches_and_paths(ranking, upload_dir)

                    # Save it to session state for future access
                    st.session_state["out"] = matched_candidates

                    # Initialize dicts if not already in session state
                    if "first_name_dict" not in st.session_state:
                        st.session_state.first_name_dict = {}
                    if "full_name_dict" not in st.session_state:
                        st.session_state.full_name_dict = {}

                    # Populate name dictionaries
                    for candidate, path in matched_candidates.items():
                            name_parts = extract_name_spacy(candidate).lower().split()
                            if name_parts:
                                st.session_state.first_name_dict[name_parts[0]] = path
                            if len(name_parts) > 1:
                                full_name = " ".join(name_parts[:2])
                                st.session_state.full_name_dict[full_name] = path

                    st.subheader("🏅 Top Candidates")
                    #st.markdown(ranking)
                    # Split the full ranking output into blocks per candidate
                    blocks = ranking.strip().split("\n\n")
                    formatted_blocks = []
                    for block in blocks:
                        lines = block.strip().split("\n", 1)
                        if len(lines) == 2:
                            first_line = f"<strong>{lines[0]}</strong>"
                            rest = lines[1]
                            formatted_block = f"{first_line}<br>{rest}"
                        else:
                            formatted_block = f"<strong>{lines[0]}</strong>"
                        formatted_blocks.append(formatted_block)
                    
                    # Join blocks using horizontal lines
                    final_html = "<div style='font-size:20px'>" + "<br>".join(formatted_blocks) + "</div>"
                    st.markdown(final_html, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please upload CVs first")
    else:
        if ranking_key in st.session_state:
            st.subheader("🏅 Top Candidates")
            st.markdown(st.session_state[ranking_key])

# ... (keep the rest of the code exactly the same)
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
        # 🔍 Print debug info
        print("🔍 First Name Dict:", st.session_state.first_name_dict)
        print("🔍 Full Name Dict:", st.session_state.full_name_dict)

        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
        if query.strip().lower() in greetings or re.match(r"^(hi|hello|hey)[.!\s]*$", query.strip(), re.IGNORECASE):
            st.session_state.chat_history.append((query, "👋 Hello! How can I assist you with candidate information today?"))
        else:
            candidate_names = extract_matched_names_from_query(
                query,
                st.session_state.full_name_dict,
                st.session_state.first_name_dict
            )

            print("🔎 Matched Names from Query:", candidate_names)

            candidate_key = None
            if candidate_names:
                candidate_key = candidate_names[0].lower()

                if candidate_key != st.session_state.last_candidate_key:
                    doc_path = st.session_state.full_name_dict.get(candidate_key) or \
                               st.session_state.first_name_dict.get(candidate_key)

                    print("📄 Loaded CV Path for Candidate:", doc_path)

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

