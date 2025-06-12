import os
import re
import spacy
import shutil
import warnings
from pathlib import Path
from collections import Counter
import tempfile


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
nlp = spacy.load("en_core_web_sm")
def rank_candidates(criteria, vectorstore, candidate_names, top_n=10, k_per_candidate=3):
    """
    Rank top N candidates based on criteria and return those who meet the score threshold.
    Handles both general info (e.g., GPA) and skill/job-related queries differently.
    """

    print(f"\n🔍 Identifying top {top_n} candidates matching: '{criteria}'")

    # Step 1: Similarity search
    all_docs = vectorstore.similarity_search(criteria, k=30)

    # Step 2: Count matching chunks
    candidate_counter = Counter()

    for doc in all_docs:
        candidate = doc.metadata.get("candidate_name")
        if candidate in candidate_names:
            candidate_counter[candidate] += 1

    # Step 3: Get top N candidates by frequency
    top_candidates = [name for name, _ in candidate_counter.most_common(top_n)]

    evaluations = []

    # Determine threshold
    general_keywords = ["gpa", "grade", "university", "college", "city", "location", "school","from","faculty","graduated"]
    lower_criteria = criteria.lower()
    is_general = any(keyword in lower_criteria for keyword in general_keywords)
    threshold = 8 if is_general else 6

    # Step 4: Evaluate candidates
    for candidate in top_candidates:
        print(f"⚙️ Evaluating: {candidate}")
        #out[candidate] = folder_path + "/" + candidate + ".pdf"
        candidate_docs = [doc for doc in all_docs if doc.metadata.get("candidate_name") == candidate][:k_per_candidate]

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

        # Parse the score from LLM output
        match = re.search(r"SCORE:\s*(\d+)", evaluation)
        if match:
            score = int(match.group(1))
            if score >= threshold:
                evaluations.append((score, evaluation))

    # Step 5: If no candidates pass threshold
    if not evaluations:
        return f"No candidates scored above the threshold of {threshold} for: '{criteria}'"

    # Step 6: Rank passed candidates
    evaluations.sort(reverse=True, key=lambda x: x[0])  # Sort by score descending
    all_evaluations = "\n\n".join([e[1] for e in evaluations])

    ranking_prompt = f"""
You are an expert HR assistant summarizing the top candidates for a position requiring:
{criteria}

Based on the evaluations below, summarize *why* these candidates are the best fit and rank them.

Evaluations:
{all_evaluations}

Format your response as:
1. Candidate Name (Score): Summary of why ranked here
2. ...
"""
    ranking = generate_response(ranking_prompt)
    return ranking

st.set_page_config(page_title="CV Ranker", layout="wide")
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

                    # Use the rank_candidates function
                    ranking = rank_candidates(criteria, vector_store, candidate_names, top_n=10, k_per_candidate=3)
                    
                    # Store the ranking result
                    st.session_state[ranking_key] = ranking
                    ranked_candidates = []
                    if isinstance(ranking, str):  
                        lines = ranking.split('\n')
                        for line in lines:
                            if line.strip() and line[0].isdigit():  
                                candidate_part = line.split('.')[1].split('(')[0].strip()
                                ranked_candidates.append(candidate_part)

                    candidates_to_process = ranked_candidates if ranked_candidates else candidate_names
                    
                    for candidate in candidates_to_process:
                        clean_candidate = candidate.replace('**', '')
                        for filename in os.listdir(upload_dir):
                            if filename.lower().endswith(".pdf") and clean_candidate in filename.lower():
                                cv_path=os.path.join(upload_dir, filename)
                                st.session_state.out[clean_candidate] = cv_path
                                name_parts = extract_name_spacy(clean_candidate).lower().split()
                                if name_parts:
                                    st.session_state.first_name_dict[name_parts[0]] = cv_path
                                    if len(name_parts) > 1:
                                        full_name = " ".join(name_parts[:2])
                                        st.session_state.full_name_dict[full_name] = cv_path
                    
                    st.subheader("🏅 Top Candidates")
                    st.markdown(ranking)
                    
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

