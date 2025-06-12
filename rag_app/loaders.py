import os
from tqdm import tqdm
from langchain_community.document_loaders import PyMuPDFLoade  # Unified loader
from langchain.schema import Document

def load_documents(directory_path):
    """
    Load all PDF resumes from a directory using PyMuPDFLoader.

    Args:
        directory_path (str): Path to the directory containing PDF resumes.

    Returns:
        documents (list): List of LangChain Document objects.
        candidate_names (list): List of candidate names derived from filenames.
    """
    documents = []
    candidate_names = []

    for filename in tqdm(os.listdir(directory_path), desc="Loading PDFs"):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(directory_path, filename)
            try:
                loader = PyMuPDFLoader(path)
                docs = loader.load()

                # Sanitize candidate name (remove extension, clean formatting)
                candidate_name = os.path.splitext(filename)[0]
                candidate_name = candidate_name.replace('_', ' ').replace('-', ' ').title()
                candidate_names.append(candidate_name)

                # Add metadata to each document chunk
                for doc in docs:
                    doc.metadata["candidate_name"] = candidate_name
                    doc.metadata["source_file"] = filename

                documents.extend(docs)
                print(f"✅ Loaded resume: {filename}")
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    print(f"\n📄 Total documents loaded: {len(documents)}")
    return documents, candidate_names

def load_document(pdf_path):
    """
    Load a single PDF resume using PyMuPDFLoader.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of LangChain Document objects.
    """
    try:
        loader = PyMuPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        print(f"❌ Failed to load document: {e}")
        return []
