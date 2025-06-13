import os
from langchain.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredWordDocumentLoader

from langchain.schema import Document

def load_documents(directory_path):
    """Load all PDF, TXT, and DOCX documents from a directory"""
    documents = []
    candidate_names = []

    for filename in os.listdir(directory_path):
        path = os.path.join(directory_path, filename)
        ext = os.path.splitext(filename)[1].lower()

        # Select loader based on file type
        if ext == ".pdf":
            loader = PyMuPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")  # ensure encoding
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(path)
        else:
            continue  # skip unsupported file types

        try:
            # Extract documents
            docs = loader.load()

            # Extract candidate name from filename
            candidate_name = os.path.splitext(filename)[0]
            candidate_names.append(candidate_name)

            # Add metadata
            for doc in docs:
                doc.metadata["candidate_name"] = candidate_name
                doc.metadata["source_file"] = filename

            documents.extend(docs)
            print(f"Loaded resume: {filename}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    print(f"Total documents loaded: {len(documents)}")
    return documents, candidate_names

def load_document(file_path):
    """
    Load a single document (PDF, TXT, or DOCX) and return as LangChain Document(s).
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PDFPlumberLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    try:
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []
