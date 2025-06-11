import os
from langchain.document_loaders import PyMuPDFLoader, PDFPlumberLoader
from langchain.schema import Document

def load_documents(directory_path):
    documents = []
    candidate_names = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            path = os.path.join(directory_path, filename)
            loader = PyMuPDFLoader(path)
            docs = loader.load()

            candidate_name = os.path.splitext(filename)[0]
            candidate_names.append(candidate_name)

            for doc in docs:
                doc.metadata["candidate_name"] = candidate_name
                doc.metadata["source_file"] = filename

            documents.extend(docs)
            print(f"Loaded resume: {filename}")

    print(f"Total documents loaded: {len(documents)}")
    return documents, candidate_names

def load_document(pdf_path):
    loader = PDFPlumberLoader(pdf_path)
    return loader.load()