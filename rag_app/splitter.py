from langchain.schema import Document

def split_by_chunk_size(documents, chunk_size=1000):
    all_chunks = []
    candidate_docs = {}

    for doc in documents:
        name = doc.metadata.get("candidate_name", "")
        candidate_docs.setdefault(name, []).append(doc)

    for name, docs in candidate_docs.items():
        full_text = " ".join([doc.page_content for doc in docs])
        source_file = docs[0].metadata.get("source_file", "")
        num_chunks = (len(full_text) + chunk_size - 1) // chunk_size

        for i in range(0, len(full_text), chunk_size):
            chunk = Document(
                page_content=full_text[i:i + chunk_size],
                metadata={
                    "candidate_name": name,
                    "source_file": source_file,
                    "chunk_index": i // chunk_size,
                    "total_chunks": num_chunks
                }
            )
            all_chunks.append(chunk)

        print(f"✅ Split resume for {name} into {num_chunks} chunks")

    print(f"\n📄 Total chunks created: {len(all_chunks)}")
    return all_chunks