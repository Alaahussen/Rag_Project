def rag_query(query, qa_chain):
    print(f"Query: {query}")
    result = qa_chain.invoke({"query": query})
    print("RAG chain output:", result)

    answer = result.get("answer") or result.get("result") or "⚠️ No answer generated."
    sources = [doc.page_content for doc in result.get("source_documents", [])]

    return {"answer": answer, "sources": sources[0] if sources else ""}
