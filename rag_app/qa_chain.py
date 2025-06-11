from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

def create_qa_chain(vector_store):
    print("Creating QA chain with Gemini...")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.4,
        convert_system_message_to_human=True
    )

    template = """You are a helpful assistant that provides accurate information based on the given context.
    Context information is below:
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the question truthfully.
    If the answer cannot be found in the context, say \"I don't have enough information to answer this question.\"
    Keep your response concise and directly address the question.
    Explain your reasoning step by step if the question requires analysis.
    Question: {question}
    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
