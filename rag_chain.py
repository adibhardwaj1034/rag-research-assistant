import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ── 1. Load the saved FAISS index ────────────────────────────────────────
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

# ── 2. Format retrieved chunks into a single string ──────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── 3. Build the RAG chain ───────────────────────────────────────────────
def build_rag_chain():
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI research assistant. Use the context below to answer
    the question clearly and accurately. If the answer is not in the context,
    say "I don't have enough information in my documents to answer that."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ── 4. Ask a question ────────────────────────────────────────────────────
def ask_question(chain, retriever, question):
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    source_names = list(set([
        os.path.basename(doc.metadata.get("source", "Unknown"))
        for doc in source_docs
    ]))
    return answer, source_names