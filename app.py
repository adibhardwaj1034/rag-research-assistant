import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_chain import build_rag_chain, ask_question

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# ── Initialize session state ─────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ── Load default chain once ──────────────────────────────────────────────
@st.cache_resource
def load_chain():
    return build_rag_chain()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 AI Research Assistant")
    st.markdown("---")

    st.markdown("### 📚 Default Knowledge Base")
    st.markdown("""
    - 📄 Attention is All You Need
    - 📄 BERT Paper
    - 📄 GPT-3 Paper
    - 📄 LLaMA 2 Paper
    """)

    st.markdown("---")
    st.markdown("### 📁 Upload Your Own PDF")
    uploaded_file = st.file_uploader(
        "Add a document to the knowledge base",
        type="pdf"
    )

    if uploaded_file is not None:
        with st.spinner("Processing your PDF..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load and chunk the uploaded PDF
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)

            # Add to existing FAISS index
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            existing_db = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            existing_db.add_documents(chunks)
            existing_db.save_local("faiss_index")

            os.unlink(tmp_path)
            st.success(f"✅ '{uploaded_file.name}' added!")
            st.cache_resource.clear()

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with LangChain, Groq & Streamlit")

# ── Main area ────────────────────────────────────────────────────────────
st.title("💬 Chat with AI Research Papers")
st.markdown("Ask anything about Transformers, BERT, GPT-3, LLaMA 2 — or upload your own PDF.")
st.markdown("---")

# Load chain
chain, retriever = load_chain()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.caption(f"📄 Sources: {', '.join(message['sources'])}")

# Chat input
question = st.chat_input("Ask a question about the research papers...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = ask_question(chain, retriever, question)
        st.markdown(answer)
        st.caption(f"📄 Sources: {', '.join(sources)}")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })