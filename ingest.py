import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── 1. Load all PDFs from the documents folder ──────────────────────────
def load_documents(folder_path="documents"):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            all_docs.extend(loader.load())
    print(f"\n✅ Total pages loaded: {len(all_docs)}")
    return all_docs

# ── 2. Split into chunks ─────────────────────────────────────────────────
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

# ── 3. Create embeddings and save FAISS index ────────────────────────────
def create_vector_store(chunks):
    print("\n⏳ Creating embeddings (this takes 2-3 minutes the first time)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    print("✅ FAISS index saved to faiss_index/")

# ── 4. Run everything ────────────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    create_vector_store(chunks)
    print("\n🎉 Ingestion complete! You can now run the app.")