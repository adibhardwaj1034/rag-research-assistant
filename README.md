# 🤖 AI Research Assistant — RAG Chatbot

A conversational AI chatbot that answers questions from landmark AI research papers using Retrieval-Augmented Generation (RAG). Built with LangChain, Groq (LLaMA 3.1), and Streamlit.

🚀 **[Live Demo](https://rag-research-assistant-uxvrrjyy7gdebxp2srylyx.streamlit.app/)**

---

## 💡 What It Does

- Answer detailed questions about foundational AI research papers
- Cite the exact source document for every answer
- Accept any user-uploaded PDF and instantly chat with it
- Runs entirely on free-tier APIs

---

## 📚 Default Knowledge Base

| Paper | Authors | Year |
|---|---|---|
| Attention is All You Need | Vaswani et al. | 2017 |
| BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | 2018 |
| Language Models are Few-Shot Learners (GPT-3) | Brown et al. | 2020 |
| LLaMA 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | 2023 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain (LCEL) |
| LLM | LLaMA 3.1 via Groq API |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## ⚙️ How It Works
User Question
↓
Embed question using MiniLM
↓
FAISS similarity search → top 4 relevant chunks
↓
Chunks + question sent to LLaMA 3.1 via Groq
↓
Answer generated + source document cited
---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/adibhardwaj1034/rag-research-assistant.git
cd rag-research-assistant
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your Groq API key**

Create a `.env` file:
GROQ_API_KEY=your_key_here
**5. Add your PDFs to the documents/ folder and run ingestion**
```bash
python ingest.py
```

**6. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure
rag-research-assistant/
│
├── app.py              ← Streamlit UI
├── ingest.py           ← PDF ingestion pipeline
├── rag_chain.py        ← RAG chain logic
├── requirements.txt
├── .env                ← API keys (not pushed to GitHub)
│
├── documents/          ← Default PDF knowledge base
└── faiss_index/        ← FAISS vector store
---

## 🔮 Future Improvements

- Add support for multiple file formats (DOCX, TXT, CSV)
- Add conversation memory for follow-up questions
- Add a document management panel to remove uploaded PDFs
- Fine-tune embeddings on domain-specific data

---

## 👤 Author

**Adi Bhardwaj**
[GitHub](https://github.com/adibhardwaj1034)

---

## 📄 License

MIT License