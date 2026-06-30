# 📄 Document Q&A Chatbot (RAG-based)

A chatbot that answers questions about your documents (PDF or text) using
**Retrieval-Augmented Generation (RAG)** — a core technique behind real-world
production GenAI applications like enterprise search, customer support bots,
and internal knowledge assistants.

---

## 🚀 Quick Setup (15 minutes)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Groq provides free, very fast access to open-source LLMs (Llama 3).
1. Go to https://console.groq.com/keys
2. Sign up (no credit card needed)
3. Create an API key

### 3. Set up your API key
Copy `.env.example` to `.env` and paste your key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`

### 5. Test it
- Upload `sample_docs/company_policy.txt` (included in this project)
- Click "Process Document"
- Ask: *"How many vacation days do employees get per year?"*
- Ask: *"Can I work remotely every day?"*

---

## 🏗️ How It Works (Architecture)

```
User uploads PDF/TXT
        │
        ▼
┌───────────────────┐
│  1. Document Load  │  (PyPDFLoader / TextLoader)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  2. Text Splitting │  (RecursiveCharacterTextSplitter)
│  Break into ~1000  │
│  char chunks       │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  3. Embedding      │  (sentence-transformers/all-MiniLM-L6-v2)
│  Convert chunks to │  Runs locally, free, no API key
│  vectors           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  4. Vector Store   │  (ChromaDB)
│  Store & index     │
│  vectors           │
└───────────────────┘
        │
   User asks question
        │
        ▼
┌───────────────────┐
│  5. Retrieval      │  Embed question → find top-3
│                    │  most similar chunks
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  6. Generation     │  (Groq + Llama 3.1)
│  LLM answers using │
│  retrieved chunks  │
└───────────────────┘
        │
        ▼
   Answer + Sources
```

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|---|---|---|
| Frontend | Streamlit | Fast way to build AI demos |
| Orchestration | LangChain | Industry-standard GenAI framework |
| Embeddings | sentence-transformers (MiniLM) | Free, local, fast |
| Vector DB | ChromaDB | Simple, embeddable, widely used |
| LLM | Groq (Llama 3.1) | Free tier, extremely fast inference |

---

## 📈 Ways to Extend This (do these to make it stand out further)

Pick 1-2 of these to add depth — each one gives you more to talk about in interviews:

1. **Add evaluation metrics** — Create a small set of Q&A pairs and measure
   answer accuracy/relevance. Mention this metric on your resume.
2. **Support multiple documents** — Let users upload several files and search across all of them.
3. **Add conversation memory** — Use `ConversationalRetrievalChain` so the bot remembers previous questions.
4. **Deploy it** — Push to [Streamlit Community Cloud](https://streamlit.io/cloud) (free) so you have a live demo link.
5. **Swap the vector DB** — Try Pinecone or Weaviate to show you understand alternatives.
6. **Add a "confidence score"** — Show similarity scores for retrieved chunks.

---

## 📝 How to Describe This on Your Resume

> **Document Q&A Assistant (RAG-based)** — Built a retrieval-augmented
> generation chatbot using LangChain, ChromaDB, and Llama 3 (via Groq) that
> answers questions over user-uploaded documents with cited sources.
> Implemented document chunking, semantic embedding with sentence-transformers,
> and a custom prompt-engineered QA chain. Deployed via Streamlit. [GitHub link] [Live demo link]

**Skills to list:** Python, LangChain, RAG, Vector Databases, LLMs, Prompt Engineering, Streamlit

---

## 🎤 Interview Prep — Be Ready to Explain:

- **What is RAG and why use it instead of just an LLM?**
  → Reduces hallucination, allows answering questions about private/recent
  data the LLM wasn't trained on, and provides source citations.

- **Why chunk documents? Why overlap?**
  → Embedding models have size limits; smaller chunks = more precise
  retrieval. Overlap prevents losing context at chunk boundaries.

- **What's an embedding?**
  → A numerical vector representation of text where semantically similar
  text has vectors that are close together (measured via cosine similarity).

- **What's the difference between this and fine-tuning?**
  → Fine-tuning changes the model's weights (expensive, static).
  RAG keeps the model frozen and injects relevant info at query time
  (cheap, always up-to-date, easy to update by just adding documents).

- **How would you scale this to production?**
  → Swap Chroma for a managed vector DB (Pinecone/Weaviate), add caching,
  add rate limiting, monitor retrieval quality, add user authentication.

---

## ⚠️ Troubleshooting

- **"No space left" / slow first run**: The embedding model (~80MB) downloads
  on first use. Make sure you have internet access and a bit of disk space.
- **Groq errors**: Double-check your API key is correctly pasted in `.env`
  with no extra spaces.
- **PDF not loading**: Make sure the PDF isn't a scanned image (no extractable text).
