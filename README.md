# RAG Document Q&A Chatbot

A chatbot I built that can answer questions about any document you upload (PDF or text file). It uses RAG (Retrieval-Augmented Generation) so the answers are actually grounded in the document content instead of the model just making stuff up.

I built this to learn how RAG pipelines work in practice — chunking documents, embedding them, storing in a vector database, and using an LLM to generate answers from retrieved context.

## What it does

- Upload a PDF or .txt file
- Ask questions about it in a chat interface
- Get answers along with the source chunks used to generate them
- Includes an evaluation script (RAGAS) that scores answer quality on faithfulness, relevancy, precision and recall

## Stack

- **LangChain** for orchestration
- **ChromaDB** as the vector store
- **sentence-transformers (all-MiniLM-L6-v2)** for embeddings — runs locally, no API needed
- **Groq (Llama 3.1)** for the LLM — free tier, fast inference
- **Streamlit** for the UI

## How to run it locally

1. Clone the repo and install dependencies:
```
pip install -r requirements.txt
```

2. Get a free Groq API key from console.groq.com/keys

3. Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

4. Run the app:
```
streamlit run app.py
```

5. Upload a doc (there's a sample one in `sample_docs/`) and start asking questions.

## Evaluation

I ran the chatbot through RAGAS on a set of 5 test questions to check answer quality:

| Metric | Score |
|---|---|
| Faithfulness | 83% |
| Answer Relevancy | 92% |
| Context Precision | 100% |
| Context Recall | 100% |

To run the evaluation yourself:
```
python evaluate.py
```

One thing I noticed while testing — one question scored lower on faithfulness (0.33) because the model added a small detail not directly supported by the retrieved chunk. Worth digging into if you're tuning chunk size or retrieval count (`k`).

## How it works (the short version)

1. Document gets split into ~1000-character chunks
2. Each chunk is converted into a vector using the embedding model
3. Vectors get stored in ChromaDB
4. When you ask a question, it's also converted to a vector and compared against stored chunks to find the most relevant ones
5. Those chunks + your question get sent to the LLM, which generates an answer using only that context

## Notes / things I'd improve next

- Right now it only handles one document at a time — multi-doc support would be a good next step
- No conversation memory yet, so it doesn't remember earlier questions in the same session
- Ran into a couple of rate-limit issues with Groq's free tier during evaluation — added retry logic to handle that

## Project structure
```
rag_chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # core RAG logic (loading, chunking, embedding, retrieval)
├── evaluate.py          # RAGAS evaluation script
├── requirements.txt
├── sample_docs/
│   └── company_policy.txt
└── evaluation_results.csv
```
