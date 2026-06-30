"""
rag_pipeline.py
----------------
This file contains the "brain" of our RAG (Retrieval-Augmented Generation) chatbot.

WHAT IS RAG? (Explain this in interviews!)
RAG = Retrieval-Augmented Generation.
Instead of relying only on what an LLM "remembers" from training, we:
  1. Take the user's documents and break them into small chunks.
  2. Convert each chunk into a vector (a list of numbers representing meaning) - this is "embedding".
  3. Store these vectors in a vector database.
  4. When the user asks a question, we embed the question too, and find the
     most SIMILAR chunks in the database (this is "retrieval").
  5. We send those relevant chunks + the user's question to an LLM, which
     generates an answer GROUNDED in the actual document content.

This avoids hallucination and lets the LLM answer questions about documents
it has never seen during training.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_document(file_path: str):
    """
    Step 1: Load a document from disk.
    Supports PDF and plain text files.
    Returns a list of LangChain "Document" objects (raw text + metadata).
    """
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()
    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Step 2: Split large documents into smaller overlapping chunks.

    WHY CHUNK?
    - Embedding models have a max input size.
    - Smaller chunks = more precise retrieval (we find the exact paragraph
      that answers the question, not a whole 50-page document).

    WHY OVERLAP?
    - Important context near chunk boundaries doesn't get cut off awkwardly.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks, persist_directory: str = "chroma_db"):
    """
    Step 3 & 4: Embed each chunk and store it in a vector database (Chroma).

    EMBEDDINGS:
    We use 'all-MiniLM-L6-v2', a small, fast, FREE model that runs locally
    (no API key needed). It converts text into a 384-dimensional vector
    that captures the semantic meaning of the text.

    VECTOR STORE:
    Chroma stores these vectors and lets us do fast "similarity search" -
    finding which stored chunks are closest in meaning to a query.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )
    return vectorstore


def build_qa_chain(vectorstore, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    """
    Step 5: Build the full RAG chain that:
      - Takes a user question
      - Retrieves the top-k most relevant chunks from the vector store
      - Sends question + chunks to the LLM (via Groq, which is FREE and very fast)
      - Returns an answer + the source chunks used (for citation)

    'k=3' means we retrieve the 3 most relevant chunks. You can tune this -
    more chunks = more context but more chance of irrelevant info confusing the model.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,  # 0 = more focused/deterministic answers, less "creative"
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Custom prompt: tells the LLM HOW to use the retrieved context.
    # This is "prompt engineering" - a key skill to mention in interviews.
    prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say you don't know -
do not try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" = put all retrieved chunks directly into the prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


def process_document_and_create_chain(file_path: str, groq_api_key: str):
    """
    Convenience function that runs the full pipeline end-to-end:
    load -> split -> embed & store -> build QA chain.
    """
    documents = load_document(file_path)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore, groq_api_key)
    return qa_chain, len(chunks)
