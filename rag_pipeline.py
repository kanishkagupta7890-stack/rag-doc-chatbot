"""
rag_pipeline.py
----------------
Core RAG (Retrieval-Augmented Generation) pipeline logic.

WHAT IS RAG?
Instead of relying only on what an LLM "remembers" from training, we:
  1. Break documents into small chunks.
  2. Convert each chunk into a vector (embedding).
  3. Store these vectors in a vector database.
  4. When the user asks a question, embed the question too, and find the
     most similar chunks in the database (retrieval).
  5. Send those relevant chunks + the user's question to an LLM, which
     generates an answer grounded in the actual document content.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_document(file_path: str):
    """Load a document from disk. Supports PDF and plain text files."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Split large documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks, persist_directory: str = "chroma_db"):
    """Embed each chunk and store it in a vector database (Chroma)."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )


def build_qa_chain(vectorstore, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    """Build the full RAG chain: retrieve relevant chunks, then generate an answer."""
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def process_document_and_create_chain(file_path: str, groq_api_key: str):
    """Convenience function: load -> split -> embed & store -> build QA chain."""
    documents = load_document(file_path)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_qa_chain(vectorstore, groq_api_key)
    return qa_chain, len(chunks)
