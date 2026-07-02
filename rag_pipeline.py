"""
rag_pipeline.py
----------------
Core RAG pipeline. Vector store is always rebuilt fresh on each
document upload — no stale data from previous sessions.
"""

import os
import shutil
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

CHROMA_DIR = "chroma_db"


def clear_vectorstore():
    """Delete the existing vector store so old documents never bleed into new sessions."""
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)


def load_document(file_path: str):
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif lower.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    elif lower.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    return docs


def load_multiple_documents(file_paths: list):
    all_docs = []
    for path in file_paths:
        docs = load_document(path)
        filename = os.path.basename(path)
        for d in docs:
            d.metadata["source_file"] = filename
        all_docs.extend(docs)
    return all_docs


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Always builds a fresh vector store — clears any previous data first."""
    clear_vectorstore()
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
    )


def build_conversational_chain(vectorstore, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    qa_prompt = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say you don't know -
do not try to make up an answer.

Context:
{context}

Question: {question}

Helpful Answer:""",
        input_variables=["context", "question"],
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )


def get_relevance_scores(vectorstore, query: str, k: int = 4):
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    scored = []
    for doc, score in results:
        scored.append({
            "content": doc.page_content,
            "source_file": doc.metadata.get("source_file", "document"),
            "score": max(0.0, min(1.0, score)),
        })
    return scored


def process_documents_and_create_chain(file_paths: list, groq_api_key: str):
    """
    Full pipeline: always starts fresh (clears old vector store),
    loads all docs, splits, embeds, builds conversational QA chain.
    """
    documents = load_multiple_documents(file_paths)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)  # clears old data automatically
    chain = build_conversational_chain(vectorstore, groq_api_key)
    return chain, vectorstore, len(chunks)
