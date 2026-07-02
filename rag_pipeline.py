"""
rag_pipeline.py
----------------
Core RAG (Retrieval-Augmented Generation) pipeline logic.

Now supports:
  - Multiple file formats (PDF, TXT, DOCX, CSV)
  - Multiple documents combined into a single searchable index
  - Conversation memory (follow-up questions use chat history)
  - Source relevance scores
"""

import os
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


def load_document(file_path: str):
    """
    Load a single document from disk. Supports PDF, TXT, DOCX, and CSV.
    Each loader returns a list of LangChain Document objects, tagged with
    metadata (including source filename) so we know where each chunk came from.
    """
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
    """
    Load and combine multiple documents into a single list.
    Each chunk keeps metadata about which file it came from, so we can
    cite sources per-file later, even when searching across several docs.
    """
    all_docs = []
    for path in file_paths:
        docs = load_document(path)
        # Tag each doc with a clean display name (just the filename)
        filename = os.path.basename(path)
        for d in docs:
            d.metadata["source_file"] = filename
        all_docs.extend(docs)
    return all_docs


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Split documents into smaller overlapping chunks for embedding."""
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


def build_conversational_chain(vectorstore, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
    """
    Build a CONVERSATIONAL RAG chain that remembers prior questions/answers
    in the same session, so follow-up questions like "what about the second one?"
    work correctly by using chat history as additional context.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Memory stores the conversation so the chain can reformulate follow-up
    # questions using earlier context (e.g. "summarize that" -> knows "that"
    # refers to the previous answer's topic).
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

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    return chain


def get_relevance_scores(vectorstore, query: str, k: int = 4):
    """
    Retrieve chunks WITH similarity scores (0 = identical, higher = less similar
    for distance-based scoring, depending on the embedding metric used).
    We convert to a 0-100% relevance scale for easy display in the UI.
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    # results is a list of (Document, score) tuples
    scored = []
    for doc, score in results:
        scored.append({
            "content": doc.page_content,
            "source_file": doc.metadata.get("source_file", "document"),
            "score": max(0.0, min(1.0, score)),  # clamp to [0, 1]
        })
    return scored


def process_documents_and_create_chain(file_paths: list, groq_api_key: str):
    """
    Full pipeline for MULTIPLE documents:
    load all -> split -> embed & store -> build conversational QA chain.
    Returns the chain, the vectorstore (for relevance scoring), and chunk count.
    """
    documents = load_multiple_documents(file_paths)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    chain = build_conversational_chain(vectorstore, groq_api_key)
    return chain, vectorstore, len(chunks)
