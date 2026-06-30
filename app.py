"""
app.py
------
This is the web interface for our RAG chatbot, built with Streamlit.

Streamlit turns a Python script into a web app with minimal code -
great for quickly building demos for AI/ML projects.

HOW TO RUN:
    streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import process_document_and_create_chain

# Load environment variables (your GROQ_API_KEY) from a .env file
load_dotenv()

st.set_page_config(page_title="Doc Q&A Chatbot (RAG)", page_icon="📄", layout="centered")

st.title("📄 Document Q&A Chatbot")
st.caption("Upload a PDF or text file, then ask questions about its content. "
           "Built with RAG (Retrieval-Augmented Generation).")

# ---- Sidebar: API key + file upload ----
with st.sidebar:
    st.header("Setup")

    # Allow user to paste their own Groq API key if not set in .env
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("Enter your Groq API key", type="password")
        st.markdown("[Get a free Groq API key](https://console.groq.com/keys)")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

    if st.button("Process Document", type="primary"):
        if not groq_api_key:
            st.error("Please provide a Groq API key.")
        elif not uploaded_file:
            st.error("Please upload a document first.")
        else:
            with st.spinner("Reading, chunking, and embedding your document..."):
                # Save uploaded file to a temporary path so LangChain loaders can read it
                suffix = "." + uploaded_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    qa_chain, num_chunks = process_document_and_create_chain(
                        tmp_path, groq_api_key
                    )
                    st.session_state.qa_chain = qa_chain
                    st.session_state.messages = []
                    st.success(f"Document processed into {num_chunks} chunks. Ask away!")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                finally:
                    os.remove(tmp_path)

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Document is split into chunks\n"
        "2. Chunks are embedded into vectors\n"
        "3. Your question retrieves the most relevant chunks\n"
        "4. An LLM answers using only those chunks"
    )

# ---- Main chat area ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources used"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(src[:500] + ("..." if len(src) > 500 else ""))

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if "qa_chain" not in st.session_state:
        st.warning("Please upload and process a document first (see sidebar).")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer from the RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"query": prompt})
                answer = result["result"]
                sources = [doc.page_content for doc in result["source_documents"]]

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources used"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(src[:500] + ("..." if len(src) > 500 else ""))

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
