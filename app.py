"""
app.py
------
Streamlit web interface for the RAG chatbot.

Features:
  - Multiple document upload (PDF, TXT, DOCX, CSV)
  - Conversation memory (follow-up questions work naturally)
  - Source relevance scores shown per answer
  - Clear chat button
  - Example questions to try
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import process_documents_and_create_chain, get_relevance_scores

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="Doc Q&A Chatbot (RAG)", page_icon="📄", layout="centered")

st.title("📄 Document Q&A Chatbot")
st.caption("Upload one or more documents (PDF, TXT, DOCX, CSV), then ask questions. "
           "Built with RAG (Retrieval-Augmented Generation) and remembers your conversation.")

# ---- Sidebar: API key + file upload ----
with st.sidebar:
    st.header("Setup")

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            groq_api_key = ""

    if not groq_api_key:
        groq_api_key = st.text_input("Enter your Groq API key", type="password")
        st.markdown("[Get a free Groq API key](https://console.groq.com/keys)")

    uploaded_files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "docx", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Process Document(s)", type="primary"):
        if not groq_api_key:
            st.error("Please provide a Groq API key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner(f"Reading, chunking, and embedding {len(uploaded_files)} document(s)..."):
                tmp_paths = []
                try:
                    for uploaded_file in uploaded_files:
                        suffix = "." + uploaded_file.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_paths.append(tmp.name)

                    chain, vectorstore, num_chunks = process_documents_and_create_chain(
                        tmp_paths, groq_api_key
                    )
                    st.session_state.chain = chain
                    st.session_state.vectorstore = vectorstore
                    st.session_state.messages = []
                    st.session_state.doc_names = [f.name for f in uploaded_files]
                    st.success(f"Processed {len(uploaded_files)} document(s) into {num_chunks} chunks. Ask away!")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                finally:
                    for p in tmp_paths:
                        try:
                            os.remove(p)
                        except OSError:
                            pass

    # Clear chat button — resets conversation but keeps the processed documents
    if "chain" in st.session_state:
        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            # Reset the chain's memory too, so old context doesn't leak into new chat
            if hasattr(st.session_state.chain, "memory"):
                st.session_state.chain.memory.clear()
            st.rerun()

        if st.session_state.get("doc_names"):
            st.caption("📂 Loaded documents:")
            for name in st.session_state.doc_names:
                st.caption(f"• {name}")

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Documents are split into chunks\n"
        "2. Chunks are embedded into vectors\n"
        "3. Your question retrieves the most relevant chunks\n"
        "4. An LLM answers using only those chunks\n"
        "5. Conversation history is remembered for follow-ups"
    )

# ---- Main chat area ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# Example questions to help users get started (only show before first message)
if "chain" in st.session_state and not st.session_state.messages:
    st.markdown("**Try asking:**")
    example_cols = st.columns(2)
    examples = [
        "What are the main topics covered?",
        "Summarize this document in 3 sentences",
    ]
    for i, example in enumerate(examples):
        if example_cols[i % 2].button(example, key=f"example_{i}"):
            st.session_state.pending_question = example
            st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources & relevance"):
                for src in message["sources"]:
                    relevance_pct = int(src["score"] * 100)
                    st.markdown(f"**{src['source_file']}** — relevance: {relevance_pct}%")
                    st.progress(src["score"])
                    st.text(src["content"][:400] + ("..." if len(src["content"]) > 400 else ""))
                    st.markdown("---")

# Handle example question clicks or normal chat input
prompt = st.chat_input("Ask a question about your document(s)...")
if "pending_question" in st.session_state:
    prompt = st.session_state.pop("pending_question")

if prompt:
    if "chain" not in st.session_state:
        st.warning("Please upload and process a document first (see sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # The conversational chain automatically uses memory for follow-ups
                result = st.session_state.chain.invoke({"question": prompt})
                answer = result["answer"]

                # Get relevance-scored sources separately for richer display
                scored_sources = get_relevance_scores(st.session_state.vectorstore, prompt)

                st.markdown(answer)
                if scored_sources:
                    with st.expander("📚 Sources & relevance"):
                        for src in scored_sources:
                            relevance_pct = int(src["score"] * 100)
                            st.markdown(f"**{src['source_file']}** — relevance: {relevance_pct}%")
                            st.progress(src["score"])
                            st.text(src["content"][:400] + ("..." if len(src["content"]) > 400 else ""))
                            st.markdown("---")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": scored_sources}
        )
