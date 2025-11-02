# streamlit_rag_app.py
# Streamlit RAG app (single-file)
# Features:
# - Upload multiple PDF files
# - Build or load FAISS index using HuggingFace embeddings
# - Ask questions with chat UI (Google Gemini as LLM)
# - Shows retrieved doc previews and citations

import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# LangChain-style imports (matching your existing pipeline names)
from langchain_classic.document_loaders import PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate

load_dotenv()

# -----------------------
# Helper functions
# -----------------------


@st.cache_data
def save_uploaded_files(uploaded_files) -> List[Path]:
    """Save uploaded files to a temporary directory and return file paths."""
    tmpdir = Path(tempfile.mkdtemp(prefix="streamlit_rag_"))
    paths = []
    for ul in uploaded_files:
        out_path = tmpdir / ul.name
        with open(out_path, "wb") as f:
            f.write(ul.getbuffer())
        paths.append(out_path)
    return paths


def process_all_pdfs(file_paths: List[Path]):
    """Load all PDFs and return list of langchain-style Document objects."""
    all_documents = []
    for file in file_paths:
        try:
            loader = PyMuPDFLoader(str(file))
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")
    return all_documents


@st.cache_data
def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


@st.cache_resource
def init_embeddings(model_name: str = "BAAI/bge-small-en-v1.5"):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        show_progress=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )


@st.cache_resource
def create_faiss_from_docs(chunks, embedding_model):
    vs = FAISS.from_documents(
        documents=chunks, embedding=embedding_model, distance_strategy="COSINE")
    # persist to a temp folder so the user can reuse during session
    tmp = Path(tempfile.mkdtemp(prefix="faiss_index_"))
    vs.save_local(str(tmp))
    return vs, str(tmp)


@st.cache_resource
def load_faiss(embedding_model_name: str, index_path: str):
    embeddings = init_embeddings(embedding_model_name)
    vs = FAISS.load_local(index_path, embeddings=embeddings,
                          allow_dangerous_deserialization=True)
    return vs


def simple_rag(query: str, vectorstore, k: int = 3):
    similar_docs = vectorstore.similarity_search(query=query, k=k)

    # Use Google Gemini model (user must set proper credentials)
    chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    context = "\n\n".join([doc.page_content for doc in similar_docs])

    prompt = ChatPromptTemplate.from_template(
        """
Using the following context to answer the question below.
If the context is insufficient, please say "I don't know".

<context>
{context}
</context>

question: {query}
"""
    )

    prompt = prompt.format_prompt(context=context, query=query)

    # call the model
    response = chat_model.invoke(prompt)

    return response.content, similar_docs


# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Simple RAG (Streamlit)", layout="wide")
st.title("Simple RAG — Upload PDFs & Chat")

with st.sidebar:
    st.header("Index / Model settings")
    embedding_model_name = st.text_input(
        "Embedding model", value="BAAI/bge-small-en-v1.5")
    chunk_size = st.number_input(
        "Chunk size", min_value=256, max_value=5000, value=1000, step=128)
    chunk_overlap = st.number_input(
        "Chunk overlap", min_value=0, max_value=500, value=200, step=50)
    k = st.number_input("Retriever k (docs)", min_value=1,
                        max_value=10, value=3)
    st.markdown("---")
    st.markdown(
        "**Tip:** set your Google GenAI credentials in env before running (e.g. GOOGLE_API_KEY or Google SDK).")

# file upload area
uploaded_files = st.file_uploader("Upload PDF files", type=[
                                  "pdf"], accept_multiple_files=True)

col1, col2 = st.columns([2, 3])

with col1:
    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")
        if st.button("Save & Process uploads"):
            saved_paths = save_uploaded_files(uploaded_files)
            docs = process_all_pdfs(saved_paths)
            if not docs:
                st.warning("No documents loaded from uploads.")
            else:
                st.session_state["docs"] = docs
                st.success(f"Loaded {len(docs)} pages from uploaded PDFs")

    # Option to build index from session docs
    if st.button("Build FAISS index from loaded docs"):
        docs = st.session_state.get("docs")
        if not docs:
            st.warning("No docs available. Upload & process PDFs first.")
        else:
            with st.spinner("Splitting documents into chunks..."):
                chunks = split_docs(docs, chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap)
                st.session_state["chunks_count"] = len(chunks)
            st.info(f"Chunks created: {len(chunks)}")
            with st.spinner("Creating embeddings and FAISS index (this may take time)..."):
                emb = init_embeddings(embedding_model_name)
                vs, idx_path = create_faiss_from_docs(chunks, emb)
                st.session_state["vectorstore"] = vs
                st.session_state["index_path"] = idx_path
                st.success("FAISS index created and saved for this session")

    # Option to load existing index from local path
    index_dir = st.text_input(
        "Or enter existing FAISS index path to load", value="")
    if st.button("Load FAISS index"):
        if not index_dir:
            st.warning("Provide index path to load")
        else:
            try:
                vs = load_faiss(embedding_model_name, index_dir)
                st.session_state["vectorstore"] = vs
                st.success("Loaded FAISS index successfully")
            except Exception as e:
                st.error(f"Could not load index: {e}")

    st.markdown("---")
    st.write("Session info:")
    st.write({
        "docs_loaded": len(st.session_state.get("docs", [])),
        "chunks": st.session_state.get("chunks_count", 0),
        "index_path": st.session_state.get("index_path", "(none)"),
    })

with col2:
    st.subheader("Chat / Ask")

    if "vectorstore" not in st.session_state:
        st.info(
            "No vectorstore found. Upload files and build or load a FAISS index first.")
    else:
        query = st.text_input("Enter your question or instruction:")
        if st.button("Ask") and query:
            vs = st.session_state["vectorstore"]
            with st.spinner("Retrieving relevant docs and generating answer..."):
                try:
                    answer, retrieved = simple_rag(query, vs, k=k)

                    # show answer
                    st.markdown("**Answer:**")
                    st.write(answer)

                    # show retrieved previews
                    st.markdown("**Retrieved document previews:**")
                    for i, doc in enumerate(retrieved, 1):
                        st.markdown(f"**Doc {i} (preview):**")
                        text_preview = doc.page_content[:500].strip(
                        ) + ("..." if len(doc.page_content) > 500 else "")
                        st.write(text_preview)

                    # append to chat history
                    history = st.session_state.get("chat_history", [])
                    history.append({"query": query, "answer": answer})
                    st.session_state["chat_history"] = history

                except Exception as e:
                    st.error(f"Error during RAG query: {e}")

    # show past Q&A
    if st.session_state.get("chat_history"):
        st.markdown("---")
        st.markdown("**Chat history**")
        for item in reversed(st.session_state.get("chat_history", [])):
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['answer']}")


# -----------------------
# Footer / run instructions
# -----------------------

st.markdown("---")
st.caption("Run this app: `streamlit run streamlit_rag_app.py`.\nMake sure environment variables/credentials for Google GenAI are set before running.")
