import streamlit as st
from pathlib import Path
import tempfile
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate
# from langchain.callbacks.base import BaseCallbackHandler
from langchain_classic.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

# ==================== STREAMLIT PAGE CONFIG ====================
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STREAMING CALLBACK ====================


class StreamHandler(BaseCallbackHandler):
    """Handles streaming tokens to Streamlit UI"""

    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# ==================== HELPER FUNCTIONS ====================


def process_uploaded_pdfs(uploaded_files):
    """Process uploaded PDF files and return documents"""
    all_documents = []

    with st.spinner("📄 Processing uploaded PDFs..."):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = Path(temp_dir) / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    loader = PyMuPDFLoader(str(temp_path))
                    documents = loader.load()
                    all_documents.extend(documents)
                    st.success(
                        f"✅ Loaded {len(documents)} pages from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    return all_documents


def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vectorstore(chunks):
    """Create FAISS vectorstore from chunks"""
    with st.spinner("🔄 Creating embeddings and vectorstore..."):
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'batch_size': 32,
                    'normalize_embeddings': True
                }
            )

            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=embeddings,
                distance_strategy='COSINE'
            )

            st.success(
                f"✅ Vectorstore created with {vectorstore.index.ntotal} vectors")
            return vectorstore, embeddings

        except Exception as e:
            st.error(f"❌ Error creating vectorstore: {e}")
            return None, None


def load_existing_vectorstore(vectorstore_path, embedding_model):
    """Load existing FAISS vectorstore"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'batch_size': 32,
                'normalize_embeddings': True
            }
        )

        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore, embeddings

    except Exception as e:
        st.error(f"❌ Error loading vectorstore: {e}")
        return None, None


def get_rag_response(query, vectorstore, stream_container=None):
    """Generate RAG response for query"""
    try:
        # Retrieve similar documents
        similar_docs = vectorstore.similarity_search(query, k=3)

        # Create context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in similar_docs])

        # Create prompt
        prompt = ChatPromptTemplate.from_template(
            '''Using the following context to answer the question below. 
            If the context is insufficient, please say "I don't know based on the provided documents".
            
            <context>
            {context}
            </context>
            
            Question: {query}
            
            Answer:'''
        )

        formatted_prompt = prompt.format_prompt(
            context=context,
            query=query
        )

        # Initialize LLM with streaming
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            streaming=True
        )

        # Generate response
        if stream_container:
            stream_handler = StreamHandler(stream_container)
            response = chat_model.invoke(
                formatted_prompt.to_messages(),
                config={"callbacks": [stream_handler]}
            )
            return response.content
        else:
            response = chat_model.invoke(formatted_prompt.to_messages())
            return response.content

    except Exception as e:
        return f"❌ Error generating response: {e}"

# ==================== INITIALIZE SESSION STATE ====================


def initialize_session_state():
    """Initialize all session state variables"""
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

# ==================== MAIN APP ====================


def main():
    initialize_session_state()

    # App header
    st.title("📚 RAG Chat Assistant")
    st.markdown("Upload PDFs and chat with your documents using AI!")

    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )

        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                # Process PDFs
                documents = process_uploaded_pdfs(uploaded_files)

                if documents:
                    # Split into chunks
                    chunks = split_documents(documents)
                    st.info(
                        f"📑 Created {len(chunks)} chunks from {len(documents)} pages")

                    # Create vectorstore
                    vectorstore, embeddings = create_vectorstore(chunks)

                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.embeddings = embeddings
                        st.session_state.processed_files = [
                            f.name for f in uploaded_files]
                        st.session_state.messages = []  # Reset chat
                        st.rerun()

        # Show processed files
        if st.session_state.processed_files:
            st.success("✅ Documents Ready!")
            st.write("**Processed Files:**")
            for filename in st.session_state.processed_files:
                st.write(f"- {filename}")

            # Clear button
            if st.button("🗑️ Clear All Documents"):
                st.session_state.vectorstore = None
                st.session_state.embeddings = None
                st.session_state.messages = []
                st.session_state.processed_files = []
                st.rerun()

        # Option to load existing vectorstore
        st.divider()
        st.header("💾 Load Existing Index")

        if st.button("Load FAISS Index from Disk"):
            if os.path.exists("faiss_index"):
                vectorstore, embeddings = load_existing_vectorstore(
                    "faiss_index",
                    "BAAI/bge-small-en-v1.5"
                )
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.embeddings = embeddings
                    st.success("✅ Loaded existing vectorstore!")
                    st.rerun()
            else:
                st.warning("⚠️ No saved index found at 'faiss_index'")

        # Save current vectorstore
        if st.session_state.vectorstore is not None:
            if st.button("💾 Save Current Index"):
                try:
                    st.session_state.vectorstore.save_local("faiss_index")
                    st.success("✅ Vectorstore saved to 'faiss_index'")
                except Exception as e:
                    st.error(f"❌ Error saving: {e}")

    # Main chat interface
    if st.session_state.vectorstore is None:
        st.info(
            "👈 Please upload and process documents from the sidebar to start chatting!")
        st.markdown("""
        ### How to use:
        1. Upload one or more PDF files using the sidebar
        2. Click "Process Documents" to create the knowledge base
        3. Start asking questions in the chat!
        
        ### Features:
        - 📄 Multi-PDF support
        - 🔍 Semantic search with FAISS
        - 💬 Context-aware responses
        - 💾 Save/load vector indices
        """)
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append(
                {"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                response = get_rag_response(
                    prompt,
                    st.session_state.vectorstore,
                    stream_container=response_placeholder
                )

            # Add assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
