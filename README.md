# Simple RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain that processes PDF documents, creates embeddings, stores them in a FAISS vector database, and enables intelligent question-answering using Google's Gemini AI.

## 🚀 Features

- **PDF Processing**: Batch process multiple PDF files from a directory
- **Smart Text Chunking**: Split documents into manageable chunks with overlap
- **Vector Embeddings**: Create embeddings using HuggingFace's BGE model
- **Vector Storage**: Efficient storage and retrieval using FAISS
- **Flexible Operations**:
  - Create new vectorstore from scratch
  - Load existing vectorstore
  - Add new documents to existing vectorstore
- **AI-Powered Q&A**: Query your documents using Google Gemini AI

## 📋 Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemini AI)
- Internet connection (for downloading embedding models)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/codewithyasho/simple-rag-pipeline.git
cd simple-rag-pipeline
```

2. **Install required packages**
```bash
pip install langchain-classic
pip install langchain-huggingface
pip install langchain-google-genai
pip install pymupdf
pip install faiss-cpu
pip install python-dotenv
```

3. **Set up environment variables**

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 📁 Project Structure

```
projects11-rag-pipeline/
├── main.py                 # Main pipeline script
├── main.ipynb             # Jupyter notebook version
├── main copy.ipynb        # Backup notebook
├── README.md              # Project documentation
├── .env                   # Environment variables (create this)
├── data/
│   ├── pdf_files/        # PDF storage directory
│   └── pdfs/             # Additional PDF storage
└── faiss_index/          # FAISS vector database
    └── index.faiss       # Vector index file
```

## 🎯 Usage

### Option 1: Create New Vectorstore

Process PDFs and create a new FAISS index:

```python
# Step 1: Process all PDFs
documents = process_all_pdfs("data/pdfs")

# Step 2: Split documents into chunks
chunked_documents = split_docs(documents)

# Step 3: Create vectorstore
vectorstore = embed_and_store(chunked_documents)

# Step 4: Query the documents
query = "What is the main topic discussed in the documents?"
answer = simple_rag(query, vectorstore)
print(answer)
```

### Option 2: Load Existing Vectorstore

Load a previously created vectorstore:

```python
# Load existing vectorstore
vectorstore = embed_and_load(
    embedding_model="BAAI/bge-small-en-v1.5",
    vectorstore_path="faiss_index"
)

# Query the documents
query = "Your question here"
answer = simple_rag(query, vectorstore)
print(answer)
```

### Option 3: Add Documents to Existing Vectorstore

Add new PDFs to an existing vectorstore:

```python
# Process new PDFs
new_documents = process_all_pdfs("data/new_pdfs")
new_chunks = split_docs(new_documents)

# Load and update vectorstore
vectorstore = load_and_add_new_docs(
    embedding_model="BAAI/bge-small-en-v1.5",
    vectorstore_path="faiss_index",
    new_chunks=new_chunks
)

# Query the updated vectorstore
query = "Your question here"
answer = simple_rag(query, vectorstore)
print(answer)
```

## 🔧 Configuration

### Text Splitting Parameters

Adjust in `split_docs()` function:
```python
chunk_size=1000      # Size of each chunk
chunk_overlap=200    # Overlap between chunks
```

### Embedding Model

Change the embedding model:
```python
embedding_model = "BAAI/bge-small-en-v1.5"  # Default
# Other options: "sentence-transformers/all-MiniLM-L6-v2", etc.
```

### Similarity Search

Adjust number of documents retrieved in `simple_rag()`:
```python
k=3  # Number of similar documents to retrieve
```

### AI Model

Change the Gemini model in `simple_rag()`:
```python
model="gemini-2.5-pro"  # Change to gemini-1.5-pro or gemini-pro
```

## 📊 Functions Overview

| Function | Description |
|----------|-------------|
| `process_all_pdfs(directory)` | Load all PDF files from a directory |
| `split_docs(documents)` | Split documents into chunks |
| `embed_and_store(chunks)` | Create embeddings and new vectorstore |
| `embed_and_load(model, path)` | Load existing vectorstore |
| `load_and_add_new_docs(model, path, chunks)` | Add documents to existing vectorstore |
| `simple_rag(query, vectorstore)` | Query the vectorstore and get AI response |

## 💡 Key Components

### 1. Document Loader
- **PyMuPDFLoader**: Fast and efficient PDF parsing
- Recursively finds all PDFs in specified directory

### 2. Text Splitter
- **RecursiveCharacterTextSplitter**: Intelligently splits documents
- Preserves context with chunk overlap

### 3. Embeddings
- **HuggingFace BGE Model**: State-of-the-art embedding model
- COSINE distance for similarity search
- Batch processing with normalization

### 4. Vector Store
- **FAISS**: Facebook AI Similarity Search
- Efficient similarity search at scale
- Persistent storage support

### 5. LLM
- **Google Gemini**: Advanced AI for response generation
- Context-aware answers
- Fallback handling for insufficient context

## 🐛 Troubleshooting

### Common Issues

1. **Module not found error**
   - Ensure all dependencies are installed
   - Check Python version compatibility

2. **API Key error**
   - Verify `.env` file exists and contains `GOOGLE_API_KEY`
   - Check API key is valid and active

3. **Memory issues with large PDFs**
   - Reduce `batch_size` in embedding configuration
   - Process PDFs in smaller batches

4. **FAISS index not found**
   - Run the pipeline with `embed_and_store()` first
   - Check `faiss_index` directory exists

## 📝 Example Output

```
====== Found 5 PDF files to process ======

[INFO] Processing: document1.pdf file
✅ Successfully Loaded <10> pages from document1.pdf
==================================================

[INFO] Embedding Initializing...
==================================================

✅ Document Splitted successfully!
Splitted <50> documents into <125> chunks.
==================================================

[INFO] Vector dimension: 384
[INFO] Total Vectors in the store: <125>
==================================================

✅✅ Successfully saved FAISS index locally

[ANSWER]
The main topic discussed in the documents is...
```

## 👤 Author

**codewithyasho**
- GitHub: [@codewithyasho](https://github.com/codewithyasho)
- Repository: [simple-rag-pipeline](https://github.com/codewithyasho/simple-rag-pipeline)

## 🙏 Acknowledgments

- LangChain for the RAG framework
- HuggingFace for embedding models
- Google for Gemini AI
- Facebook AI for FAISS

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Google Gemini API](https://ai.google.dev/)
- [HuggingFace Models](https://huggingface.co/models)

---

⭐ If you find this project helpful, please consider giving it a star!