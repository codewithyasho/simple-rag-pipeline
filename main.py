'''
simple rag pipeline

'''

from pathlib import Path
from langchain_classic.document_loaders import PyMuPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# ==========================================================================

# 1.Data ingestion pipeline
# read all the pdfs inside the directory


def process_all_pdfs(directory):
    '''Process all pdfs in a directory using PyMuPDF'''

    all_documents = []
    pdf_dir = Path(directory)

    # finding all pdfs recursively
    pdf_files = list(pdf_dir.glob('**/*.pdf'))

    print(f"\n====== Found {len(pdf_files)} PDF files to process ======")

    for file in pdf_files:
        print(f"\n[INFO] Processing: {file.name} file")

        try:
            loader = PyMuPDFLoader(
                str(file)
            )
            documents = loader.load()

            # .extend() adds individual items to the list
            all_documents.extend(documents)

            print(
                f"\n✅ Successfully Loaded <{len(documents)}> pages from {file.name}")
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")
            continue

    print(f"\n\n[INFO] Total documents loaded: <{len(all_documents)}>\n")
    return all_documents


# ==========================================================================

# 2.splitting documents into chunks
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunked_documents = text_splitter.split_documents(documents)

    print("\n✅ Document Splitted successfully!")
    print(
        f"\nSplitted <{len(documents)}> documents into <{len(chunked_documents)}> chunks.")
    print("=" * 50)

    return chunked_documents


# ==========================================================================

# 3.creating new vectorstore from scratch
def embed_and_store(chunks):

    try:
        print("\n[INFO] Embedding Initializing...")
        print("=" * 50)

        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5", show_progress=True,
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'batch_size': 32,
                'normalize_embeddings': True
            }

        )

        print("\n[INFO] VectorStore Initializing...")
        print("=" * 50)

        # Creates a new FAISS index from scratch
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model,
            distance_strategy='COSINE'  # Better for normalized embeddings
        )

        print(f"\n[INFO] Vector dimension: {vectorstore.index.d}")

        print(
            f"[INFO] Total Vectors in the store: <{vectorstore.index.ntotal}>")
        print("=" * 50)

        # Save
        vectorstore.save_local("faiss_index")
        print("\n✅✅ Successfully saved FAISS index locally")

        return vectorstore

    except Exception as e:
        print(f"❌ Error during embedding and storing: {e}")


# ==========================================================================

# 3.1 if vector store exists, directly load it from directory
# loading the vectorstore from disk
def embed_and_load(embedding_model, vectorstore_path):
    try:
        print("\n[INFO] Embedding Initializing...")
        print("=" * 50)

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, show_progress=True,
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'batch_size': 32,
                'normalize_embeddings': True
            }

        )

        print("\n[INFO] VectorStore Initializing...")
        print("=" * 50)

        # loading existing vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        print(f"\n[INFO] Vector dimension: {vectorstore.index.d}")

        print(
            f"[INFO] Total Vectors in the store: <{vectorstore.index.ntotal}>")
        print("=" * 50)

        print("\n✅✅ Successfully LOADED Embeddings and Vectorstore.")

        return vectorstore

    except Exception as e:
        print(f"❌ Error during LOADING: {e}")


# ==========================================================================

# 3.2 if vectorstore exist and want to add more documents, load existing vectorstore and add more docs/ new chunks to it

# loading existing vectorstore and adding more documents/ new chunks
def load_and_add_new_docs(embedding_model, vectorstore_path, new_chunks):
    try:
        print("\n[INFO] Embedding Initializing...")
        print("=" * 50)

        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, show_progress=True,
            model_kwargs={
                'device': 'cpu'
            },
            encode_kwargs={
                'batch_size': 32,
                'normalize_embeddings': True
            }

        )

        print("\n[INFO] VectorStore Initializing...")
        print("=" * 50)

        # loading existing vectorstore
        vectorstore = FAISS.load_local(
            vectorstore_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        print("\n[INFO] Adding new CHUNKS to the Vectorstore...")

        # adding new documents/ chunks to existing vectorstore
        vectorstore.add_documents(new_chunks)

        print(f"\n[INFO] Vector dimension: {vectorstore.index.d}")

        print(
            f"[INFO] Total Vectors in the store: <{vectorstore.index.ntotal}>")
        print("=" * 50)

        print("\n✅✅ Successfully ADDED new chunks to the Vectorstore.")

        return vectorstore

    except Exception as e:
        print(f"❌ Error during LOADING and ADDING: {e}")


# ==========================================================================

# 4. create RAG pipeline

def simple_rag(query, vectorstore):
    # Retrieve similar documents
    similar_docs = vectorstore.similarity_search(
        query=query,
        k=3
    )

    # Initialize the Google Generative AI chat model
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro"
    )

    # Create a prompt by combining the query with the content of similar documents
    context = "\n\n".join([doc.page_content for doc in similar_docs])

    prompt = ChatPromptTemplate.from_template(
        '''
    Using the following context to answer the question below. 
    If the context is insufficient, please say "I don't know".
    
    <context>
    {context}
    </context>

    question: {query}
    '''
    )

    prompt = prompt.format_prompt(
        context=context,
        query=query
    )

    # Generate a response using the chat model
    response = chat_model.invoke(prompt)

    # print(context)

    return response.content


# ==========================================================================

if __name__ == "__main__":
    # Step 1: Process all PDFs in the specified directory
    # documents = process_all_pdfs("data/pdfs")

    # Step 2: Split documents into chunks
    # chunked_documents = split_docs(documents)

    # Step 3: Embed and store the chunks in a vectorstore
    # vectorstore = embed_and_store(chunked_documents)

    # Step 3.1: Alternatively, load existing vectorstore from disk
    vectorstore = embed_and_load(embedding_model="BAAI/bge-small-en-v1.5",
                                 vectorstore_path="faiss_index")

    # # Step 3.2: Alternatively, load existing vectorstore and add new documents
    # vectorstore = load_and_add_new_docs(embedding_model="BAAI/bge-small-en-v1.5",
    #                                     vectorstore_path="faiss_index",
    #                                     new_chunks=chunked_documents)

    query = "What is the main topic discussed in the documents?"

    # Step 4: Perform RAG using the query and vectorstore
    answer = simple_rag(query, vectorstore)

    print("\n[ANSWER]")
    print(answer)
