import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # <--- NEW
from langchain_chroma import Chroma

# --- Configuration ---
PDF_PATH = "data/acl-protocol.pdf"
DB_PATH = "chroma_db"

def build_knowledge_base():
    print("🔄 Loading PDF...")
    
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"❌ Error: PDF not found at {PDF_PATH}")

    # 1. Load
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    print(f"📄 Loaded {len(docs)} pages.")

    # 2. Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"🧩 Split document into {len(chunks)} chunks.")

    # 3. Embed & Save (Locally)
    print("🧠 Generating Embeddings (Running locally on your CPU)...")
    
    # We use a standard, lightweight model. It downloads once.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Clear old DB if it exists to prevent conflicts
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )
    
    print("🚀 Success! Knowledge Base saved to /chroma_db")

def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    build_knowledge_base()