import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def chunk_documents(pages, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    return chunks


def embed_and_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    vectorstore_path = "vectorstore1"
    
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(
            vectorstore_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore.save_local(vectorstore_path)
    return vectorstore

def ingest_document(file_path: str, chunk_size=500, chunk_overlap=50):
    print(f"[Ingest] Loading: {file_path}")
    pages = load_pdf(file_path)
    print(f"[Ingest] Loaded {len(pages)} pages")
    
    print(f"[Ingest] Chunking...")
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    print(f"[Ingest] Created {len(chunks)} chunks")
    
    print(f"[Ingest] Embedding and storing...")
    embed_and_store(chunks)
    print(f"[Ingest] Done.")
    
    return chunks

'''
from rag.ingest import ingest_document
chunks = ingest_document("data\docs\OS Notes.pdf")
print(len(chunks))
print(chunks[0].page_content)
print(chunks[0].metadata)
'''