import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )

def load_document(file_path: str) -> list:
    """Load any supported document type (PDF, TXT, MD)."""
    ext = os.path.splitext(file_path)[1].lower()
    print(f"[Ingest] Loading file with extension: {ext}")

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext in (".txt", ".md"):
        # TextLoader with autodetect encoding to handle different file encodings
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        except Exception:
            loader = TextLoader(file_path, encoding="latin-1")
            return loader.load()
    else:
        # Fallback: read as plain text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": file_path})]

def chunk_documents(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(pages)

def clear_old_chunks(document_id: str):
    """Delete existing chunks for a document before re-ingesting."""
    supabase = get_supabase()
    result = supabase.table("chunks").delete().eq("document_id", document_id).execute()
    print(f"[Ingest] Cleared old chunks for document_id={document_id}")
    return result

def embed_and_store(chunks, document_id: str, filename: str, user_id: str):
    supabase = get_supabase()
    texts = [chunk.page_content for chunk in chunks]
    
    if not texts:
        print("[Ingest] No chunks to embed.")
        return

    all_embeddings = list(EMBED_MODEL.embed(texts))

    rows = []
    batch_size = 50
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        rows.append({
            "document_id": document_id,
            "content": chunk.page_content,
            "embedding": embedding.tolist(),
            "chunk_index": i,
            "filename": filename,
            "user_id": user_id
        })

    for i in range(0, len(rows), batch_size):
        supabase.table("chunks").insert(rows[i:i+batch_size]).execute()

    print(f"[Ingest] Stored {len(rows)} chunks in Supabase")

def ingest_document(file_path: str, document_id: str, user_id: str, chunk_size=500, chunk_overlap=50):
    print(f"[Ingest] Loading: {file_path}")
    pages = load_document(file_path)
    print(f"[Ingest] Loaded {len(pages)} pages/sections")

    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    print(f"[Ingest] Created {len(chunks)} chunks")

    # Clear old chunks for this document before inserting new ones
    clear_old_chunks(document_id)

    print(f"[Ingest] Embedding and storing...")
    filename = os.path.basename(file_path)
    embed_and_store(chunks, document_id, filename, user_id)
    print(f"[Ingest] Done.")

    return chunks