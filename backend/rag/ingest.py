import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "C:/sentence_transformers_cache"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )

def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

def chunk_documents(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(pages)

def embed_and_store(chunks, document_id: str, filename: str):
    supabase = get_supabase()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=True)

    rows = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        rows.append({
            "document_id": document_id,
            "content": chunk.page_content,
            "embedding": embedding.tolist(),
            "chunk_index": i,
            "filename": filename
        })

    batch_size = 50
    for i in range(0, len(rows), batch_size):
        supabase.table("chunks").insert(rows[i:i+batch_size]).execute()

    print(f"[Ingest] Stored {len(rows)} chunks in Supabase")  


def ingest_document(file_path: str, document_id: str, chunk_size=500, chunk_overlap=50):
    print(f"[Ingest] Loading: {file_path}")
    pages = load_pdf(file_path)
    print(f"[Ingest] Loaded {len(pages)} pages")

    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    print(f"[Ingest] Created {len(chunks)} chunks")

    print(f"[Ingest] Embedding and storing...")
    filename = os.path.basename(file_path)
    embed_and_store(chunks, document_id, filename)
    print(f"[Ingest] Done.")

    return chunks