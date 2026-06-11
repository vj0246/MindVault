import os
import base64
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastembed import TextEmbedding
from supabase import create_client
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp", ".tiff": "image/tiff"
}

def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )

def load_document(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path).load()

    elif ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()

    elif ext in (".docx", ".doc"):
        return Docx2txtLoader(file_path).load()

    elif ext in IMAGE_EXTENSIONS:
        return load_image_via_groq(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_image_via_groq(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = IMAGE_MIME.get(ext, "image/jpeg")

    with open(file_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all content from this image in full detail. "
                        "If it contains text or notes, transcribe exactly. "
                        "If it contains diagrams, charts, or tables, describe them thoroughly. "
                        "Preserve all structure and information."
                    )
                }
            ]
        }],
        max_tokens=2000
    )

    extracted = response.choices[0].message.content
    filename = os.path.basename(file_path)
    print(f"[Ingest] Image extracted via Groq vision: {len(extracted)} chars")
    return [Document(page_content=extracted, metadata={"source": filename, "type": "image"})]

def chunk_documents(pages, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(pages)

def embed_and_store(chunks, document_id: str, filename: str, user_id: str):
    supabase = get_supabase()
    texts = [chunk.page_content for chunk in chunks]
    all_embeddings = list(EMBED_MODEL.embed(texts))

    rows = []
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        rows.append({
            "document_id": document_id,
            "content": chunk.page_content,
            "embedding": embedding.tolist(),
            "chunk_index": i,
            "filename": filename,
            "user_id": user_id
        })

    batch_size = 50
    for i in range(0, len(rows), batch_size):
        supabase.table("chunks").insert(rows[i:i+batch_size]).execute()

    print(f"[Ingest] Stored {len(rows)} chunks in Supabase")

def ingest_document(file_path: str, document_id: str, user_id: str, chunk_size=500, chunk_overlap=50):
    print(f"[Ingest] Loading: {file_path}")
    pages = load_document(file_path)
    print(f"[Ingest] Loaded {len(pages)} pages/sections")

    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    print(f"[Ingest] Created {len(chunks)} chunks")

    filename = os.path.basename(file_path)
    embed_and_store(chunks, document_id, filename, user_id)
    print(f"[Ingest] Done.")

    return chunks
