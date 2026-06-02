import os
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )

def log_document(filename: str, path: str, chunk_count: int, document_id: str) -> str:
    supabase = get_supabase()
    
    existing = (
        supabase.table("documents")
        .select("id")
        .eq("filename", filename)
        .execute()
    )
    
    if existing.data:
        # Reuse existing id — return it so chunks reference correct row
        existing_id = existing.data[0]["id"]
        supabase.table("documents").update({
            "chunk_count": chunk_count,
            "uploaded_at": datetime.utcnow().isoformat()
        }).eq("filename", filename).execute()
        return existing_id
    else:
        supabase.table("documents").insert({
            "id": document_id,
            "filename": filename,
            "chunk_count": chunk_count,
            "uploaded_at": datetime.utcnow().isoformat()
        }).execute()
        return document_id

def get_all_documents() -> list:
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("filename, chunk_count, uploaded_at")
        .order("uploaded_at", desc=True)
        .execute()
    )
    return result.data

def get_document(filename: str) -> dict | None:
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("*")
        .eq("filename", filename)
        .execute()
    )
    return result.data[0] if result.data else None