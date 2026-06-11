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

def log_document(filename: str, path: str, chunk_count: int, document_id: str, user_id: str) -> str:
    supabase = get_supabase()
    
    # Check existing per user — not global
    existing = (
        supabase.table("documents")
        .select("id")
        .eq("filename", filename)
        .eq("user_id", user_id)
        .execute()
    )
    
    if existing.data:
        existing_id = existing.data[0]["id"]
        supabase.table("documents").update({
            "chunk_count": chunk_count,
            "uploaded_at": datetime.utcnow().isoformat()
        }).eq("id", existing_id).execute()
        return existing_id
    else:
        supabase.table("documents").insert({
            "id": document_id,
            "filename": filename,
            "chunk_count": chunk_count,
            "uploaded_at": datetime.utcnow().isoformat(),
            "user_id": user_id
        }).execute()
        return document_id

def get_all_documents(user_id: str) -> list:
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("id, filename, chunk_count, uploaded_at")
        .eq("user_id", user_id)
        .order("uploaded_at", desc=True)
        .execute()
    )
    return result.data

def get_document(filename: str, user_id: str) -> dict | None:
    supabase = get_supabase()
    result = (
        supabase.table("documents")
        .select("*")
        .eq("filename", filename)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None