import os
from datetime import datetime
from rag.db import get_supabase
from dotenv import load_dotenv

load_dotenv()


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
        # Re-upload of same filename: delete OLD chunks for this document
        # before re-ingestion runs. Without this, new chunks get appended
        # alongside stale ones under the same document_id -- chunk_count
        # tracks only the new count, but the chunks table accumulates both,
        # so retrieval returns duplicate/outdated content.
        supabase.table("chunks").delete().eq("document_id", existing_id).execute()
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