import os
from datetime import datetime, timezone
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )

# ─────────────────────────────────────────────────────────────
# Message history (existing)
# ─────────────────────────────────────────────────────────────

def get_session_history(session_id: str, user_id: str) -> list:
    supabase = get_supabase()
    result = (
        supabase.table("sessions")
        .select("role, content, timestamp")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("timestamp")
        .execute()
    )
    return result.data

def save_session_message(session_id: str, role: str, content: str, user_id: str):
    supabase = get_supabase()
    supabase.table("sessions").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id
    }).execute()
    # Keep last_active fresh on every message
    _touch_chat_session(session_id, user_id)

def clear_session_messages(session_id: str, user_id: str):
    supabase = get_supabase()
    supabase.table("sessions").delete()\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .execute()

# ─────────────────────────────────────────────────────────────
# Chat session metadata (new — multi-session support)
# ─────────────────────────────────────────────────────────────

def list_chat_sessions(user_id: str) -> list:
    """Return all named sessions for user, newest first."""
    supabase = get_supabase()
    result = (
        supabase.table("chat_sessions")
        .select("id, name, created_at, last_active")
        .eq("user_id", user_id)
        .order("last_active", desc=True)
        .execute()
    )
    return result.data or []

def create_chat_session(session_id: str, user_id: str, name: str = "New Chat") -> dict:
    """Create a new named session. Returns the created row."""
    supabase = get_supabase()
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": session_id,
        "user_id": user_id,
        "name": name,
        "created_at": now,
        "last_active": now,
    }
    supabase.table("chat_sessions").insert(row).execute()
    return row

def rename_chat_session(session_id: str, user_id: str, name: str):
    supabase = get_supabase()
    supabase.table("chat_sessions")\
        .update({"name": name[:60]})\
        .eq("id", session_id)\
        .eq("user_id", user_id)\
        .execute()

def delete_chat_session(session_id: str, user_id: str):
    """Delete session metadata + all messages."""
    supabase = get_supabase()
    supabase.table("sessions").delete()\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .execute()
    supabase.table("chat_sessions").delete()\
        .eq("id", session_id)\
        .eq("user_id", user_id)\
        .execute()

def _touch_chat_session(session_id: str, user_id: str):
    """Update last_active. Called automatically on every message save."""
    try:
        supabase = get_supabase()
        supabase.table("chat_sessions")\
            .update({"last_active": datetime.now(timezone.utc).isoformat()})\
            .eq("id", session_id)\
            .eq("user_id", user_id)\
            .execute()
    except Exception:
        pass  # Non-critical — don't break message flow

# ─────────────────────────────────────────────────────────────
# History compression for prompt (existing, kept)
# ─────────────────────────────────────────────────────────────

def get_history_for_prompt(history: list, llm) -> str:
    if len(history) <= 6:
        return "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ])
    old_history = history[:-4]
    recent_history = history[-4:]
    old_text = "\n".join([f"{m['role']}: {m['content']}" for m in old_history])
    summary_prompt = f"""Summarize this conversation in 2 sentences maximum.
Capture only the key topics discussed.

Conversation:
{old_text}

Summary:"""
    summary = llm.invoke(summary_prompt).content
    recent_text = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent_history
    ])
    return f"Earlier conversation summary: {summary}\n\nRecent messages:\n{recent_text}"