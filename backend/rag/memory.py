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
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id
    }).execute()

def clear_session(session_id: str, user_id: str):
    supabase = get_supabase()
    supabase.table("sessions").delete()\
        .eq("session_id", session_id)\
        .eq("user_id", user_id)\
        .execute()

def get_history_for_prompt(history: list, llm) -> str:
    if len(history) <= 6:
        return "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ])
    else:
        old_history = history[:-4]
        recent_history = history[-4:]

        old_text = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in old_history
        ])

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