import json
import os
from datetime import datetime

MEMORY_FILE = "data/sessions.json"

def _load_all_sessions() -> dict:
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def _save_all_sessions(sessions: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(sessions, f, indent=2)

def get_session_history(session_id: str) -> list:
    sessions = _load_all_sessions()
    return sessions.get(session_id, [])


def save_session_message(session_id: str, role: str, content: str):
    sessions = _load_all_sessions()
    
    if session_id not in sessions:
        sessions[session_id] = []
    
    sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    _save_all_sessions(sessions)


def clear_session(session_id: str):
    sessions = _load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
    _save_all_sessions(sessions)

def get_history_for_prompt(history: list, llm) -> str:
    if len(history) <= 6:
        # Just format raw history
        return "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ])
    else:
        # Summarize everything older than last 4 messages
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