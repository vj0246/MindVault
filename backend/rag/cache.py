import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from rag.db import get_supabase

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not (args and callable(args[0])) else args[0]

CACHE_TTL_HOURS = 12

# ── "Does this user have any documents at all" existence check ─────────────
# query_rag/stream_rag both hit Supabase for this on every single call just
# to decide whether to show the "no documents uploaded" message. It rarely
# changes, so a short in-process TTL cache turns that into a Supabase round
# trip roughly once a minute per user instead of once per message. Uploads
# call invalidate_has_chunks() immediately so a fresh upload is never masked
# by a stale cached "no documents" result.
_HAS_CHUNKS_TTL_SECONDS = 60
_has_chunks_cache: dict[str, tuple[bool, float]] = {}


def has_any_chunks(user_id: str | None) -> bool:
    key = user_id or "__anonymous__"
    cached = _has_chunks_cache.get(key)
    now = time.monotonic()
    if cached and cached[1] > now:
        return cached[0]

    supabase = get_supabase()
    query = supabase.table("chunks").select("id")
    if user_id:
        query = query.eq("user_id", user_id)
    result = query.limit(1).execute()
    value = bool(result.data)
    _has_chunks_cache[key] = (value, now + _HAS_CHUNKS_TTL_SECONDS)
    return value


def invalidate_has_chunks(user_id: str) -> None:
    _has_chunks_cache.pop(user_id, None)

def make_cache_key(question: str, user_id: str, document_ids: list, mode: str) -> str:
    """Deterministic key from everything that affects the answer.
    Same question text legitimately produces different correct answers for
    different users, different scoped document sets, or different modes --
    all four must be in the key."""
    normalized_q = " ".join(question.strip().lower().split())
    sorted_docs = sorted(document_ids) if document_ids else []
    raw = json.dumps({
        "q": normalized_q,
        "user_id": user_id,
        "docs": sorted_docs,
        "mode": mode
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

@traceable(name="cache_get", run_type="tool")
def get_cached(cache_key: str) -> dict | None:
    try:
        supabase = get_supabase()
        now = datetime.now(timezone.utc).isoformat()
        result = (
            supabase.table("query_cache")
            .select("result")
            .eq("cache_key", cache_key)
            .gt("expires_at", now)
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]["result"]
        return None
    except Exception as e:
        # Cache failures should never break a live query -- fall through to
        # the normal pipeline as if it were simply a cache miss.
        print(f"[Cache] get failed (treating as miss): {e}")
        return None

@traceable(name="cache_set", run_type="tool")
def set_cached(cache_key: str, result: dict, ttl_hours: int = CACHE_TTL_HOURS):
    try:
        supabase = get_supabase()
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=ttl_hours)
        supabase.table("query_cache").upsert({
            "cache_key": cache_key,
            "result": result,
            "created_at": now.isoformat(),
            "expires_at": expires.isoformat(),
        }).execute()
    except Exception as e:
        print(f"[Cache] set failed (non-fatal): {e}")