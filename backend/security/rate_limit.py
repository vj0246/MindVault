from fastapi import HTTPException
from rag.db import get_supabase

# IP ceiling set higher than per-user limit -- per-user is the primary control,
# per-IP is a secondary backstop against one IP running many accounts.
IP_MULTIPLIER = 3


def _check(action: str, identifier: str, limit: int, window_seconds: int) -> bool:
    supabase = get_supabase()
    key = f"{action}:{identifier}"
    result = supabase.rpc("check_rate_limit", {
        "p_key": key,
        "p_limit": limit,
        "p_window_seconds": window_seconds
    }).execute()
    data = result.data
    # Scalar-returning RPCs sometimes come back as bare bool, sometimes as
    # a single-element list depending on supabase-py version -- handle both.
    if isinstance(data, list):
        data = data[0] if data else False
    return bool(data)


def enforce_rate_limit(action: str, user_id: str, ip: str, user_limit: int,
                        window_seconds: int = 3600):
    """Raises 429 if either the per-user or per-IP window is exceeded.
    Fails OPEN on Supabase errors -- a rate-limiter outage should never
    take down the whole API."""
    try:
        if not _check(action, f"user:{user_id}", user_limit, window_seconds):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: max {user_limit} {action} requests per hour."
            )
        if not _check(action, f"ip:{ip}", user_limit * IP_MULTIPLIER, window_seconds):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded for this network. Try again later."
            )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[RateLimit] check failed, allowing request through: {e}")