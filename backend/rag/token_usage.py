from rag.db import get_supabase

# Our own fairness-cap constant, not Groq's actual account quota -- Groq's
# real per-model TPD limits aren't queryable from the API, and vary by model
# and account tier. This is a self-imposed daily budget purely so users have
# a transparency signal, not a hard billing ceiling.
DAILY_TOKEN_BUDGET = 100_000
_WINDOW_SECONDS = 86400


def record_token_usage(user_id: str | None, total_tokens: int | None) -> dict:
    """Adds total_tokens to the user's rolling 24h counter (reuses the
    existing rate_limits table via increment_token_usage RPC -- see
    migration add_increment_token_usage_rpc) and returns the running total
    plus its percentage of DAILY_TOKEN_BUDGET. Fails open: any error, or no
    user_id/tokens to record, returns zeros instead of breaking the answer
    that's already been generated."""
    if not user_id or not total_tokens:
        return {"daily_used": 0, "daily_pct": 0}
    try:
        supabase = get_supabase()
        result = supabase.rpc("increment_token_usage", {
            "p_key": f"tokens:user:{user_id}",
            "p_tokens": total_tokens,
            "p_window_seconds": _WINDOW_SECONDS
        }).execute()
        data = result.data
        if isinstance(data, list):
            data = data[0] if data else 0
        daily_used = int(data or 0)
        pct = min(100, round(100 * daily_used / DAILY_TOKEN_BUDGET))
        return {"daily_used": daily_used, "daily_pct": pct}
    except Exception as e:
        print(f"[TokenUsage] record failed, omitting from response: {e}")
        return {"daily_used": 0, "daily_pct": 0}
