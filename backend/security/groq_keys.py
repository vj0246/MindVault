import os
from groq import Groq

_KEYS = None
_CLIENTS = {}


def get_client(key: str) -> Groq:
    """One raw Groq SDK client per key, cached -- avoids opening a fresh
    httpx connection pool per call across every module that needs one."""
    if key not in _CLIENTS:
        _CLIENTS[key] = Groq(api_key=key)
    return _CLIENTS[key]


def get_groq_keys() -> list:
    """GROQ_API_KEY, comma-separated if multiple. Cached after first read.
    Works identically for one key or many -- no separate env var needed."""
    global _KEYS
    if _KEYS is None:
        raw = os.environ.get("GROQ_API_KEY", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise RuntimeError("GROQ_API_KEY is not set")
        _KEYS = keys
    return _KEYS


def call_with_key_fallback(fn):
    """Calls fn(key) for each configured Groq key in order, moving to the
    next key if one raises (rate limit / quota / invalid key). Re-raises the
    last error only if every key fails."""
    last_err = None
    for key in get_groq_keys():
        try:
            return fn(key)
        except Exception as e:
            last_err = e
            continue
    raise last_err
