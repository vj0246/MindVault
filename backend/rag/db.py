import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Single shared Supabase client — imported by every module that needs it.
# create_client() opens an httpx connection pool; creating a fresh one on
# every function call (across 6+ modules, each invoked multiple times per
# request) accumulates uncleaned connections and OOMs Render's 512MB tier
# after a handful of requests. One shared client = one pool, reused forever.
_CLIENT = None

def get_supabase():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_KEY"]
        )
    return _CLIENT