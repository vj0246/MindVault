"""
Re-embed all chunks in Supabase using the new BAAI/bge-small-en-v1.5 model.

Run ONCE after deploying embedder.py change:
    cd backend
    python migrate_embeddings.py

What it does:
- Fetches all chunks from Supabase in batches of 200
- Re-embeds each batch with the new model
- Updates the embedding column in-place
- Does NOT touch content/metadata/user_id

Safe to re-run: uses .update() not .insert(), so no duplicates.
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rag.db import get_supabase
from rag.embedder import EMBED_MODEL

BATCH_SIZE = 200

def migrate():
    supabase = get_supabase()

    # Count total chunks
    count_res = supabase.table("chunks").select("id", count="exact").execute()
    total = count_res.count or 0
    print(f"[Migrate] Total chunks: {total}")

    if total == 0:
        print("[Migrate] Nothing to migrate.")
        return

    offset = 0
    migrated = 0

    while offset < total:
        print(f"[Migrate] Fetching {offset}-{offset + BATCH_SIZE}...")
        res = supabase.table("chunks")\
            .select("id, content")\
            .range(offset, offset + BATCH_SIZE - 1)\
            .execute()

        rows = res.data or []
        if not rows:
            break

        texts = [r["content"] for r in rows]
        embeddings = list(EMBED_MODEL.embed(texts))

        for row, emb in zip(rows, embeddings):
            supabase.table("chunks")\
                .update({"embedding": emb.tolist()})\
                .eq("id", row["id"])\
                .execute()

        migrated += len(rows)
        offset += BATCH_SIZE
        print(f"[Migrate] {migrated}/{total} done")
        time.sleep(0.3)  # Supabase free tier rate limit courtesy

    print(f"[Migrate] Done. {migrated} chunks re-embedded with BAAI/bge-small-en-v1.5")

if __name__ == "__main__":
    migrate()