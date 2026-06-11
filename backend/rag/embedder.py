from fastembed import TextEmbedding

# Single shared instance — imported by both ingest.py and retrieve1.py.
# Prevents fastembed loading twice on startup (would OOM Render 512MB free tier).
EMBED_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
