from fastembed import TextEmbedding

# Single shared instance — imported by both ingest.py and retrieve1.py.
# Prevents fastembed loading twice on startup (would OOM Render 512MB free tier).
#
# BAAI/bge-small-en-v1.5 replaces all-MiniLM-L6-v2 -- same 384-dim output
# (matches the existing pgvector column, no schema migration), stronger
# retrieval benchmarks. Eval baseline showed weak context_precision (0.62)
# and context_recall (0.56); a better embedding model is the deepest lever
# for both since they're determined at the retrieval layer, not generation.
# Requires running migrate_embeddings.py once after deploy to re-embed all
# existing chunks -- old vectors are a different, incompatible space.
EMBED_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
