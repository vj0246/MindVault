# MindVault — Engineering Documentation

This folder is the handover doc: file-by-file, what each piece does, why it's built that way, and what to watch out for. The top-level `README.md` is the portfolio-facing pitch; this is the internal reference for someone taking over the codebase.

Read in this order if you're new to the project:

1. [`01-ARCHITECTURE.md`](01-ARCHITECTURE.md) — system topology, deployment, the request lifecycle in full detail
2. [`02-BACKEND-APP.md`](02-BACKEND-APP.md) — `app.py`, every route, auth, background tasks
3. [`03-BACKEND-RAG-PIPELINE.md`](03-BACKEND-RAG-PIPELINE.md) — the `rag/` module: embedding, ingestion, retrieval, caching, memory. This is the core of the product.
4. [`04-BACKEND-SECURITY.md`](04-BACKEND-SECURITY.md) — `security/` module: Groq key fallback, input moderation, rate limiting
5. [`05-BACKEND-GRAPH.md`](05-BACKEND-GRAPH.md) — `graph/` module: entity extraction, graph storage/caching
6. [`06-BACKEND-METADATA-EVAL-MISC.md`](06-BACKEND-METADATA-EVAL-MISC.md) — `metadata/tracker.py`, `eval/evaluate.py`, `migrate_embeddings.py`, `rag/db.py`, and the dead code worth knowing about
7. [`07-FRONTEND.md`](07-FRONTEND.md) — the Next.js app: `page.tsx` (the whole UI lives in one file), `lib/`, `components/`, auxiliary routes
8. [`08-DATABASE.md`](08-DATABASE.md) — full schema, RLS policies, RPC functions, and how they map to `backend/migrations/001_initial_schema.sql`
9. [`09-TRADEOFFS-AND-DECISIONS.md`](09-TRADEOFFS-AND-DECISIONS.md) — every "why this and not that" call made on this project, with the evidence behind it
10. [`10-KNOWN-ISSUES-AND-HANDOVER.md`](10-KNOWN-ISSUES-AND-HANDOVER.md) — dead code, unfinished work, operational gotchas, how to debug the failure modes that have actually happened in production

## Fastest orientation

- **What is this**: a multi-tenant RAG app. Upload documents, ask questions, get cited answers grounded in what you uploaded, with a fallback to general knowledge when retrieval doesn't find a real match — and a knowledge graph of extracted entities/relationships on the side.
- **Where the real complexity lives**: `backend/rag/retrieve1.py` (1076 lines, one file, the entire query pipeline — query rewriting, hybrid retrieval, RRF fusion, reranking, Corrective RAG, Self-RAG, caching, streaming, token accounting) and `frontend/app/page.tsx` (the entire UI, ~1300+ lines, one file — this was never split into components as the app grew, which is itself a documented tradeoff, see doc 9).
- **Where it runs**: FastAPI backend on Render (free tier — 512MB RAM, shared single vCPU; this constraint drove a large fraction of the actual engineering, see doc 10), Next.js frontend on Vercel, Postgres+pgvector on Supabase.
- **What's dead code**: `backend/rag/retrieve.py` (no "1") and `frontend/components/GraphExplorer.tsx` — both fully-formed alternate implementations, neither imported anywhere. See doc 6 and doc 7.
