# MindVault

Personal RAG knowledge-base app: upload documents, ask questions grounded in them (Corrective RAG + Self-RAG verification), chat memory, knowledge graph, folder organization.

## Stack

- **Backend:** Python 3.11+, FastAPI, Supabase (Postgres + pgvector + RLS), Groq (`langchain_groq` + raw `groq` SDK, multi-key comma-separated fallback), `fastembed` local embeddings, LangChain 0.3.x, LangSmith tracing.
- **Frontend:** Next.js 14 App Router, TypeScript, Tailwind, deployed on Vercel.
- **Backend deploy:** Render (free tier — weak CPU, cold starts; UptimeRobot pings it to stay warm).

## Run / build / test

```
# backend
cd backend && pip install -r requirements.txt && uvicorn app:app --reload
# eval (separate deps)
pip install -r requirements.txt -r eval/requirements_eval.txt
python eval/evaluate.py   # writes to eval/results_final/

# frontend
cd frontend && npm install && npm run dev
npx tsc --noEmit -p .     # typecheck before every commit
```

Backend sanity check before every commit: `python -m py_compile app.py` (or the changed files).

## Directory map

- `backend/app.py` — all FastAPI routes.
- `backend/rag/` — `ingest.py` (parse/chunk/embed), `retrieve1.py` (CRAG + Self-RAG query chain), `cache.py`, `token_usage.py`, `memory.py`.
- `backend/security/` — `groq_keys.py` (multi-key fallback), `guardrails.py` (input moderation), `rate_limit.py`.
- `backend/metadata/tracker.py` — document CRUD (folders, delete).
- `backend/graph/` — entity/relation extraction + knowledge graph store.
- `backend/eval/` — RAGAS evaluation, results in `eval/results_final/`.
- `frontend/app/page.tsx` — single-page app shell (chat, sidebar, all modals). `frontend/lib/api.ts` — backend client.

## Env vars (names only)

Backend: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `GROQ_API_KEY` (comma-separated for fallback), `ALLOWED_ORIGINS`, `FRONTEND_URL`.
Frontend: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`.

## Known gotchas

- `backend/eval_env/`, `backend/eval_venv/`, `mindvault/` were committed venvs — gitignored now, never re-add.
- Document ingestion runs as a FastAPI `BackgroundTask` (not inline in `/upload`) — Render's request timeout killed inline ingestion on large PDFs. `_ingest_and_finalize` wraps `ingest_document` in a `ThreadPoolExecutor` with a 300s hard timeout so a stuck ingestion surfaces as a real failure instead of an eternal `chunk_count=-1` zombie row.
- Non-ASCII characters in `print()` crash on Windows cp1252 stdout — keep log lines ASCII-only.
- pip pins: don't guess narrow version ceilings — dry-run the full `requirements.txt` graph before pushing (two prior Render deploy failures from this).
