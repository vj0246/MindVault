# `metadata/tracker.py`, `eval/evaluate.py`, `migrate_embeddings.py`, `rag/db.py`, and dead code

## `rag/db.py` (20 lines, the whole file)

```python
_CLIENT = None
def get_supabase():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _CLIENT
```

The single most-imported function in the backend — every module that touches Postgres imports `get_supabase` from here rather than creating its own client. Same singleton reasoning as everywhere else in this codebase (`groq_keys.py`, `retrieve1.py`'s LLM caches): a fresh `create_client()` per call opens a fresh `httpx` connection pool, and across 6+ modules each invoked multiple times per request, that accumulates into an OOM on Render's 512MB tier — this specific failure mode is documented in the main `README.md`'s production-engineering section as a real, previously-shipped bug.

Uses the **service role key** (`SUPABASE_SERVICE_KEY`), not the anon key — meaning every backend query bypasses Postgres RLS entirely. RLS policies (doc 8) exist as defense-in-depth against a hypothetical future code path that uses the anon key directly; the actual, current access control is that every query function explicitly filters by `user_id`, verified upstream by `get_current_user`'s JWT check (doc 2).

## `metadata/tracker.py` (90 lines)

Document CRUD, all scoped by `user_id`:

- `log_document(filename, path, chunk_count, document_id, user_id)` — called once per upload, before background ingestion starts. Has a re-upload special case: if a document with the same filename already exists for this user, it **deletes the old chunks first**, then updates the existing row in place (same `document_id`) rather than creating a duplicate. Without this, re-uploading a file with the same name would leave two overlapping sets of chunks in the vector store — the new upload's `chunk_count` would reflect only the new chunks, but retrieval would still surface the stale ones too, silently degrading answer quality with no visible symptom.
- `get_all_documents(user_id)` — ordered `uploaded_at desc` (most-recent-first). This ordering detail matters elsewhere: the frontend's landing-page suggestion generator used to rely on index `[0]`/`[1]` for "the current documents," which meant it only ever surfaced the newest 1-2 uploads regardless of vault size — fixed to spread across indices 0-3 (see doc 7).
- `set_document_folder`, `delete_document` (deletes chunks first, then the document row, scoped by both `id` and `user_id` so a user can never delete another user's document by guessing an ID), `get_document` (lookup by filename, used internally by the re-upload check in `log_document`).

## `eval/evaluate.py` (290 lines)

The RAGAS evaluation harness. Calls `rag.retrieve1.query_rag` **in-process** — imports the actual pipeline module and calls the function directly, no HTTP request, no running server required. This wasn't the original design: an earlier version hit the live deployed API over HTTP specifically to sidestep a dependency conflict (`langchain==0.3.25` pinned in production vs. `ragas==0.1.21` requiring `langchain<0.3`, which cannot coexist in one Python environment). That version also had a **hardcoded live JWT committed in plaintext across three separate commits** — a real, since-remediated secret-handling mistake, not a hypothetical risk.

The current version resolves the actual dependency conflict properly instead of routing around it: `ragas>=0.4` resolves cleanly against the production `langchain`/`langchain-core` pins, verified by a real `pip install --dry-run` (not guessed) against a dedicated `backend/eval_venv/` virtual environment (see the main `README.md`'s "Running this locally" section for the exact setup commands). Running in-process means eval exercises the *exact current code* — caching, Corrective RAG, Self-RAG, everything — instead of testing an HTTP contract that could silently drift from what the code actually does.

**Question set**: 25 hand-written Q&A pairs against a specific uploaded document (`OS Notes.pdf`), including one deliberately adversarial question (`"What is the Banker's Algorithm for deadlock avoidance?"`) whose ground truth is explicitly "the document does not contain this — a correct answer should say so" — designed to catch the system confidently hallucinating rather than admitting the gap. Running all 25 costs real Groq quota (`25 questions × 4 RAGAS metrics ≈ 80-100 judge calls`) and can take 25-40+ minutes with free-tier rate-limit retries; the script supports slicing (`QUESTIONS = QUESTIONS[:8]`, commented-out lines near the top) for cheaper partial runs during iterative tuning.

**A real bug this project hit and fixed**: the very first time this rewritten script actually ran end to end, all ~32 RAGAS judge calls completed successfully, then the script crashed on `tabulate(..., tablefmt="rounded_outline")` — Windows cp1252 stdout choking on Unicode box-drawing characters — *before* the results were ever saved to disk. Real Groq quota spent, zero results persisted. Fixed by moving the `json.dump()` save to run before any display/print step (so a crash in the pretty-printing can never cost the underlying data) and switching to `tablefmt="grid"` (plain ASCII). This is the same class of bug (non-ASCII crashing Windows console output) that was independently found and fixed in `ingest.py` and `migrate_embeddings.py` — a recurring gotcha on this project specifically because development happens on Windows while the actual deployment target (Render) is Linux, where the same code would never have crashed.

Results are written to `backend/eval/results_final/` as timestamped JSON, one file per run — committed to the repo (not gitignored) as a running record of measured accuracy over time.

## `migrate_embeddings.py` (70 lines, the whole file)

A one-off, safely-rerunnable migration script: fetches every row from `chunks` in batches of 200, re-embeds `content` with whatever model `rag/embedder.py` currently points at, and `.update()`s the `embedding` column in place (never `.insert()` — rerunning it is idempotent, no duplicate rows possible). This is the script that had to run once after the `all-MiniLM-L6-v2` → `bge-small-en-v1.5` swap, since old and new embeddings are vectors in different, mutually-incompatible spaces and mixing them in one similarity search silently produces meaningless rankings — there's no way to "gradually migrate," it has to be a full pass. A small `time.sleep(0.3)` between batches is a courtesy against hitting Supabase's own free-tier rate limits during the bulk update, not a correctness requirement.

## Dead code: `rag/retrieve.py`

165 lines, never imported anywhere in the live backend (verified — no `from rag.retrieve import` anywhere in `app.py`, `rag/*.py`, or `eval/*.py`). This is a leftover from an earlier architecture: FAISS as the vector store, `OllamaEmbeddings`/`ChatOllama` for embeddings and generation — both explicitly *rejected* decisions documented in the main `README.md`'s "Engineering decisions" table (FAISS lost to pgvector because Render's filesystem is ephemeral; Ollama lost to Groq because Render has no GPU). The file is essentially a fossil of the pre-Supabase, pre-Groq version of this project. It also imports `get_history_for_prompt` from `rag/memory.py` — the *only* live caller of that function (see doc 3), meaning that function is effectively dead too, just physically sitting in an otherwise-active file.

**Recommendation for whoever takes this over**: safe to delete `rag/retrieve.py` entirely. Nothing imports it, and its own imports (`langchain_community.vectorstores.FAISS`, `langchain_community.embeddings.OllamaEmbeddings`, `langchain_community.chat_models.ChatOllama`) aren't otherwise used in the live code either.
