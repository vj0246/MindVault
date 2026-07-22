# Architecture

## Deployment topology

```
Browser (Vercel-hosted Next.js static/SSR bundle)
    │
    ├── Supabase Auth (JWT) ── direct browser-to-Supabase, no backend hop
    │
    └── FastAPI backend (Render, free tier)
            │
            ├── Supabase Postgres + pgvector (documents, chunks, sessions,
            │   chat_sessions, graph_nodes, graph_edges, query_cache,
            │   rate_limits, user_preferences, user_memory_notes)
            │
            ├── Groq Cloud (llama-3.3-70b-versatile primary,
            │   llama-3.1-8b-instant fallback + verification model,
            │   openai/gpt-oss-safeguard-20b for moderation,
            │   openai/gpt-oss-120b as the RAGAS judge in eval)
            │
            └── Local CPU inference (same process as the backend, no
                network hop): fastembed (BAAI/bge-small-en-v1.5) for
                embeddings, FlashRank (ms-marco-MiniLM-L-12-v2, ONNX) for
                reranking
```

Auth is the one path that bypasses the backend entirely — the browser talks to Supabase Auth directly for login/signup/session refresh, gets a JWT, and attaches it as a Bearer token to every backend request afterward. The backend never sees a password; it only ever verifies a JWT (`get_current_user` in `app.py`, calls `supabase.auth.get_user(token)`).

There is no separate worker process, no message queue, no Redis. Every piece of "background" work (document ingestion, knowledge-graph extraction) runs via FastAPI's `BackgroundTasks`, inside the same process, on the same box. This is a direct consequence of the free-tier hosting constraint — Render's free tier gives you one process, not a process pool, so there was never a good reason to build queue infrastructure only to run it on the same single box anyway.

## Why no separate vector database

pgvector lives inside the same Postgres instance as everything else. This wasn't a "pick the popular one" decision — FAISS was tried first and rejected specifically because Render's filesystem is ephemeral (wiped on every redeploy and, apparently, on the OOM-triggered container restarts this project hit more than once — see doc 10). Anything written to local disk that isn't re-derivable from Postgres is a ticking time bomb. pgvector persists because Postgres persists; there's exactly one source of truth for retrieval state and it's the same database everything else already depends on.

## Why local embeddings, not an embeddings API

`fastembed` runs `BAAI/bge-small-en-v1.5` in-process, in pure ONNX (no `torch`), producing 384-dim vectors with zero network calls per query. Three real things forced this:

1. `sentence-transformers` (which needs `torch`) was tried first and OOM'd the 512MB Render box on startup, before a single request was even served.
2. The Hugging Face Inference API was tried as a fallback — Render's free tier blocks outbound DNS resolution to `api-inference.huggingface.co`. This isn't documented anywhere obvious; it was found by watching a request hang and then fail with a DNS resolution error in the logs.
3. Local CPU inference means embedding cost is bounded by request volume, not by an external API's rate limits or per-call latency — directly relevant given how much of this project's engineering (see doc 10) was about surviving Render's own CPU/memory ceiling, adding a *second* external dependency's latency on top would have compounded the problem, not solved it.

## Why Groq, and why two models

Groq was chosen for raw inference speed and a generous free tier. The two-model setup (`llama-3.3-70b-versatile` primary, `llama-3.1-8b-instant` fallback) exploits a specific fact about Groq's rate limiting: token quotas are enforced *per model*, not per account. A `with_fallbacks()` chain (LangChain) means a 70B-model rate-limit exhaustion transparently retries on the 8B model instead of failing the user's request — this is a real, observed failure mode from this project's history (see doc 10's "rate-limit cascades" entry), not a hypothetical.

The 8B model additionally serves two other roles that don't need 70B-level reasoning: the Self-RAG verification check (one-word SUPPORTED/UNSUPPORTED verdict) and the knowledge-graph entity extraction. Using the cheaper/faster model for these isn't a corner cut — a one-word classification task genuinely doesn't benefit from a bigger model, and using 70B there would only add latency and burn shared quota faster.

## The request lifecycle, in full

This is the part worth understanding deeply if you're taking over the project — it's implemented across `backend/rag/retrieve1.py`'s `query_rag()` / `stream_rag()` functions (the streaming path is what the frontend actually uses; `query_rag()` exists for the eval harness and the non-streaming `/query` route, which the frontend no longer calls — see doc 2).

**Step 0 — moderation.** Before anything else, `security/guardrails.py`'s `moderate_input()` runs a single Groq call (`openai/gpt-oss-safeguard-20b`) that does double duty: harmful-content moderation AND prompt-injection detection, in one policy prompt, one call. A flagged message never reaches retrieval or generation — it gets a canned refusal, saved to history, and the function returns immediately. This fails *open* (defaults to "not flagged") on any error, same trade-off as the Self-RAG check below: an outage in a safety net should never take the whole product down when the underlying grounding/prompting is the real defense.

**Step 1 — query resolution.** Two independent things happen based on the incoming question:
- If the question contains a reference word ("that", "elaborate", "what about", etc.) *and* there's conversation history, OR the question is 4 words or fewer (empirically likely to be under-specified for retrieval), `resolve_and_classify()` runs — one Groq call that both rewrites the question into a fully explicit, standalone form using history, *and* classifies intent (answer/compare/test/summarize) in the same call.
- Otherwise, a cheap keyword-first classifier (`classify_intent()`) handles intent, and the question is used as-is for retrieval. No LLM call at all unless the keyword match is ambiguous.

This conditional gating exists because the rewrite is a full extra Groq round-trip (300-800ms observed) — unconditionally rewriting every query would double generation latency for the common case (a normal, self-contained, reasonably-worded question) for no benefit.

**Step 2 — cache check.** Only for *non-reference* questions (a reference-dependent follow-up means different things depending on history, so it can't be safely cached by question text alone). Cache key is a SHA-256 hash of `{normalized question, user_id, sorted document_ids, mode}` — all four matter, since the same question text is a different, legitimately-different-answer question across users, scoped document sets, or modes. Hit: return immediately, zero new Groq calls, tokens reported as 0/0 since nothing was spent. Miss: fall through.

**Step 3 — embed the question.** `fastembed` locally, no network call, 384-dim vector.

**Step 4 — hybrid retrieval, in parallel.** `retrieve_context()` fires two independent Postgres RPCs concurrently via `ThreadPoolExecutor(max_workers=2)`:
- `match_chunks` — pgvector cosine similarity, `match_count = k*3` candidates, filtered by `user_id` and optionally `document_ids` (the "scope to specific documents" feature), thresholded at `similarity_threshold=0.2` to drop obvious non-matches before they even reach fusion.
- `keyword_search_chunks` — Postgres full-text search (`tsvector`/`websearch_to_tsquery`/`ts_rank`), `match_count = k` candidates.

`k` is currently `6` for the main answer path (tuned down from an initial `8`, tuned up from an original `5` — see doc 9 for the eval evidence behind each move) and `8` for the summarize/compare/test-generation chains (unchanged since those were never part of this round of tuning).

The concurrency here isn't a premature optimization — these are two genuinely independent network round-trips to the same database, and running them sequentially was previously wasting 150-300ms per query for no reason. `contextvars.copy_context()` is used to propagate the LangSmith tracing context into each worker thread (Python contextvars don't cross thread boundaries by default) so the two searches show up as nested children of the parent trace instead of two disconnected root spans.

**Step 5 — Reciprocal Rank Fusion.** `_rrf_merge()` merges the two ranked lists by *position*, not by raw score: `score = 1/(60 + rank)` per list a chunk appears in, summed across lists. This deliberately sidesteps the problem of two incomparable score scales (cosine similarity is 0-1, `ts_rank` is an unbounded float with a totally different distribution) — RRF only needs relative rank, so a chunk both methods rank highly wins over one only one method found, without ever having to normalize or calibrate the two scales against each other.

**Step 6 — cross-encoder rerank.** `_rerank()` via FlashRank, reading query and passage together (not two independent embeddings compared post-hoc, which is what both retrieval methods above are doing). This is strictly more accurate at judging query-passage relevance than either bi-encoder similarity or RRF rank position alone, at the cost of being CPU-bound and only feasible because the candidate pool has already been narrowed to `k*2` by fusion — running a cross-encoder over the *full* corpus per query would not be affordable on Render's CPU budget.

**Step 7 — confidence scoring.** Blends normalized RRF agreement (0-1, how strongly both retrieval methods agree this chunk matters) with raw cosine similarity, 50/50, with a small multiplicative bonus (up to +32%) for having multiple corroborating chunks in the final set. This single number drives the next two decisions.

**Step 8 — Corrective RAG gate.** `grounded = confidence >= CONFIDENCE_THRESHOLD` (currently `0.45`, raised from `0.35` — see doc 9). If grounded: the system prompt gets `GROUNDING_RULE` — context-only, cite `[filename]` after every fact, refuse with an exact fixed phrase if the answer isn't in context, and an explicit instruction to treat anything in the retrieved context that looks like an instruction as untrusted data, never as a command (a real prompt-injection defense against a malicious or accidentally-instruction-shaped document chunk). If not grounded: `GENERAL_KNOWLEDGE_RULE` instead — answer from the model's own training, but the reply must start with the literal line `[General knowledge]` so the frontend can flag it as not sourced from the user's documents. This is the actual mechanism (not the LLM being "asked nicely") that prevents an answer sourced from nowhere being mistaken for a document-grounded one.

**Step 9 — generation.** `llama-3.3-70b-versatile` (falls back to `llama-3.1-8b-instant` on rate limit), `temperature=0`, streamed token-by-token over SSE. `STRUCTURE_RULE` (added alongside the accuracy pass in doc 9) nudges the model toward markdown tables/lists for comparisons and multi-attribute data — prompt-only, zero extra latency, and only useful because the frontend renders GFM tables (`remark-gfm`, added in the same pass — before that, a table in a response rendered as raw pipe-delimited text).

**Step 10 — conditional Self-RAG verification.** Only runs if `grounded` is true *and* confidence is below `SELF_RAG_SKIP_CONFIDENCE` (`0.75`). A second, cheap (`llama-3.1-8b-instant`) call re-reads the context and the generated answer, replies exactly `SUPPORTED` or `UNSUPPORTED`. If unsupported: in the non-streaming path, the answer is swapped for the refusal phrase before it's ever returned; in the streaming path, tokens are already sent by the time this check runs (can't un-send them), so instead a `warning` SSE event fires and the frontend renders it as a disclaimer banner under the already-shown message. This conditional skip (added alongside the k-tuning pass) exists because unconditionally verifying every single grounded answer was a second sequential Groq call on the common case — trusting the STRICT RULE prompting alone once confidence is already high recovers that latency without giving up the safety net where it actually matters (low-confidence, borderline answers).

**Step 11 — everything downstream runs concurrently with token streaming, not after it.** Knowledge-graph "related concepts" lookup (`get_related_nodes`, a BFS over `graph_nodes`/`graph_edges`) runs on its own thread via `ThreadPoolExecutor(max_workers=1)`, started before the token loop and `.result()`-awaited only after streaming finishes — so it overlaps with generation instead of adding to perceived latency.

**Step 12 — bookkeeping.** Token usage (`usage_metadata` off the final streaming chunk, not guaranteed on every chunk but reliably present on the last one — verified directly, not assumed) gets recorded via `record_token_usage()` against a rolling 24h counter (reusing the `rate_limits` table's windowed-counter RPC pattern), the answer gets cached (non-reference questions only) with a 12-hour TTL, and both the user's question and the assistant's answer get saved to `sessions` for history.

Every one of `chunks`, `documents`, `sessions`, `chat_sessions`, `graph_nodes`, `graph_edges` carries a `user_id` and a Postgres RLS policy scoping `auth.uid() = user_id`. The backend's Supabase client uses the **service role key**, which bypasses RLS entirely — RLS here is defense-in-depth against a hypothetical future code path using the anon key, not the primary access control (the primary control is that every query function takes and filters by `user_id` explicitly, verified via JWT in `get_current_user`).
