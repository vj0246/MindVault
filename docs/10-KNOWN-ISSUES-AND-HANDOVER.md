# Known issues and operational handover

## Dead code to clean up

- **`backend/rag/retrieve.py`** (165 lines) — legacy FAISS/Ollama implementation, never imported. Safe to delete (doc 6).
- **`frontend/components/GraphExplorer.tsx`** (649 lines) — a full, unused, canvas-based alternative to `GraphPanel.tsx`. Never imported. Either delete it or deliberately swap it in for `GraphPanel` (doc 7) — don't leave two competing implementations where only one is discoverable.
- **`rag/memory.py`'s `get_history_for_prompt`** — only caller was the dead `retrieve.py`. Effectively dead itself, still exported from an otherwise-live file (doc 3).

## Not yet done (see main README's "What I have not done" for the user-facing version)

- **No vector index (HNSW/IVFFlat) on `chunks.embedding`.** Fine today; will become the dominant latency source once chunk counts grow past a few thousand, since similarity search risks a sequential scan (doc 8).
- **No HyDE.** Query rewriting for short/vague questions is a cheaper partial substitute, not equivalent (doc 9).
- **Hyperparameters not exhaustively grid-searched** — reasoned defaults validated by before/after comparisons only, due to real Groq free-tier quota constraints (doc 9).
- **No queue/worker separation** — background work runs in-process (doc 9). Fine at current scale.
- **Frontend is one 1479-line file** — the single highest-value pure refactor available (doc 7).

## Operational reality: Render's free tier

512MB RAM, a single shared vCPU, cold starts, and — this is the one that isn't obvious until you hit it — **the platform will kill and restart the container if it decides the service is unresponsive**, which includes CPU starvation from your own code, not just an actual crash. Three separate real incidents trace back to this one constraint:

1. **Ingestion hanging indefinitely with zero error anywhere** — traced via Supabase's own API request logs (not application logs, which showed nothing) to a `POST /documents` insert firing and then genuinely nothing happening for hours. Root-caused to inline synchronous ingestion exceeding Render's own request timeout, killing the connection before Python's exception handling ever ran. Fixed by moving ingestion to a background task (doc 2) — but that fix alone wasn't sufficient, see below.
2. **A large document still hanging even after the background-task fix**, this time traced via Render's own application logs (the ones that show container restarts) to the semantic chunker's per-section embedding calls pegging the single weak vCPU hard enough that Render's platform-level health check decided the whole service was unresponsive and force-restarted the container *mid-upload*. Fixed by skipping semantic chunking entirely for documents over 20,000 characters (doc 3).
3. **A different large document still hanging after *that* fix**, traced to `embed_and_store`'s single batched embedding call spiking transient memory enough to trip an OOM kill for documents with enough chunks. Fixed by sub-batching embedding calls at 16 texts per call (doc 3).

**The pattern worth internalizing**: on this hosting tier, "the request hung" rarely means "there's an infinite loop in the code." It usually means something CPU- or memory-bound ran long/heavy enough that the *platform* intervened, and the evidence is in Render's own logs or Supabase's API logs, not in whatever your own `print()` statements captured — because the process got killed before those statements' output ever left the box, or the crash happened somewhere logging itself couldn't run (see the Unicode-crash incidents below). If a hang shows real evidence of "nothing happened after step N," suspect the platform, not just the code, and check for a restart banner in the logs around that timestamp before assuming it's a genuine deadlock.

## Recurring gotcha: non-ASCII characters crashing on Windows

Development happens on Windows; the deployment target (Render) is Linux. Windows' console defaults to cp1252, not UTF-8. This project hit the *exact same bug class* three separate times in three different files: an arrow character in an `ingest.py` log line, an em-dash in a `migrate_embeddings.py` log line, and — worst of all — `tabulate`'s Unicode box-drawing table format in `eval/evaluate.py`, which crashed *after* a full expensive RAGAS run completed, losing all the results because the crash happened before the save step. If you add a new `print()` or console-output call anywhere in this backend, assume it needs to be ASCII-only, and if you're building anything that computes an expensive result before displaying it, save the result to disk *before* any display/formatting step that could itself fail — never let a "just for display" step be able to cost real, already-computed work.

## The ingestion zombie-row saga, end to end

Worth reading as one story since it happened in stages, each fix uncovering the next real cause:

1. Original bug: silent `chunk_count: 0` rows on parse failure (e.g. a scanned PDF with no extractable text) — fixed by checking extracted text length before chunking and raising a clear error, with the row deleted on failure.
2. Then: inline synchronous ingestion timing out against Render's own request limit, leaving `chunk_count=-1` rows with no chunks and no error — fixed by moving to a background task (`BackgroundTasks` + a bounded `ThreadPoolExecutor` with a 5-minute `future.result(timeout=...)`, doc 2).
3. Then: the background task itself could still hang forever inside `ingest_document()` with no exception ever raised (see the Render CPU-starvation incident above) — the 5-minute timeout *should* have caught this via `concurrent.futures.TimeoutError`, and mostly does, but the underlying causes (CPU starvation, then OOM) needed their own fixes at the source (large-doc chunking bypass, embedding sub-batching) because a stuck platform-killed container doesn't necessarily give Python a chance to hit the timeout cleanly — the whole process can die first.
4. Along the way, a **ReDoS-shaped regex** in table detection (`(?:\|.+\|\s*\n)+`, nested unbounded quantifiers over pipe-heavy lines) was found and replaced with a linear line-scan — not confirmed as *the* cause of any specific incident (a synthetic adversarial-input reproduction attempt didn't actually trigger a hang), but a real, provable latent risk removed regardless, and worth knowing about since CPython's `re` engine doesn't release the GIL mid-match — a stuck regex anywhere in this codebase could in principle starve the entire process, not just the one request.

If a future upload silently gets stuck again, the diagnostic order that actually worked here was: (1) check Supabase API logs for the exact request sequence around the stuck document's `uploaded_at` timestamp, (2) check Render's application logs for a restart banner around the same window, (3) only then suspect application-level logic. Guessing at the code before checking those two log sources wasted real time across this project's history.

## Other real production incidents (see main README's "Production engineering" section for the user-facing versions)

- **Memory leaks from per-call client instantiation** (Supabase + Groq clients) → OOM-driven 502s. Fixed with module-level singletons everywhere (`rag/db.py`, `security/groq_keys.py`, `retrieve1.py`'s `get_llm`/`get_verify_llm`, `graph/extractor.py`'s `_EXTRACTOR_LLM`). If you ever see intermittent 502s with no clear pattern, check for a new per-call client instantiation somewhere before looking elsewhere.
- **Groq per-model rate-limit cascades** — fixed via `.with_fallbacks()` (70B → 8B) plus multi-key fallback (`call_with_key_fallback`). If a burst of legitimate traffic (or your own eval runs against the same key) starts failing requests with visible `429`s reaching the user, check whether both fallback layers are actually configured (a single `GROQ_API_KEY` with no fallback still works, but has no headroom).
- **PDF export failing with an unreadable error** — root cause was a missing `fpdf2` dependency on the server, but the *actual bug that had to be fixed first* was the frontend's `responseType: 'blob'` masking the JSON error body. If a future feature adds another blob-typed response (file downloads, binary exports), apply the same JSON-content-type detection pattern from `exportSessionPDF` proactively rather than rediscovering this the hard way.
- **React re-render performance at scale** — `React.memo` on the message component wasn't sufficient alone because a callback prop passed into it was recreated every render, silently defeating the memoization; needed `useCallback` too. If chat performance regresses with many messages, check for a newly-introduced inline function/object prop passed into a memoized component before assuming the memoization itself is broken.

## Where to look first for common categories of future bugs

- **"Answer quality regressed"** → check `backend/eval/results_final/` for the most recent run vs. history, and doc 9's tuning-decision list for anything that might interact with a recent change.
- **"Upload/ingestion is stuck or slow"** → doc 2 and this doc's "Render free tier" section above; check Supabase API logs and Render application logs, in that order, before the code.
- **"Latency feels worse"** → doc 1's request-lifecycle walkthrough lists every point where an LLM call happens and why it's conditional or not; check whether a conditional gate (query rewrite, Self-RAG verify) is firing more often than expected for the traffic pattern in question.
- **"Something crashed with no clear error on Windows but works fine when actually deployed"** → check for a non-ASCII character in a `print()`/logging call in the changed code, per the recurring gotcha above.
