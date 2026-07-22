# `backend/rag/` — the core pipeline

This is the module that matters most. Doc 1 already walks the full request lifecycle end to end; this doc covers implementation details per file that the lifecycle walkthrough skipped, plus the chains that aren't part of the main answer path.

## `embedder.py` (13 lines, the whole file)

```python
from fastembed import TextEmbedding
EMBED_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
```

One module-level singleton, imported by both `ingest.py` and `retrieve1.py`. The comment in the file is explicit about why: creating a second `TextEmbedding` instance would load a second copy of the model into memory, which on a 512MB Render box is the difference between running and OOM-killing on startup.

**Model history**: this was `sentence-transformers/all-MiniLM-L6-v2` until an accuracy pass (see doc 9) swapped it for `BAAI/bge-small-en-v1.5` — same 384-dim output (no pgvector schema change needed), stronger retrieval benchmarks. The swap required re-embedding every existing chunk (`migrate_embeddings.py`, doc 6) since old and new vectors live in different, incompatible embedding spaces — you cannot mix them in one nearest-neighbor search and get meaningful results.

**A subtlety worth knowing if this model changes again**: `fastembed`'s `TextEmbedding` class exposes `embed()`, `query_embed()`, and `passage_embed()` as (in principle) distinct methods — some BGE-family models are trained with an asymmetric instruction prefix that should only be applied to the query side, not the passage/document side. This was checked empirically for `bge-small-en-v1.5` specifically (not assumed): all three methods produce byte-identical output for this model in the installed `fastembed` version, matching `fastembed`'s own model registry metadata, which notes prefixes are "not so necessary" for v1.5 specifically (as opposed to the older `bge-small-en`, which is marked "necessary"). If this model is ever swapped for one where the asymmetry actually matters, the codebase does **not** currently distinguish query vs. passage embedding calls anywhere — every call site just uses `.embed()`. That would need to change.

## `ingest.py` (385 lines)

The parse → chunk → embed → store pipeline, called from `app.py`'s background task.

**`load_document(file_path)`** dispatches by extension: `PyPDFLoader` (PDF), `TextLoader` (txt/md), `Docx2txtLoader` (docx/doc), or `load_image_via_groq` (images — sends the image to Groq Vision, `meta-llama/llama-4-scout-17b-16e-instruct`, asks for a detailed transcription/description, wraps the response as a single `Document`).

**Smart chunking** (`chunk_documents`) is a multi-stage pipeline, not a single splitter call:
1. Combine all loaded pages into one text blob.
2. **Size gate**: if the combined text exceeds `LARGE_DOC_CHAR_THRESHOLD` (20,000 chars), skip straight to the cheap fixed-size fallback splitter (see below) — this exists because of a real, traced production incident: semantic chunking's per-section embedding calls were CPU-heavy enough on Render's single weak vCPU to peg it hard enough that Render's own platform-level health check decided the service was unresponsive and force-restarted the container *mid-ingestion*. This wasn't a Python-level hang — the platform killed the process. Below the threshold, semantic chunking proceeds normally; above it, quality is traded for actually finishing.
3. **Table/code protection** (`_protect_blocks`): tables are detected via a hand-written line-by-line scan (`_find_table_blocks`), *not* a regex — the original implementation used a regex with nested unbounded quantifiers over pipe-delimited lines (`(?:\|.+\|\s*\n)+`), a textbook catastrophic-backtracking shape. Documents heavy in `|` characters (shell pipe examples, ASCII diagrams — exactly what technical/CS notes tend to contain) are exactly the kind of input that can trigger pathological backtracking, and — worse — CPython's `re` engine doesn't release the GIL mid-match, so a stuck regex can starve *every other thread in the process*, including a thread waiting on a timeout. The line-scan replacement is O(n) and structurally cannot blow up regardless of input. Fenced code blocks are still protected via a regex (`` ```[\s\S]*?``` ``, non-greedy, no nested-quantifier risk).
4. **Section splitting** (`_parse_sections`) via `HEADING_RE` — matches markdown headers, ALL-CAPS titles, and numbered sections (`1.`/`1.1`/`1.1.1`), all with bounded quantifiers (no ReDoS risk here).
5. **Semantic splitting** (`_semantic_split`) per section: embeds every sentence (fastembed, batched), walks the list splitting wherever cosine similarity between consecutive sentences drops below `threshold=0.65` (a genuine topic-shift signal) *or* the accumulated chunk exceeds `max_chars=900` (a hard cap, tuned up from `600` alongside the fixed-splitter chunk-size increase — see doc 9). Below `min_chars=200`, a similarity drop alone won't force a split — avoids pathologically tiny chunks.
6. **Fallback splitter**: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` (tuned up from `500/50` — doc 9) — used both for documents over the large-doc threshold and for the rare case where smart chunking genuinely finds zero chunks in a normal-sized document.

**`embed_and_store`**: embeds all final chunks and inserts them in batches of 50 rows (Supabase insert batching, unrelated to the embedding batching below) into the `chunks` table. The embedding step itself is sub-batched at `EMBED_BATCH_SIZE=16` — not one giant `EMBED_MODEL.embed(all_texts)` call — because embedding an entire large document's chunks in a single call was traced to spiking transient memory enough to get the container OOM-killed on bigger uploads (a second, distinct incident from the CPU-starvation one above; same root constraint — 512MB RAM with the model already resident — different mechanism). Sub-batching caps peak memory regardless of how many chunks a document produces.

## `retrieve1.py` (1076 lines)

The file doc 1's lifecycle walkthrough is drawn from. Implementation notes doc 1 didn't cover:

**LLM singletons** (`get_llm`, `get_verify_llm`) — both cached by a dict keyed on temperature (only `0.0` and `0.15` are ever used), each value being a LangChain `.with_fallbacks()` chain across every configured Groq key × the two-model fallback (70B primary, 8B fallback) for `get_llm`, or just the 8B model across keys for `get_verify_llm` (the verification check doesn't need 70B-level reasoning for a one-word verdict). The comment on this is explicit: creating a fresh `ChatGroq` per call opens a fresh `httpx` connection pool that doesn't clean up fast enough under load, and this was a real, observed cause of OOM-driven 502s in production (doc 10) before the fix.

**The reranker singleton** (`_get_reranker`) — same lazy-singleton pattern, `flashrank.Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")`. `_rerank()` fails soft: if FlashRank throws for any reason, it falls back to the RRF-ranked order truncated to `top_k`, rather than propagating the exception and breaking the whole query.

**`build_retrieval_chain`** wraps the main answer path as a closure (`run_chain`) rather than a plain function — `build_retrieval_chain(mode, user_id, document_ids)` precomputes the preference hint and the LLM instance once, and returns a function that does retrieval + prompt selection + the actual call only when invoked with the resolved question. This is the function `query_rag()` (the non-streaming path) calls.

**`stream_rag`** is a generator, not a plain function — it's the one the frontend actually consumes. It handles cache hits by pseudo-streaming the cached answer in 4-word groups (so the frontend's token-accumulation logic needs zero special-casing for "this came from cache" vs "this came from a live generation" — same event shape either way, just faster). It handles the router decision for reference-dependent follow-ups (`decision == "history"` — answered from conversation history alone, no retrieval, when the router determines the question can be resolved without hitting the vault at all) and the three non-answer intents (`compare`/`test`/`summarize`) by running those chains non-streaming and yielding one `meta` + one `done` event instead of a real token stream (those chains aren't built to stream — a reasonable simplification since they're structured, multi-part outputs where partial rendering wouldn't add much value anyway).

**`summarize_chain` / `comparison_chain` / `test_generator_chain`** — all three share a shape: retrieve at `k=8` (untouched by the k-tuning pass, which only touched the main answer path's `k`), build a task-specific prompt, call the LLM non-streaming with `temperature=0.15` (slightly creative — these are synthesis tasks, not strict fact lookup, so a small amount of temperature is intentional here even though the main answer path uses `0.0`). `comparison_chain`'s prompt explicitly offers a markdown-table format for attribute-based comparisons (part of the structured-output pass, doc 9) alongside a similarities/differences/key-insight fallback structure for comparisons that don't reduce cleanly to a table.

**`_prepare_attachment_answer`** — shared setup used by both `query_with_attachment` and `stream_with_attachment`, since retrieval + prompt-input construction is identical between the invoke() and stream() variants and only how the LLM is actually called differs. Builds two *separate*, clearly-labeled sections in the prompt (attachment content vs. vault context) — mixing them into one blob previously caused the model to confuse "what does this attached image show" with "what does my document vault say," a real, observed failure mode this structure fixes.

**`resolve_and_classify`** — the query-rewrite chain, one Groq call, returns both a rewritten question and a classified intent in a single structured-text response (`RESOLVED: ...` / `INTENT: ...` parsed by prefix matching, not JSON — simple enough that a JSON schema would be overkill, but fragile if the model ever deviates from the expected format; there's a keyword-based intent override applied *after* parsing as a safety net for exactly that case).

**`get_related_nodes`** import from `graph/store.py`, used both inline (non-streaming path) and via a background thread (streaming path, doc 1 step 11).

## `cache.py` (97 lines)

Two independent caching mechanisms living in one file:

1. **`has_any_chunks`** — an in-process (not database-backed) TTL dict cache, 60 seconds, answering "does this user have any documents at all." This is checked on every single query just to decide whether to show the "no documents uploaded" message, and it rarely changes, so a 60-second in-memory cache turns a guaranteed-every-message Supabase round trip into roughly one per minute per user. `invalidate_has_chunks(user_id)` is called immediately after any upload or delete completes, so a cache hit can never mask a just-happened change.
2. **`get_cached` / `set_cached`** — the actual query-result cache, backed by the `query_cache` Postgres table (not in-process — this one needs to survive across requests/processes and be shareable), 12-hour TTL. `make_cache_key` hashes `{question, user_id, document_ids, mode}` together — all four are load-bearing, since any one of them changing legitimately changes the correct answer.

Both cache read/write functions fail *open*: a Supabase error on cache lookup is treated as a cache miss (falls through to the real pipeline), and a cache-write failure is logged and swallowed, never allowed to break an already-generated answer.

## `token_usage.py` (35 lines, the whole file)

`DAILY_TOKEN_BUDGET = 100_000` — a **self-imposed fairness cap**, explicitly not Groq's actual account quota (which isn't queryable via the API and varies by model/tier). `record_token_usage` reuses the `rate_limits` table (the same one `security/rate_limit.py` uses for request-rate limiting) via the `increment_token_usage` RPC — a windowed counter, same increment-and-return-total pattern as `check_rate_limit`, just accumulating token counts instead of request counts, in the same table, keyed by a different prefix (`tokens:user:{id}` vs whatever `security/rate_limit.py` uses). Fails open: any error, or no `user_id`/`total_tokens` to record, returns `{0, 0}` rather than breaking the response that already generated successfully.

## `memory.py` (253 lines)

Everything that isn't retrieval: message history (`sessions` table — raw turn-by-turn), chat session metadata (`chat_sessions` table — naming, sharing, the clean sequential `#N` numbering shown to users), user preferences (name/tone/priorities/custom system prompt/theme), and long-term memory notes (manual, user-added facts — deliberately no auto-extraction; the user controls exactly what's remembered, there's no background process scanning conversations for "facts to remember").

Worth flagging: `search_messages` escapes `%`/`_` in the user's search query before using it in a Postgres `ILIKE` pattern (`\\`, `\%`, `\_`) — without this, a literal `%` or `_` typed by the user would behave as a SQL wildcard instead of a literal character, which isn't a security hole (RLS + `user_id` filter still scope it) but would produce confusing, wrong search results.

`get_history_for_prompt` compresses long conversation history for prompt injection: histories of 6 messages or fewer are used verbatim; longer ones get the older messages summarized by an LLM call (2 sentences max) and only the last 4 messages kept verbatim. **Confirmed dead code in practice**: grepping the whole backend, its only caller is `rag/retrieve.py` — the legacy, never-imported FAISS/Ollama file (see doc 6). `retrieve1.py` (the live pipeline) uses a different, simpler function (`format_history`) instead. The function itself is fine and still exported from `memory.py`, it just has no live caller.
