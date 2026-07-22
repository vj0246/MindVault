# `backend/app.py` — the FastAPI application

843 lines, every route in one file. No routers/blueprints split — at this size it's still navigable with Ctrl+F, and splitting it wasn't a priority relative to actual correctness/security work. If this grows much further, splitting by resource (`routes/documents.py`, `routes/sessions.py`, `routes/query.py`) would be the natural next step.

## Startup: constants, CORS, security headers

Lines 29-59 define every rate limit and input-size cap as a module-level constant — `QUERY_RATE_LIMIT=100`, `UPLOAD_RATE_LIMIT=20`, `SESSION_RATE_LIMIT=30`, `EXPORT_RATE_LIMIT=30`, `MEMORY_RATE_LIMIT=60`, `SHARE_RATE_LIMIT=30`, `SEARCH_RATE_LIMIT=30` (all per-user-per-hour; per-IP is `security/rate_limit.py`'s `IP_MULTIPLIER=3` times these, a secondary backstop against one IP running many accounts), plus `MAX_UPLOAD_BYTES=25MB`, `QUESTION_MAX_CHARS=8000`, `MEMORY_NOTE_MAX_CHARS=500`, and the preference field caps. Every one of these exists because client input reaching either an LLM prompt (cost/latency abuse) or a DB row (storage abuse) needs a ceiling — this is the direct output of the security remediation pass (doc 9).

CORS (`_DEFAULT_ORIGINS` + `ALLOWED_ORIGINS` env var, comma-separated) is an allowlist, not a wildcard — deliberate even though auth is Bearer-token (not cookies), so a wildcard wouldn't have been a credential-theft vector on its own; pinning to known origins is still tighter.

`security_headers` middleware sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, `Strict-Transport-Security`, `Permissions-Policy: camera=(), microphone=(), geolocation=()` on every response — a standard hardening pass, not tailored to any specific finding.

## Auth: `get_current_user`

A FastAPI dependency, not middleware — every route that needs auth declares `user=Depends(get_current_user)`. It takes the `Authorization: Bearer <jwt>` header, calls `supabase.auth.get_user(token)` (verifies the JWT against Supabase Auth), and returns the user object or raises `401`. There is no session/cookie state on the backend at all; every request is independently authenticated by its bearer token.

## `/upload` and the background ingestion pipeline

This is the most-rewritten code path in the project's history (see doc 10 for the full incident timeline). Current shape:

1. Validate extension against an allowlist, sanitize the filename to strip path-traversal characters (`re.sub(r"[^A-Za-z0-9._-]", "_", ...)` on the basename only — a client-supplied `../../etc/passwd`-style filename can never escape `data/docs/`).
2. Check `Content-Length` as a cheap early rejection (spoofable, not the real guard) then actually cap the read at `MAX_UPLOAD_BYTES + 1` bytes (the real guard — a lying or missing header can never bypass this).
3. Write to `data/docs/{uuid}_{safe_filename}` on local disk (ephemeral — gone on redeploy, which is fine, the file only needs to survive long enough for the background task to read it once).
4. Insert a `documents` row with `chunk_count=-1` (the "still processing" sentinel) via `log_document()`.
5. Schedule `_ingest_and_finalize` as a `BackgroundTasks` job and return `{"status": "processing", "document_id": ...}` **immediately** — the HTTP response is never at risk of Render's own request timeout, which is precisely the bug this architecture replaced (see doc 10: ingestion used to run inline, and a slow document could get killed mid-request by Render before Python's own exception handling ever ran, leaving a permanent `chunk_count=-1` zombie row with no error anywhere).

`_ingest_and_finalize` (module-level function in `app.py`, not in `rag/ingest.py` — it orchestrates ingestion + the finalization side effects, `ingest_document()` itself is the pure parse/chunk/embed logic):

- Submits `ingest_document(...)` to a dedicated `ThreadPoolExecutor(max_workers=4)` and waits on `future.result(timeout=INGEST_TIMEOUT_SECONDS)` (`300` seconds). This exists because a real production incident showed ingestion can hang *indefinitely* inside the parse/chunk/embed pipeline with zero error surfaced anywhere (confirmed via Supabase API logs: `POST /documents` fires, then nothing, for hours) — Python cannot forcibly kill a stuck thread, so this bounds how long the request *waits* on it instead. Past the timeout, the wait is abandoned, the row is deleted, and the abandoned thread eventually finishes or dies on its own without holding up anything user-visible.
- On success: updates `chunk_count`, calls `invalidate_has_chunks(user_id)` (so the very next query doesn't see a stale "no documents" cached result — see `rag/cache.py`), then runs `_build_graph_for_chunks` on the first 10 chunks (entity/relationship extraction, one Groq call per chunk, best-effort — a single chunk's extraction failure is caught and logged, never aborts the whole graph build).
- On *any* exception (including the timeout above): logs the full traceback, then deletes the document row. The traceback print itself is wrapped in its own nested try/except — a genuinely real bug found during this project's history: a non-ASCII character in a caught error message raised `UnicodeEncodeError` on Windows' cp1252 stdout, which *skipped the cleanup delete* because the crash happened inside the except block's own logging call, before reaching the delete. That's the exact zombie-row bug this function exists to prevent, self-inflicted by its own error handling. Fixed by never letting the logging itself be a single point of failure.

## Query routes

Four query-shaped routes, not one, because streaming and attachments both need their own request/response shape:

- `POST /query` — non-streaming, JSON in/out. Moderates input first; on a flag, saves a canned refusal to history and returns immediately without ever calling `query_rag`. Otherwise calls `query_rag()` (see doc 3), saves both the user question and the assistant answer to session history, returns the full result including `tokens` (message-level + running daily total/percentage) and `answer_type` (`grounded` vs `general_knowledge`). **Not currently called by the frontend** — `handleSend` in `page.tsx` always streams (see doc 7); this route exists for the eval harness (`query_rag` is called in-process by eval, but this HTTP route is the same logic path exposed for any other consumer) and as a non-streaming fallback if ever needed.
- `POST /query/stream` — SSE. Same moderation-first check, but on a flag it emits the refusal as SSE events (`meta` → `token` → `done`) rather than a JSON body, so the frontend's single SSE-consuming code path handles both the blocked and normal cases identically. Delegates to `stream_rag()` (doc 3) and re-parses the `done` event server-side just to extract `full_answer` for saving to history — a small inefficiency (double JSON parse) traded for keeping the history-save logic in one place regardless of what `stream_rag` yields.
- `POST /query-with-attachment` / `POST /query-with-attachment/stream` — the one-off attachment flow (image or document attached to a single question, not permanently stored in the vault). Multipart form data, not JSON, since a file is involved. Loads the attachment via `load_image_via_groq` (images) or `load_document` (everything else), caps extraction at `ATTACHMENT_CHAR_BUDGET=6000` characters **during** the page-by-page accumulation loop, not after joining the whole document — a real fix (doc 10) for a bug where a multi-hundred-page OCR'd PDF attachment cost CPU proportional to its full length even though only the first ~6000 characters were ever used downstream.

## Document management routes

`GET /documents`, `PATCH /documents/{id}/folder`, `DELETE /documents/{id}` — all straightforward CRUD scoped by `user_id`, delegating to `metadata/tracker.py` (doc 6). `DELETE` calls `invalidate_has_chunks` afterward for the same reason `/upload` does on success — a user's "do I have any documents" cached answer must never lag behind a change that just happened.

## Export routes

`POST /export` supports `format: "markdown"` (returns a formatted string, no external dependency) and `format: "pdf"` (via `_build_session_pdf`, using `fpdf2`). The PDF builder handles both the old and new `fpdf2` API (`XPos`/`YPos` vs `ln=True`) via a `cell()` wrapper and a runtime `ImportError` probe — defensive against a `fpdf2` version bump changing its API, which is exactly what happened once already (a missing `fpdf2` dependency on the server caused PDF export to fail silently — see doc 10's "frontend bug that hid its own root cause" entry, which was actually a backend dependency issue masked by a frontend error-handling bug).

Session labels in exports are always `Session #N` (the clean sequential number from `list_chat_sessions`), never the raw UUID — nobody reading an exported PDF should have to make sense of a UUID.

## Session, preferences, memory, share routes

Thin wrappers over `rag/memory.py` (doc 3) — `list_chat_sessions`, `create_chat_session`, `rename_chat_session`, `delete_chat_session`, `get/save_user_preferences`, `list/add/delete_memory_note`, `generate/revoke_share_token`, `get_shared_session`. One notable check: `POST /preferences` moderates the user's custom `system_prompt` field before saving it — because that field gets spliced into every future answer's system prompt (`_preference_hint` in `rag/retrieve1.py`), a one-time unmoderated save would be a persistent jailbreak vector, not a single bad message.

`GET /share/{token}` is the one route with no auth dependency at all — it's the public, read-only shared-conversation view. Rate-limited per-IP (there's no `user_id` to key on for an anonymous visitor), by passing the IP as both the "user" and "IP" identifier to `enforce_rate_limit` — both checks collapse into one effective per-IP limit.

## Graph routes

`GET /graph/{topic}` (BFS from a starting concept) and `GET /graph` (the whole graph) — both delegate to `graph/store.py` (doc 5), which as of a recent perf pass caches the raw node/edge lists per-user with a 60-second TTL, invalidated on every graph write.

## The catch-all OPTIONS handler

`@app.options("/{rest_of_path:path}")` — a single handler answering CORS preflight for every route, checking the `Origin` header against `ALLOWED_ORIGINS` and echoing it back only if it matches. Exists because FastAPI's default CORS middleware handles preflight automatically in most setups, but this project apparently needed (or once needed) an explicit fallback; it's harmless either way since it only ever returns an empty body with CORS headers.
