# `frontend/` — Next.js app

## The one-file-app decision

`app/page.tsx` is 1479 lines and contains the entire application: every modal, the sidebar, the chat view, all state management, all API-calling logic. There is no component-per-file split for anything inside it (the only extracted components are `GraphPanel.tsx` and the dead `GraphExplorer.tsx`, both graph-visualization-specific and genuinely reusable/self-contained in a way the rest of the UI isn't).

This is worth naming explicitly as a tradeoff, not silently working around it: at this size, `page.tsx` is past the point where a fresh contributor can hold the whole file in their head, and splitting `Sidebar`, `PreferencesModal`, `MemoryModal`, `DocumentManagerModal`, and `EmptyState` (all currently defined as separate functions *within* `page.tsx`, just not in separate files) out into their own files would be a pure win with basically no behavior change required — they already don't share internal state with each other, only props passed down from `Home`. This wasn't done because the app grew incrementally, feature by feature, in a single long-running session-based workflow, and splitting was never the highest-priority next step relative to shipping working features and fixing production bugs. It's the single most valuable pure-refactor available in this codebase for a new contributor to tackle early.

## Top-level types (lines 37-99)

`Mode` (`student`/`lawyer`/`developer`/`default` — the tone-of-voice selector, maps directly to `_MODE_VOICE` in `retrieve1.py`), `Intent` (`answer`/`compare`/`test`/`summarize`), `Message` (chat bubble state — notably carries `answer_type`, `unverified` (set true if a Self-RAG warning event fired), and `tokens`), `Doc`, `PreferencesState` (aliases the `Preferences` type imported from `lib/api.ts`), `MemoryNote`, `ChatSession` (carries the clean sequential `number` field from the backend), `ChunkSource`, `Toast`, `GraphData`.

## `getDynamicSuggestions(docs)` (line 116)

Generates the 4 suggestion chips shown on the empty-chat landing state. **Real bug fixed in this project's history**: originally used `docs[0]`/`docs[1]` directly for 3 of the 4 suggestions — since `docs` is ordered most-recently-uploaded-first (matches `get_all_documents`'s `uploaded_at desc`, doc 6), this meant every suggestion but one always pointed at the newest 1-2 uploads regardless of how large the rest of the vault was. Fixed to a `pick(i) => names[i % names.length]` helper spreading across indices 0-3 (wrapping via modulo for vaults with fewer than 4 documents), so the landing page actually reflects the whole knowledge base.

## `EmptyState` component

The "what do you want to know?" landing view, shown when there are no messages yet. Renders the dynamic suggestion chips and an empty-vault-specific hint.

## `Sidebar` component (the largest single component in the file)

Tabbed as of a UI-decongestion pass: **Documents** and **Chats** tabs (a segmented control), replacing an earlier design where the upload zone, chat search, chat list, and document list were all stacked in one column — cramped enough on a narrow sidebar that most content was fighting for the same vertical space. Each tab now gets the full remaining height. Chat search (searches message *content* across all sessions, via `/sessions/search`) lives under the Chats tab specifically because it searches conversations, not documents — it didn't belong permanently visible in the Documents view.

The Documents tab: an upload dropzone (drag-and-drop or click-to-browse), a "Find a document…" filter input (only shown once there are more than 4 documents — not worth the UI weight below that), the document list itself (grouped by folder, `Uncategorized` sorted last), each row showing filename + chunk count + upload date, or `⟳ Processing…` in place of chunk count while `chunk_count === -1` (the backend's processing sentinel, doc 2). Folder-tagging and delete used to be inline icon buttons on every row — found to be too cramped to use reliably at sidebar width, so both were moved into a dedicated `DocumentManagerModal` (below), reached via a "Manage" button, leaving the inline rows simplified to just selection + status.

The daily token-usage meter (a thin progress bar, color-shifting from accent → amber → red as usage climbs) lives at the top of the sidebar regardless of which tab is active — it's global usage-transparency state, not tab-specific.

## `PreferencesModal`

The onboarding/settings modal — name, tone, priorities (multi-select), a custom system prompt (moderated server-side before saving, since it gets spliced into every future answer — see doc 2), and theme (Light/Dark). Shown automatically on first login if the backend reports no saved preferences (`onboardingRequired`), reachable afterward via a "Preferences" button in the sidebar.

## `MemoryModal`

Long-term memory notes — manual, user-added facts, no auto-extraction (see doc 3's note on `memory.py`). A simple add/list/delete UI over `/memory`.

## `DocumentManagerModal`

The dedicated document-management surface split out of the cramped inline sidebar rows: a searchable list of every document, each row showing a labeled "Folder" button (opens an inline text input) and a labeled red "Delete" button (with a `window.confirm` before actually calling `deleteDocument`) — deliberately more spacious and readable than the compact sidebar rows, since this is the "I actually need to manage things carefully" surface, not the "glance and select" one.

## Theme handling — a genuinely subtle bug, fixed twice

**First pass**: the CSS default (`:root`, no `data-theme` attribute) is light; `[data-theme="dark"]` in `globals.css` overrides it. On every page load, the page would render light, then flip to dark 1-3 seconds later once `/preferences` resolved — a visible flash on every single load, not just first-ever visits. Fixed by adding an inline, synchronous `<script>` in `layout.tsx`'s `<head>` that reads a cached `mv_theme` value from `localStorage` and sets `data-theme` **before Next.js hydrates**, plus caching the resolved theme to `localStorage` whenever `/preferences` actually resolves.

**That fix didn't fully work, and the real root cause was different**: `preferences` state in `page.tsx` always *initialized* to `DEFAULT_PREFERENCES` (`theme: 'Light'`) — a `useEffect` depending on `preferences.theme` fires on mount with whatever the initial state is, which meant it fired with the *wrong* default value immediately after mount, stomping the inline script's already-correct cached theme back to light, and only correcting it again once the real `/preferences` fetch resolved seconds later. That overwrite-then-correct round trip *was* the flash, on every load, independent of caching. The actual fix: `useState`'s lazy initializer form reads the same cached `mv_theme` value the inline script uses, so the state's initial value already matches what's painted, and the mount-time effect fires as a no-op instead of a wrong value. Worth understanding both layers if this regresses again — the inline script alone looked like it should have been sufficient and wasn't, because the bug was in React state initialization, not in the paint timing the inline script addresses.

## Token-usage meter — also fixed once, non-obviously

`setDailyTokenPct` was originally only called `if (done.tokens?.daily_used)` — truthy-checking the token *count*. Several legitimate response paths (cache hits, empty-vault replies, compare/summarize/test intents) correctly report `daily_used: 0` (no new tokens were actually spent), but a falsy `0` meant the meter update was silently skipped rather than showing 0/unchanged — so the meter could get stuck on whatever the last "real" generation reported, or never appear at all (stays `null`, and the meter widget doesn't render — see the `dailyTokenPct != null &&` guard in `Sidebar`) if the session's first reply happened to hit one of those paths. Fixed to check for the presence of the `tokens` object at all, not the truthiness of the count inside it.

A related, separate gap found and fixed in the same pass: `stream_with_attachment` (backend) used `StrOutputParser()` on its LLM call, which discards the raw `AIMessageChunk` and therefore never captured `usage_metadata` at all — attachment-question replies never reported *any* token usage, a strictly worse bug than the truthy-check one (no data at all vs. a display bug over real data).

## `handleSend` / streaming consumption

Always streams — there is no code path in the current frontend that calls the non-streaming `/query` endpoint (it's still exposed server-side, doc 2, for the eval harness and as a documented fallback, but the UI never hits it). Uses `streamQuery` / `streamQueryWithAttachment` from `lib/api.ts`, both driving the shared `readSSE` parser (see below) with `onMeta`/`onToken`/`onDone`/`onError`/`onWarning` callbacks that incrementally build up the assistant `Message` in state as SSE events arrive.

## `handleUpload` and processing-state polling

Upload is fire-and-forget from the frontend's perspective: `POST /upload` returns `{"status": "processing", document_id}` almost immediately (doc 2's background-ingestion architecture), and the frontend polls `GET /documents` to detect when `chunk_count` moves off `-1` (success) or the document disappears entirely (failure — the backend deletes failed rows, doc 2). The polling window and failure-signal design went through real iteration:

- Originally capped at 2 minutes (40 attempts × 3s) — found to be too short for large documents on Render's weak CPU; the UI would just freeze on "Processing…" forever once polling gave up, even though the backend was still working (or had already failed) with no way for the UI to know.
- Extended to ~10 minutes (150 attempts × 4s), added a success toast per file when it actually finishes, and an explicit "still processing" toast if the window elapses with documents still pending — replacing silent, indefinite freezing with an honest "here's the actual state" signal either way.

## `readSSE` (`lib/api.ts`)

One shared parser for both `streamQuery` and `streamQueryWithAttachment` — both hit the exact same event shape (`meta`/`token`/`warning`/`done`/`error`, matching what `stream_rag`/`stream_with_attachment` yield server-side, doc 3), only the request construction (JSON body vs. multipart form) differs. Reads the response body as a stream, buffers partial lines across chunk boundaries (SSE frames can split across TCP packets mid-line), and dispatches each parsed `data: {...}` line to the matching callback.

## `lib/api.ts` — one axios instance + one fetch-based SSE path

Most endpoints go through a shared `axios` instance (`api`) with a request interceptor that attaches the current Supabase session's JWT as a Bearer token automatically — every call site just calls e.g. `getDocuments()` without manually fetching or attaching a token. Streaming endpoints use raw `fetch` instead of axios (axios doesn't have first-class streaming-response support in the way this needs), manually attaching the token and driving `readSSE`.

**A real, subtle bug fixed here**: `exportSessionPDF` uses `responseType: 'blob'` (needed to receive binary PDF data correctly) — but when the backend returned a `500` with a JSON error body, axios delivered that error as an *unreadable binary Blob* instead of throwing a normal error, because `responseType: 'blob'` applies to error responses too. This masked the real underlying bug (a missing `fpdf2` dependency on the server, doc 2) behind a completely opaque failure for several iterations of debugging — the error reporting itself was broken, not just the feature. Fixed by detecting a JSON-typed blob in the response and parsing it back out to surface the real backend error message.

## `lib/supabase.ts`

Thin wrapper: one lazy-singleton `SupabaseClient` (browser-only — throws if called during SSR, since Supabase's browser client isn't meant to run server-side here), `getToken()` (pulls the current session's access token), `signOut()` (Supabase sign-out + redirect to `/login`).

## `app/login/page.tsx`

Single-purpose sign-in/sign-up form, talks directly to Supabase Auth (`sb.auth.signUp` / `sb.auth.signInWithPassword`) — no backend round-trip for authentication itself, matching doc 1's note that auth is the one path bypassing the FastAPI backend entirely.

## `app/share/[token]/page.tsx`

The public, read-only shared-conversation view (doc 2's `GET /share/{token}`). No auth, no Supabase client — talks to the backend over plain unauthenticated `fetch`. Notably sets its own `height: 100vh; overflow-y: auto` container, because the main app's `globals.css` sets `overflow: hidden` on `<body>` for its fixed sidebar/chat shell, and this page isn't part of that shell — it needs its own explicit scroll region instead of relying on document-level scroll, which would otherwise just not scroll at all.

## `components/GraphPanel.tsx` — the live graph visualization

640 lines, imported and used in `page.tsx` (confirmed — the only graph component actually wired up). SVG-based, hand-rolled force-directed layout (200 iterations of a basic repulsion/attraction/center-gravity simulation, no D3 dependency), rendered as a side panel (fullscreen-toggleable) showing nodes sized by connection count and colored by whether they're a "hub" (>3 connections), multi-source, or standard node. Supports search-to-center-on-node, drag-to-reposition, pan/zoom, and a bottom info panel for the selected node with an "Ask MindVault →" button that feeds the node's label back into the chat as a new question.

## `components/GraphExplorer.tsx` — dead code

649 lines, **never imported anywhere** (confirmed — `page.tsx` only imports `GraphPanel`). This is a full second implementation of the same idea: canvas-based (not SVG) instead of SVG, its own physics constants, its own fullscreen/pin/history/search UI, visually similar but a completely separate, more heavyweight implementation (canvas redraw loop via `requestAnimationFrame`, node pinning on drag, a topic-history breadcrumb trail). Whether this was an earlier or later attempt at the same feature isn't determinable from the code alone, but it is definitively unused right now.

**Recommendation for whoever takes this over**: either delete it, or — if the canvas-based approach has real advantages (it does support node pinning and a history breadcrumb that `GraphPanel` doesn't) — deliberately decide to replace `GraphPanel` with it and wire it in, rather than leaving two competing implementations where only one is discoverable by reading the app's actual render tree.
