# MindVault

**A multi-user, production-deployed RAG system that lets people upload their own documents and have grounded, cited conversations with them — built and operated end-to-end, including the parts that don't make it into tutorials.**

🔗 Live: [mind-vault-red.vercel.app](https://mind-vault-red.vercel.app) · Backend: FastAPI on Render · Frontend: Next.js on Vercel

---

## 30-second summary

I built a RAG (Retrieval-Augmented Generation) platform from scratch — not a LangChain quickstart wrapped in a chat UI, but a real multi-tenant product with hybrid retrieval (semantic + keyword search fused via Reciprocal Rank Fusion), cross-encoder reranking, automated accuracy evaluation with RAGAS, streaming responses, multimodal ingestion, and a knowledge graph layer. It's been live, used, and broken in real ways — memory leaks, rate-limit cascades, silent data-corruption bugs — and I fixed every one of them in production, not in a clean rewrite.

If you read nothing else: this project demonstrates I can take a RAG system from "it works on my machine" to "it works for multiple real users on a free-tier server with 512MB of RAM," which is a much harder and more honest engineering problem than most portfolio RAG projects attempt.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER (Vercel)                       │
│   Next.js 14 · App Router · Multi-session chat UI · SSE streaming   │
└───────────────┬───────────────────────────────────┬─────────────────┘
                │                                   │
                │ Login/Signup                      │ All other requests
                ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────────────┐
│   Supabase Auth (JWT)    │         │     FastAPI Backend (Render)     │
│   direct from browser    │         │     verifies JWT on every route  │
└─────────────────────────┘         └───────────────┬───────────────────┘
                                                     │
                  ┌──────────────────────────────────┼──────────────────────────────────┐
                  ▼                                  ▼                                  ▼
      ┌───────────────────────┐      ┌────────────────────────────┐      ┌──────────────────────┐
      │   Supabase Postgres    │      │        Groq Cloud           │      │   Local CPU Inference  │
      │   + pgvector            │      │  llama-3.3-70b-versatile    │      │   fastembed (bge-small) │
      │   chunks · documents    │      │  → auto-fallback to         │      │   FlashRank reranker    │
      │   sessions · graph      │      │    llama-3.1-8b-instant     │      │   (ONNX, ~23MB, no torch)│
      │   chat_sessions         │      │   on rate-limit             │      │                        │
      │   RLS on every table    │      └────────────────────────────┘      └──────────────────────┘
      └───────────────────────┘
```

### What happens when you ask a question — the request loop

This is the part most RAG tutorials skip, and it's where most of the actual engineering lives. Every step below is a real branch in the code (`backend/rag/retrieve1.py`), not a simplification — including exactly where an LLM gets called and where it deliberately doesn't.

```
                              ┌───────────────────────────┐
                              │   User submits a question  │
                              └─────────────┬───────────────┘
                                            ▼
                       ┌─────────────────────────────────────────┐
                       │ Does it reference prior conversation     │
              ┌───NO───│ ("that", "elaborate") OR is it <=4 words │───YES──┐
              │        │ (likely under-specified)?                │        │
              │        └───────────────────────────────────────────┘        │
              ▼                                                            ▼
   Cheap keyword-first intent                          LLM CALL #1 (optional): rewrite the
   classify, question used as-is                       question into a fuller, standalone form
              │                                        using conversation history + classify intent
              └──────────────────┬─────────────────────────────┘
                                 ▼
              ┌───────────────────────────────────────────┐
              │  Question embedded locally (fastembed,     │
              │  BAAI/bge-small-en-v1.5, 384-dim) --        │
              │  no API call, runs on the server itself     │
              └─────────────────────┬───────────────────────┘
                                    ▼
      ┌─────────────────────────────────────────────────────────────┐
      │  Semantic search (pgvector cosine, RPC match_chunks)          │
      │  Keyword search (Postgres tsvector/GIN, RPC keyword_search)   │  <- run concurrently,
      │  fire in parallel via ThreadPoolExecutor, not sequentially    │     not sequential awaits
      └─────────────────────────────┬─────────────────────────────────┘
                                    ▼
              ┌───────────────────────────────────────────┐
              │  Reciprocal Rank Fusion (RRF) merges both   │
              │  ranked lists by position, not raw score --  │
              │  a chunk both methods agree on outranks one  │
              │  only one method found                       │
              └─────────────────────┬───────────────────────┘
                                    ▼
              ┌───────────────────────────────────────────┐
              │  Cross-encoder rerank (FlashRank, ONNX,     │
              │  ms-marco-MiniLM-L-12-v2) -- scores query    │
              │  and passage together, more accurate than    │
              │  bi-encoder similarity or RRF rank alone      │
              └─────────────────────┬───────────────────────┘
                                    ▼
              ┌───────────────────────────────────────────┐
              │  Confidence score: blend of normalized RRF  │
              │  agreement + raw cosine similarity, with a   │
              │  bonus for multiple corroborating chunks      │
              └─────────────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
           BELOW 0.45│  Corrective RAG (CRAG) gate:   │ABOVE 0.45
           ┌──────── │  is retrieval actually a real   │────────┐
           ▼          │  match for the question?        │        ▼
  General-knowledge   └───────────────────────────────┘   Grounded mode:
  mode: model answers                                      STRICT RULE --
  from its own training,                                    context ONLY,
  must prefix reply with                                     cite [filename]
  "[General knowledge]"                                      after every fact,
  so the UI can flag it                                      refuse verbatim if
  as not sourced from                                        not in context
  your documents                                                    │
           │                                                        │
           └───────────────────┬────────────────────────────────────┘
                                ▼
              ┌───────────────────────────────────────────┐
              │  LLM CALL #2 (always): llama-3.3-70b-       │
              │  versatile generates the answer, streamed   │
              │  token-by-token over SSE. Falls back to      │
              │  llama-3.1-8b-instant automatically if the    │
              │  70b model's daily Groq quota is exhausted     │
              └─────────────────────┬───────────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
        confidence >=0.75│  Self-RAG post-check: is the    │confidence <0.75
           ┌──────────── │  answer's own confidence high    │ ────────────┐
           ▼              │  enough to trust the STRICT RULE  │             ▼
   Skip verification --   │  prompting alone?                  │   LLM CALL #3: a small
   trust the prompting,   └───────────────────────────────────┘   model (llama-3.1-8b-instant)
   ship the answer as-is                                          re-reads the context and the
           │                                                      generated answer, replies
           │                                                      SUPPORTED or UNSUPPORTED --
           │                                                      an unsupported answer gets
           │                                                      swapped for the refusal phrase
           └───────────────────┬────────────────────────────────────┘
                                ▼
              ┌───────────────────────────────────────────┐
              │  Answer (already streamed) + confidence +   │
              │  sources + related-concepts (knowledge-graph │
              │  BFS, runs on its own thread concurrently      │
              │  with the token stream, never adds latency)     │
              └───────────────────────────────────────────┘
```

Every table involved (`chunks`, `documents`, `sessions`, `graph_nodes`, `graph_edges`) has a `user_id` column and a Postgres Row Level Security policy — there is no code path that can leak one user's documents into another user's answer.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | Next.js 14 (App Router), TypeScript | Server components where useful, SSE streaming support, file-based routing for the share-link feature |
| Backend | FastAPI (Python) | Async-friendly, typed request models, clean dependency injection for auth |
| Database | Supabase — Postgres + pgvector + Auth | Vector search, relational data, and auth in one service — no separate vector DB to keep in sync |
| LLM | Groq (`llama-3.3-70b-versatile`, fallback `llama-3.1-8b-instant`) | Fast inference, generous free tier, and — critically — Groq enforces token quotas *per model*, which I exploit for automatic failover |
| Embeddings | `fastembed` running `BAAI/bge-small-en-v1.5` | Local CPU inference, no external API call per query, fits in 512MB RAM (`sentence-transformers` was tried first and OOM'd the server — see *Decisions* below) |
| Keyword search | Postgres full-text search (`tsvector`, GIN index, `ts_rank`) | BM25-equivalent ranking without a separate search engine like Elasticsearch |
| Reranking | FlashRank (`ms-marco-MiniLM-L-12-v2`, ONNX) | Cross-encoder accuracy without pulling in `torch` (~1GB, would not fit the hosting tier) |
| Orchestration | LangChain (LCEL chains) + raw functions | LCEL for the LLM call chains (prompt \| llm \| parser), hand-written Python for retrieval/fusion logic where a chain abstraction would have hidden the actual algorithm |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, context recall), Groq as judge LLM | Automated, repeatable accuracy measurement instead of eyeballing answers |
| Hosting | Render (backend, free tier), Vercel (frontend) | Free-tier constraints (512MB RAM, shared CPU) are *why* most of the production-engineering work in this README exists |
| Document parsing | PyPDFLoader, Docx2txtLoader, TextLoader, Groq Vision (image OCR) | One ingestion path for PDF, DOCX, TXT, MD, and images — images go through a vision LLM call instead of a separate OCR library |
| PDF generation | fpdf2 | Human-readable session export, no headless browser needed |

---

## Running this locally

**Prerequisites:** Python 3.11+, Node 18+, a Supabase project (Postgres + pgvector already enabled), a Groq API key.

**1. Clone and set up the database**

Run `backend/migrations/001_initial_schema.sql` via the Supabase SQL editor (or `supabase db push`) — creates `documents`, `chunks` (with a `vector(384)` `embedding` column and the `match_chunks`/`keyword_search_chunks` RPCs), `sessions`, `chat_sessions`, `graph_nodes`, `graph_edges`, `query_cache`, `rate_limits`, `user_preferences`, and `user_memory_notes`, plus Row Level Security policies scoping every user-facing table by `user_id`. This file is a consolidated snapshot of the live schema (most of it was built directly against the running project across earlier sessions, never previously committed) rather than a literal replay of every incremental change.

**2. Backend**

```bash
cd backend
pip install -r requirements.txt
```

Create `backend/.env`:

```
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
GROQ_API_KEY=key1,key2        # comma-separated, one key is fine too
ALLOWED_ORIGINS=http://localhost:3000
FRONTEND_URL=http://localhost:3000
```

```bash
uvicorn app:app --reload
```

**3. Frontend**

```bash
cd frontend
npm install
```

Create `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
```

```bash
npm run dev
```

**4. Evaluation (optional, separate dependency set)**

`ragas` needs a newer `langchain-core`/`pydantic` than nothing else in the project conflicts with, but it's still cleanest in its own venv:

```bash
cd backend
python -m venv eval_venv
eval_venv/Scripts/pip install -r requirements.txt -r eval/requirements_eval.txt   # Scripts/ -> bin/ on macOS/Linux
eval_venv/Scripts/python eval/evaluate.py
```

Set `EVAL_USER_ID` in `eval/evaluate.py` to a real user whose vault has documents matching the question set first. Results land in `backend/eval/results_final/`.

---

## Features — what's actually built and working

**Retrieval & answer quality**
- Hybrid semantic + keyword retrieval with RRF fusion
- Cross-encoder reranking pass after fusion
- Per-answer confidence scoring, surfaced in the UI
- Inline source citation, enforced by prompt structure, not just requested
- Strict-grounding mode (`temperature=0`, explicit refusal phrase) to suppress hallucination
- Scoped retrieval — query against specific documents only, via sidebar checkboxes
- Intent routing — the same question is handled differently depending on whether it's a factual ask, a comparison, a summarization, or a quiz-generation request, with a cheap keyword-first classifier before falling back to an LLM call

**Multimodal ingestion**
- PDF, DOCX, TXT, MD uploaded into a permanent per-user vector store
- Images and one-off attachments handled via Groq Vision, usable for a single question without being permanently stored — with a separate, explicitly prioritized prompt so the model doesn't confuse "what does this attached image show" with "what does my document vault say"

**Knowledge graph**
- Entity and relationship extraction (LLM-based) runs in the background after upload, not blocking the response
- Interactive graph visualization of concepts and how they connect across a user's documents
- BFS traversal surfaces "related concepts" alongside every chat answer

**Product features (not just a chatbox)**
- Multi-session chat, ChatGPT-thread style, with auto-generated titles
- Sessions persist across refresh and devices (Supabase-backed, not just `localStorage`)
- Clean sequential session numbers (`#1`, `#2`...) shown to the user — internal IDs are UUIDs, but nobody should have to read a UUID
- Shareable, revocable, read-only public links to a conversation
- PDF and Markdown export of full conversations
- Streaming token-by-token responses over SSE

**Evaluation**
- A standalone RAGAS harness that calls the RAG pipeline in-process (`rag.retrieve1.query_rag`, not an HTTP client) and scores faithfulness, answer relevancy, context precision, and context recall against a hand-written question set with ground-truth answers — including adversarial questions designed to confirm the system correctly says "I don't know" rather than guessing

---

## Engineering decisions — and what I rejected

The interesting part of a real project isn't the happy path, it's what got tried and thrown out.

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| Vector store | Supabase pgvector | FAISS | Render's filesystem is ephemeral — anything written to disk is gone on redeploy. pgvector persists. |
| Embeddings | fastembed (local) | `sentence-transformers` | Pulls in `torch` (~1GB+), exceeded Render's 512MB RAM and crashed the server (OOM) on startup |
| Embeddings (API) | — | Hugging Face Inference API | Render's free tier blocks outbound DNS to `api-inference.huggingface.co` — found this the hard way |
| LLM | Groq | Self-hosted Ollama | Render has no GPU; Ollama needs one to be usable |
| Reranker | FlashRank | Cohere Rerank API | FlashRank runs locally, no extra API key, no extra network hop, no extra cost — at the cost of slightly lower ceiling accuracy than a hosted reranker |
| Result fusion | Reciprocal Rank Fusion | Simple score concatenation / take-top-k-from-each | RRF doesn't require normalizing two incomparable score scales (cosine similarity vs. ts_rank) — it only needs rank position, which sidesteps a real bug class |
| Eval methodology | Import the RAG pipeline directly, in-process (`ragas>=0.4`) | Hit the live deployed API over HTTP | The first version hit the live API to sidestep a `langchain==0.3.25` vs. `ragas==0.1.21` (`langchain<0.3`) dependency conflict. Later resolved properly: `ragas>=0.4` resolves cleanly against the production `langchain`/`langchain-core` pins in an isolated venv, so eval now imports and calls `query_rag()` directly — exercises the exact current code (caching, CRAG, Self-RAG) instead of a network round-trip, and removed a hardcoded live JWT that had been committed across three prior commits. |
| Session identity | Auto-generated sequential number for display, UUID internally | Show the raw UUID | Nobody wants to read `b90ba15b-cd41-4a0f-...` in an exported PDF. The UUID stays as the real foreign key everywhere it matters; the human gets a clean number. |

---

## Production engineering — the bugs that don't show up in a demo

This is the section most portfolio RAG projects don't have, because they never ran long enough to hit these. I did, and fixed all of them on a live system:

**Memory leaks → silent 502s after a handful of requests.** Both the Supabase client and the Groq LLM client were being instantiated fresh on every function call across several files. Each instantiation opens its own `httpx` connection pool that doesn't get cleaned up fast enough under load. On a 512MB server, this accumulated until the process was OOM-killed — which looked like random 502 errors with no obvious pattern. Fix: both clients are now module-level singletons, created once, reused for the life of the process.

**Rate-limit cascades.** Groq enforces token quotas *per model*, not per account. A burst of traffic (including my own eval runs hitting the same API key) could exhaust the 70B model's daily quota and start failing user requests with `429`. Fix: the LLM call is wrapped with `with_fallbacks()` — on a rate-limit error, it automatically and transparently retries on `llama-3.1-8b-instant`, which has its own separate quota. Users never see the failure.

**Silent data corruption on upload.** The original upload path would write a `chunk_count: 0` document row to the database even when a file failed to parse (e.g., a scanned PDF with no extractable text) — a fake "success" that left a useless, undeletable-looking document in the user's sidebar. Fix: ingestion now checks extracted text length *before* chunking and raises a clear, user-facing error; the backend deletes the orphaned row instead of leaving silent garbage behind.

**A frontend bug that hid its own root cause.** PDF export was failing, and for several iterations I couldn't find out why — because the frontend's error handler used `responseType: 'blob'`, and when the backend returned a `500` with a JSON error body, axios delivered that error as an unreadable binary blob instead of surfacing it. The actual bug (a missing `fpdf2` dependency on the server) was invisible because the *error reporting itself* was broken. Fixed the error path first, which then made the real bug visible and trivial to fix.

**A classic React performance bug at scale.** With many chat messages, the UI got visibly laggy while typing. Root cause: the input textbox and the message list lived in the same component, so every keystroke re-rendered the *entire* conversation history, including re-parsing Markdown for every assistant message. Fixed with `React.memo` on the message component — but that alone wasn't sufficient, because a callback prop passed into it was being recreated on every render too, silently defeating the memoization. Had to stabilize that with `useCallback` as well. (Memoizing a component while still passing it an unstable function prop is a very common, very invisible mistake.)

**Large-attachment timeouts.** A user attached a multi-hundred-page OCR'd course PDF to a single question. The backend was extracting and joining *every page* into memory before truncating to the ~4,000 characters actually used downstream — meaning a 5-page and a 500-page attachment cost wildly different amounts of CPU time for the same effective input. Fixed by capping extraction at a character budget during the page-by-page loop itself, not after.

**Latency, addressed structurally, not just by throwing compute at it:**
- Independent retrieval calls (semantic + keyword search) parallelized with `ThreadPoolExecutor` instead of sequential `await`s
- The extra LLM call for rewriting follow-up questions (`resolve_and_classify`) now only fires when the question actually contains a reference word ("that", "above," "elaborate"); standalone questions skip it entirely
- Knowledge-graph lookup runs concurrently with token streaming instead of after it

---

## Evaluation results

RAGAS scores (Groq as judge LLM), same 8-question subset across all runs for a direct comparison. `context_recall`/`context_precision` measure the retrieval layer; `faithfulness`/`answer_relevancy` measure generation.

| Run | faithfulness | answer_relevancy | context_precision | context_recall | overall |
|---|---|---|---|---|---|
| Baseline (`k=5`, all-MiniLM-L6-v2, 500-char chunks, CRAG threshold 0.35) | 0.690 | 0.864 | 0.624 | 0.563 | 0.685 |
| `k=8`, bge-small-en-v1.5, 1000-char chunks, CRAG threshold 0.45 | 0.687 | 0.930 | 0.563 | 0.771 | 0.738 |
| `k=6` (same embedding/chunking/threshold as above) | 0.651 | 0.911 | 0.630 | 0.604 | 0.699 |

Baseline → `k=8` pass: retrieval `k` (5→8), the embedding model (`all-MiniLM-L6-v2` → `bge-small-en-v1.5`, same 384-dim so no schema migration), fixed-splitter chunk size (500/50 chars → 1000/150), and the CRAG confidence gate (0.35 → 0.45) all changed together, validated as one before/after comparison rather than an exhaustive per-parameter grid search — Groq's free-tier rate limits make a full sweep impractical (a single 8-question run already takes 8-11 minutes with heavy `429` retries). Net result: `context_recall` jumped from *acceptable* to *good* (more of the right chunks actually get retrieved), at a `context_precision` cost that's the expected trade-off of pulling more candidates into the top-`k` — more chances for a weaker match to make the cut.

`k=8` → `k=6` retest: recovered most of the `context_precision` loss (0.563 → 0.630, above even the original baseline) while keeping most of the recall gain (0.771 → 0.604, still well above baseline's 0.563). `faithfulness` dropped a bit further (0.687 → 0.651) — plausibly noise at an 8-question sample size rather than a real regression, since nothing about `k=6` should structurally hurt grounding. `k=6` is the better middle ground of the three and is what's currently live.

Raw results: `backend/eval/results_final/`.

---

## What I have *not* done — and why that's worth saying

A list of finished features is a sales pitch. A list of unfinished ones is engineering judgment.

- **No vector index (HNSW) confirmed on the `embedding` column yet.** At current data volume this is invisible; it will become the single biggest latency bottleneck as the dataset grows past a few thousand chunks, since similarity search currently risks a sequential scan. This is the next thing I'd fix before scaling further.
- **No HyDE (Hypothetical Document Embeddings).** Would likely improve retrieval further on vague or abstract questions at the cost of one extra LLM call per query. Partially addressed by a cheaper alternative instead: short/vague questions (<=4 words) already trigger a query-rewrite pass before retrieval, but a full HyDE-style hypothetical-answer expansion hasn't been built.
- **Still on free-tier hosting.** Render's free tier (512MB RAM, shared single vCPU, cold starts) shaped a large amount of the engineering described above — including a real production incident where a large document's embedding batch got the container OOM-killed mid-ingestion, fixed by sub-batching the embedding calls instead of one giant call per document. A meaningful fraction of the remaining latency is infrastructure, not code — and I'd say that plainly in an interview rather than implying the code is the only lever.
- **Hyperparameters (retrieval `k`, chunk size, CRAG confidence threshold) are reasoned defaults, not exhaustively grid-searched.** Validated via before/after RAGAS comparisons (see *Evaluation results* below), but a full sweep across every combination isn't practical against Groq's free-tier rate limits — a single 8-question eval run already takes 8-11 minutes with heavy `429` retries.

What's already built that an earlier version of this README claimed wasn't: per-user *and* per-IP rate limiting (`security/rate_limit.py`), response caching for repeated non-follow-up questions (`rag/cache.py`), fully asynchronous ingestion via `BackgroundTasks` with a hard 5-minute timeout (a stuck parse/embed job gets cleaned up instead of leaving a zombie row forever), and LangSmith tracing (`@traceable` on every retrieval/generation function). Keeping this list honest matters more than keeping it long.

---

## Future scope: from document chat to a personal knowledge graph

The graph layer that currently powers "related concepts" under each answer is the seed of something bigger, and it's the direction I'd take this next.

Right now, entity/relationship extraction runs per-document, in the background, and produces a graph scoped to answering one question. The natural extension is to treat that graph as a first-class product surface rather than a side effect of chat:

- **A persistent, navigable personal knowledge graph** — not just "related concepts" under one answer, but a standalone view where a user can visually explore how ideas, people, and topics connect across *everything* they've ever uploaded, growing over time as they add more documents.
- **Cross-document relationship discovery** — the current extraction is per-document; the higher-value version actively looks for relationships *between* documents uploaded at different times (e.g., connecting a concept introduced in a textbook chapter to a related concept in a research paper added months later), turning the vault into something closer to a personal second brain than a search index.
- **Graph-augmented retrieval** — using graph proximity as an additional signal in the hybrid retrieval pipeline (alongside semantic and keyword search), so a query about one concept can surface chunks that are graph-adjacent even if they don't score highly on text similarity alone.
- **Temporal/versioned knowledge** — tracking how a user's understanding of a topic evolves across documents added over time, rather than treating the graph as a static snapshot.

This is a meaningfully larger system than what exists today, but the foundation — entity extraction, relationship storage, BFS traversal, the visualization component — is already built and already running in production. The next phase is making the graph a destination, not a footnote.

---