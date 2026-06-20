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

### What happens when you ask a question

This is the part most RAG tutorials skip, and it's where most of the actual engineering lives:

1. **Question embedded locally** via `fastembed` (BAAI/bge-small-en-v1.5, 384-dim) — no API call, runs on the Render server itself.
2. **Two retrieval methods fire in parallel** (via `ThreadPoolExecutor`, not sequentially):
   - **Semantic search** — cosine similarity against `pgvector`, via a Postgres RPC function (`match_chunks`)
   - **Keyword search** — PostgreSQL full-text search (`tsvector` + GIN index, `ts_rank`, BM25-style) via a second RPC function (`keyword_search_chunks`)
3. **Results fused with Reciprocal Rank Fusion (RRF)** — not a naive concatenation. Each chunk's score is `1/(60+rank)` per list it appears in, so a chunk that both methods agree on outranks a chunk only one method found. This is the actual fix for the classic RAG failure mode where semantic search misses exact terms (acronyms, proper nouns, numbers) and keyword search misses meaning (synonyms, paraphrases).
4. **Cross-encoder reranking** via FlashRank (`ms-marco-MiniLM-L-12-v2`) — RRF merges by rank position; the reranker scores query-passage relevance directly by reading both texts together, which is strictly more accurate than either bi-encoder similarity or RRF score alone.
5. **Confidence score computed** — blends normalized RRF agreement (did both methods find this?) with raw cosine similarity, 50/50, with a small bonus for multiple corroborating chunks. Surfaced to the user as a colored badge (🟢/🟡/🔴) so they can tell when the system is genuinely unsure rather than confidently hallucinating.
6. **Context labeled per-source** (`[Doc: filename.pdf]`) and the LLM is instructed — with an explicit, repeated grounding rule, at `temperature=0` — to cite the source after every factual claim, and to say a fixed, exact phrase if the answer isn't in context. This is the actual anti-hallucination mechanism: not a hope, a structural constraint.
7. **Answer streams token-by-token** over Server-Sent Events. The knowledge-graph lookup for "related concepts" runs on a background thread *concurrently* with the token stream, so it never adds to perceived latency.
8. **Everything is scoped to the requesting user** — every table has a `user_id` column and a Postgres Row Level Security policy. There is no code path that can leak one user's documents into another user's answer.

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
- A standalone RAGAS harness that hits the *live* deployed API (not a local re-implementation) and scores faithfulness, answer relevancy, context precision, and context recall against a hand-written question set with ground-truth answers — including adversarial questions designed to confirm the system correctly says "I don't know" rather than guessing

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
| Eval methodology | Hit the live deployed API over HTTP | Import the RAG pipeline locally and call it directly | The production stack pins `langchain==0.3.25`; RAGAS 0.1.21 requires `langchain<0.3`. These cannot coexist in one environment. Treating eval as a black-box HTTP client sidesteps a real, unsolvable dependency conflict and — as a side effect — tests exactly what a user experiences, not an idealized local code path. |
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

## What I have *not* done — and why that's worth saying

A list of finished features is a sales pitch. A list of unfinished ones is engineering judgment.

- **No vector index (HNSW) confirmed on the `embedding` column yet.** At current data volume this is invisible; it will become the single biggest latency bottleneck as the dataset grows past a few thousand chunks, since similarity search currently risks a sequential scan. This is the next thing I'd fix before scaling further.
- **No LangSmith / structured tracing in production yet.** The chains are already built with LangChain primitives, so adding it is a configuration change, not a rewrite — it just hasn't been prioritized over correctness bugs yet.
- **No HyDE (Hypothetical Document Embeddings).** Would likely improve retrieval on vague or abstract questions at the cost of one extra LLM call per query — a deliberate latency/accuracy trade-off I haven't made yet.
- **No per-user rate limiting.** Right now a single user could exhaust the shared Groq quota for everyone. This is a real gap, not an oversight — it's next on the list.
- **No response caching.** Identical repeated questions re-run the full pipeline every time.
- **Synchronous ingestion.** Large document uploads block the HTTP request for the full duration of chunking and embedding rather than queuing the work and returning immediately.
- **Still on free-tier hosting.** Render's free tier (512MB RAM, shared CPU, cold starts) shaped a large amount of the engineering described above. A meaningful fraction of the remaining latency is infrastructure, not code — and I'd say that plainly in an interview rather than implying the code is the only lever.

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

## A note on how this was built

I worked with Claude (Anthropic) throughout this build as a pairing partner for implementation — writing and debugging code, reasoning through production incidents, and structuring architectural decisions. Every line shipped to production was reviewed, tested against real failures, and iterated on by me. The judgment calls — what to build, which trade-offs to accept, what "good enough" looks like for a free-tier-hosted multi-user system — were mine, made by working through dozens of real, live bugs rather than a single clean build. I'm saying this plainly because I think it's a more honest and more impressive story than pretending otherwise: knowing how to direct, debug, and ship with AI assistance is itself the skill, and it's the same skill I'd bring to a team.