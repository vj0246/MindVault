"""
MindVault — RAGAS Evaluation Script
=====================================
Calls the RAG pipeline directly, in-process (rag.retrieve1.query_rag) --
no live deployment, no bearer token, no HTTP round-trip. This tests the
exact current backend code (including caching, Corrective RAG, and
Self-RAG verification) against real document chunks in the live Supabase
project, run locally.

HOW TO RUN:
-----------
1. Create/use a venv with both the main app deps and the eval-only deps:
     python -m venv eval_venv
     eval_venv/Scripts/pip install -r requirements.txt -r eval/requirements_eval.txt
2. Make sure backend/.env has SUPABASE_URL, SUPABASE_SERVICE_KEY, GROQ_API_KEY
   (same vars the main app uses -- no separate eval credentials needed).
3. Set EVAL_USER_ID below to a user_id whose vault has real, chunked
   documents matching QUESTIONS (see the README note near QUESTIONS).
4. cd backend && eval_venv/Scripts/python eval/evaluate.py
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import datetime

# Make `rag.*` / `security.*` importable when running this file directly
# (python eval/evaluate.py) rather than as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve1 import query_rag

# ── EDIT THESE ─────────────────────────────────────────────────────────────
# user_id of the account whose vault has OS Notes.pdf uploaded and chunked.
# Check: select id, filename, chunk_count from documents where user_id = '...'
EVAL_USER_ID = "630d7763-4967-481a-83e9-a1b9e9b523fe"
RETRIEVAL_MODE = "student"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in backend/.env")


def _grade(score: float) -> str:
    if score >= 0.9: return "Excellent"
    if score >= 0.7: return "Good"
    if score >= 0.5: return "Acceptable"
    return "Needs work"


# ── 25 Q&A pairs from OS_Notes.pdf ────────────────────────────────────────
# NOTE: 25 questions x 4 metrics = ~80-100 Groq judge calls.
# Groq free tier: 100k tokens/day total. Running ALL 25 may exhaust it.
# Recommended: run in batches by slicing QUESTIONS below, e.g.
#     QUESTIONS = QUESTIONS[:8]   # first batch
#     QUESTIONS = QUESTIONS[8:16] # second batch (next day or new API key)
#     QUESTIONS = QUESTIONS[16:]  # third batch
QUESTIONS = [
    {"question": "What is the difference between a program and a process?",
     "ground_truth": "A program is a passive entity stored on disk. A process is an active entity running in memory. A program becomes a process when loaded into memory and execution begins."},
    {"question": "How does a program become a process?",
     "ground_truth": "When execution starts, the OS loads the program into RAM, allocates resources, creates a Process Control Block (PCB), and assigns the CPU for execution."},
    {"question": "Can one program create multiple processes? Give an example.",
     "ground_truth": "Yes. Multiple users opening the same web browser creates separate processes. Each has its own memory space, registers, and program counter."},
    {"question": "What are the five states of a process?",
     "ground_truth": "New, Ready, Running, Waiting (Blocked), and Terminated (Exit)."},
    {"question": "What does the Running state of a process mean?",
     "ground_truth": "The process is currently being executed by the CPU. Only one process per CPU can be in the running state at a time."},
    {"question": "What triggers a transition from Running to Waiting state?",
     "ground_truth": "The process moves to Waiting when it requests I/O, file access, or waits for user input."},
    {"question": "What is a Process Control Block (PCB) and what does it store?",
     "ground_truth": "A PCB is a data structure used by the OS to track process information. It stores process state, PID, program counter, register values, memory limits, CPU quantum, priority, and list of open files."},
    {"question": "What happens after a process terminates?",
     "ground_truth": "Memory and resources are released, and the PCB is deleted."},
    {"question": "What is process suspension and resumption?",
     "ground_truth": "The OS may temporarily suspend a process to manage memory by storing it on disk. It can later be resumed and moved back to the ready state."},
    {"question": "What is a thread and why is it called a lightweight process?",
     "ground_truth": "A thread is the smallest unit of execution within a process. It is called a lightweight process (LWP) because it requires fewer resources than a full process."},
    {"question": "What does a thread have of its own vs what it shares with its process?",
     "ground_truth": "Each thread has its own program counter, register set, and stack. Threads share the code section, data section, heap memory, and open files."},
    {"question": "What are User-Level Threads (ULT) and their main disadvantage?",
     "ground_truth": "ULTs are managed in user space by a thread library without kernel involvement. Main disadvantage: blocking system call blocks the entire process."},
    {"question": "How do Kernel-Level Threads (KLT) differ from User-Level Threads?",
     "ground_truth": "KLTs are managed by the OS kernel. A blocked KLT does not stop other threads. They support true parallelism but have slower context switching than ULTs."},
    {"question": "What is the Many-to-Many thread model?",
     "ground_truth": "Many user-level threads mapped to many kernel threads. Flexible, supports better concurrency than Many-to-One. A compromise between Many-to-One and One-to-One."},
    {"question": "What is a race condition?",
     "ground_truth": "A race condition occurs when the final outcome depends on which process executes first. Two or more processes access shared data simultaneously without proper control."},
    {"question": "What is deadlock in an operating system?",
     "ground_truth": "Deadlock occurs when two or more processes are waiting for each other's resources and none can proceed. All involved processes remain blocked permanently."},
    {"question": "What is starvation and what causes it?",
     "ground_truth": "Starvation happens when a process never gets required resources. Caused by improper scheduling where high-priority processes continuously get CPU time."},
    {"question": "What are the three requirements for a correct critical section solution?",
     "ground_truth": "1) Mutual Exclusion — only one process in critical section at a time. 2) Progress — selection cannot be postponed indefinitely. 3) Bounded Waiting — limit on how many times others enter before a waiting process gets its turn."},
    {"question": "What is a semaphore and what two atomic operations does it use?",
     "ground_truth": "A semaphore is a synchronization integer variable. It uses wait(S) — decrements, blocks if < 0 — and signal(S) — increments and wakes a waiting process."},
    {"question": "What is the difference between a counting semaphore and a binary semaphore?",
     "ground_truth": "Counting semaphore ranges from 0 to n, manages multiple resource instances. Binary semaphore has only 0 or 1, used for mutual exclusion like a mutex."},
    {"question": "What are the limitations of semaphores?",
     "ground_truth": "Priority inversion, deadlock risk if misused, complex to manage, and busy waiting in basic implementations."},
    {"question": "How does a mutex differ from a binary semaphore?",
     "ground_truth": "A mutex enforces strict ownership — only the thread that locks it can unlock it. Uses priority inheritance to avoid priority inversion. Binary semaphore has no ownership enforcement."},
    {"question": "What three semaphores are used in the Producer-Consumer solution?",
     "ground_truth": "mutex — mutual exclusion on buffer. full — counts filled slots, prevents consumer from empty buffer. empty — counts empty slots, prevents producer from overfilling."},
    {"question": "In the Readers-Writers problem, what is Readers Preference and its drawback?",
     "ground_truth": "Readers Preference gives priority to readers — no reader waits if resource is open for reading. Drawback: writers may starve if readers keep arriving."},
    {"question": "What is the Banker's Algorithm for deadlock avoidance?",
     "ground_truth": "The document does not contain information about the Banker's Algorithm. A correct answer should say this topic is not in the uploaded documents. (Note: MindVault's Corrective RAG fallback means a low-confidence retrieval may now answer this from general knowledge instead of refusing -- check answer_type below rather than assuming a refusal is correct.)"},
]


# ── EDIT: uncomment ONE line below to run a smaller batch ──────────────────
# QUESTIONS = QUESTIONS[:8]
# QUESTIONS = QUESTIONS[8:16]
# QUESTIONS = QUESTIONS[16:]

# ── Step 1: Run each question in-process against query_rag ─────────────────
print(f"[Eval] Running {len(QUESTIONS)} questions in-process (user_id={EVAL_USER_ID})...")
print("-" * 65)

SESSION_ID = f"eval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
records = []

for i, item in enumerate(QUESTIONS):
    q, gt = item["question"], item["ground_truth"]
    print(f"[{i+1:02d}/{len(QUESTIONS)}] {q[:60]}...")

    try:
        result = query_rag(
            question=q,
            history=[],
            mode=RETRIEVAL_MODE,
            user_id=EVAL_USER_ID,
            document_ids=None,
        )
        answer     = result.get("answer", "")
        confidence = result.get("confidence", 0.0)
        answer_type = result.get("answer_type", "grounded")
        chunks     = result.get("chunks", [])
        contexts   = [c["content"] for c in chunks] if chunks else [answer]

        records.append({
            "question": q, "answer": answer, "contexts": contexts,
            "ground_truth": gt, "confidence": confidence,
            "answer_type": answer_type,
            "sources": result.get("sources", []),
        })
        print(f"         -> {answer[:80]}...")
        print(f"            chunks={len(contexts)}  confidence={confidence:.2f}  answer_type={answer_type}")

    except Exception as e:
        print(f"         ERROR: {e}")
        records.append({
            "question": q, "answer": f"ERROR: {e}",
            "contexts": ["ERROR"], "ground_truth": gt,
            "confidence": 0.0, "answer_type": "error", "sources": [],
        })

print("-" * 65)


# ── Step 2: RAGAS evaluation ────────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.run_config import RunConfig
    from langchain_groq import ChatGroq
    from langchain.embeddings.base import Embeddings
except ImportError as e:
    print(f"\n[Eval] Missing dep: {e}")
    print("Run: pip install -r eval/requirements_eval.txt")
    sys.exit(1)

print(f"\n[Eval] Scoring {len(records)} records via RAGAS + Groq (~1-3 min)...")

# llama-3.1-8b-instant / a fast judge model keeps this well under Groq's
# free-tier rate limits compared to the 70b tier used for real answers.
ragas_llm = LangchainLLMWrapper(ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY,
    max_retries=5
))

# Reuse the SAME embedding model the live app uses (rag/embedder.py) instead
# of loading a second copy -- same reasoning as that module's own singleton
# comment: avoids doubling memory for an identical model.
from rag.embedder import EMBED_MODEL

class _SharedEmbeddings(Embeddings):
    def embed_documents(self, texts): return [e.tolist() for e in EMBED_MODEL.embed(texts)]
    def embed_query(self, text): return list(EMBED_MODEL.embed([text]))[0].tolist()

ragas_embeddings = LangchainEmbeddingsWrapper(_SharedEmbeddings())

metrics = [
    Faithfulness(llm=ragas_llm),
    AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

valid = [r for r in records if r["answer_type"] != "error"]
if not valid:
    print("[Eval] No valid records. Fix errors above.")
    sys.exit(1)

samples = [
    SingleTurnSample(
        user_input=r["question"],
        retrieved_contexts=r["contexts"],
        response=r["answer"],
        reference=r["ground_truth"],
    )
    for r in valid
]
dataset = EvaluationDataset(samples=samples)

# Sequential-ish execution (max_workers=2) with generous retries/timeout
# to survive Groq free-tier rate limits without cascading failures.
run_config = RunConfig(max_workers=2, timeout=120, max_retries=5, max_wait=30)

try:
    result = evaluate(
        dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )
except Exception as e:
    print(f"[Eval] RAGAS failed: {e}")
    sys.exit(1)

df = result.to_pandas()
scores = {
    "faithfulness":      round(float(df["faithfulness"].mean()),      4),
    "answer_relevancy":  round(float(df["answer_relevancy"].mean()),  4),
    "context_precision": round(float(df["context_precision"].mean()), 4),
    "context_recall":    round(float(df["context_recall"].mean()),    4),
}
overall = round(sum(scores.values()) / len(scores), 4)


# ── Save first -- a display/encoding crash below must never cost the
# actual RAGAS judge results, which just spent real Groq quota to compute.
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_final")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"eval_{ts}.json")
with open(out_path, "w") as f:
    json.dump({"timestamp": ts, "scores": scores, "overall": overall,
               "per_question": records}, f, indent=2)
print(f"\n[Eval] Saved -> {out_path}")

# ── Results ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RAGAS EVALUATION -- MindVault (in-process)")
print("=" * 65)

try:
    from tabulate import tabulate
    rows = [[m, s, _grade(s)] for m, s in scores.items()]
    rows += [["-"*22, "-"*6, "-"*12], ["OVERALL", overall, _grade(overall)]]
    # "grid" is plain ASCII (+/-/|) -- tablefmt options with box-drawing
    # unicode (e.g. rounded_outline) crash print() on Windows cp1252
    # stdout, same class of bug as the ingest.py arrow character earlier
    # this project fixed.
    print(tabulate(rows, headers=["Metric", "Score", "Grade"], tablefmt="grid"))
except ImportError:
    for m, s in scores.items():
        print(f"  {m:<25} {s:.4f}  {_grade(s)}")
    print(f"  {'OVERALL':<25} {overall:.4f}  {_grade(overall)}")

print()
print("PER-QUESTION CONFIDENCE / ANSWER TYPE:")
for r in records:
    flag = "[LOW] " if r["confidence"] < 0.4 else "[OK]  "
    print(f"  {flag}[{r['confidence']:.2f}] ({r['answer_type']}) {r['question'][:55]}")

print()
print("METRIC GUIDE:")
print("  faithfulness      -> grounding in docs (anti-hallucination)")
print("  answer_relevancy  -> does answer address the question")
print("  context_precision -> are retrieved chunks relevant")
print("  context_recall    -> did retrieval find all needed chunks")
print("  <0.5 needs work | 0.5-0.7 ok | >0.7 good | >0.9 excellent")
print("=" * 65)
