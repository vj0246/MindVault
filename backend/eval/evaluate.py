"""
MindVault — RAGAS Evaluation Script
=====================================
Calls your LIVE Render API — no local pipeline import needed.
Tests the actual deployed system exactly as users experience it.

HOW TO RUN:
-----------
1. pip install ragas==0.1.21 datasets pandas tabulate requests

2. Set these 4 variables below:
   - RENDER_API_URL  → your Render backend URL
   - USER_EMAIL      → email of test user (who has OS_Notes.pdf)
   - USER_PASSWORD   → their password
   - SUPABASE_URL + SUPABASE_ANON_KEY → from your .env

3. cd backend && python eval/evaluate.py
"""
from dotenv import load_dotenv

load_dotenv()
import os, sys, json, datetime, requests

# ── EDIT THESE ─────────────────────────────────────────────────────────────
RENDER_API_URL = "https://mindvault-98xb.onrender.com"
RETRIEVAL_MODE = "student"

# Paste your JWT token here.
# How to get it:
#   1. Open your MindVault site in browser
#   2. DevTools (F12) → Network tab → click any request (e.g. /documents)
#   3. Headers → Authorization → copy the value after "Bearer "
#   4. Paste the full eyJ... token below
BEARER_TOKEN="eyJhbGciOiJFUzI1NiIsImtpZCI6IjE3MzExODg1LTNiMzctNDRkZC1hMTY4LWY3OTRlMTY3YmIxMSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL3dtY2h0ZHR4bnNibGx4a3ltdXFjLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiI2MzBkNzc2My00OTY3LTQ4MWEtODNlOS1hMWI5ZTliNTIzZmUiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzgxNDYzMzgyLCJpYXQiOjE3ODE0NTk3ODIsImVtYWlsIjoidml2YWFuLmphaW4yNDZAZ21haWwuY29tIiwicGhvbmUiOiIiLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJlbWFpbCIsInByb3ZpZGVycyI6WyJlbWFpbCJdfSwidXNlcl9tZXRhZGF0YSI6eyJlbWFpbCI6InZpdmFhbi5qYWluMjQ2QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJwaG9uZV92ZXJpZmllZCI6ZmFsc2UsInN1YiI6IjYzMGQ3NzYzLTQ5NjctNDgxYS04M2U5LWExYjllOWI1MjNmZSJ9LCJyb2xlIjoiYXV0aGVudGljYXRlZCIsImFhbCI6ImFhbDEiLCJhbXIiOlt7Im1ldGhvZCI6InBhc3N3b3JkIiwidGltZXN0YW1wIjoxNzgwNzUwMTEwfV0sInNlc3Npb25faWQiOiJkNjEzMDYwNy1jMmRkLTRkMTMtYmJjNi1mMzg1NDgyZTAxMGYiLCJpc19hbm9ueW1vdXMiOmZhbHNlfQ.7lRRFCnYSU3ojTu6L_RXsooCEycT1pT54wQpdcZkmLzE3k5t7FJzrHaYUHcKlQS1YM2JFd3ptku4pbXkzry2xQ"
# Separate Groq key for eval — keeps eval traffic from competing with
# production's daily token quota. Get one free at console.groq.com.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROK_API_KEY not found in environment variables")

def _grade(score: float) -> str:
    if score >= 0.9: return "🟢 Excellent"
    if score >= 0.7: return "✅ Good"
    if score >= 0.5: return "🟡 Acceptable"
    return "🔴 Needs work"


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
     "ground_truth": "The document does not contain information about the Banker's Algorithm. A correct answer should say this topic is not in the uploaded documents."},
]


# ── EDIT: uncomment ONE line below to run a smaller batch ──────────────────
# QUESTIONS = QUESTIONS[:8]
# QUESTIONS = QUESTIONS[8:16]
# QUESTIONS = QUESTIONS[16:]

# ── Step 1: Validate token + API connection ────────────────────────────────
if BEARER_TOKEN == "":
    print("[Eval] ERROR: Set BEARER_TOKEN in evaluate.py")
    print("  1. Open your MindVault site")
    print("  2. DevTools → Network → any request → Headers → Authorization")
    print("  3. Copy the eyJ... value and paste it as BEARER_TOKEN")
    sys.exit(1)

if not GROQ_API_KEY:
    print("[Eval] ERROR: GROQ_API_KEY not found in backend/.env")
    sys.exit(1)

headers = {"Authorization": f"Bearer {BEARER_TOKEN}", "Content-Type": "application/json"}

# Verify token works against Render
print(f"[Eval] Verifying connection to {RENDER_API_URL}...")
test_res = requests.get(f"{RENDER_API_URL}/documents", headers=headers, timeout=60)
if test_res.status_code == 401:
    print("[Eval] ERROR: Token invalid or expired. Get a fresh one from DevTools.")
    sys.exit(1)
elif test_res.status_code != 200:
    print(f"[Eval] WARNING: /documents returned {test_res.status_code} — continuing anyway")
else:
    docs = test_res.json().get("documents", [])
    print(f"[Eval] Connected. User has {len(docs)} document(s) uploaded.")
    if not any("OS_Notes" in d.get("filename","") for d in docs):
        print("[Eval] WARNING: OS_Notes.pdf not found in this user's vault.")
        print("  Upload it first, then re-run evaluation.")


# ── Step 2: Run each question against live /query endpoint ─────────────────
print(f"\n[Eval] Running {len(QUESTIONS)} questions against {RENDER_API_URL}...")
print("─" * 65)

SESSION_ID = f"eval_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
records = []

for i, item in enumerate(QUESTIONS):
    q, gt = item["question"], item["ground_truth"]
    print(f"[{i+1:02d}/{len(QUESTIONS)}] {q[:60]}...")

    try:
        res = requests.post(
            f"{RENDER_API_URL}/query",
            headers=headers,
            json={"question": q, "session_id": SESSION_ID, "mode": RETRIEVAL_MODE},
            timeout=60
        )
        if res.status_code != 200:
            raise Exception(f"HTTP {res.status_code}: {res.text[:100]}")

        data       = res.json()
        answer     = data.get("answer", "")
        confidence = data.get("confidence", 0.0)
        chunks     = data.get("chunks", [])
        contexts   = [c["content"] for c in chunks] if chunks else [answer]

        records.append({
            "question": q, "answer": answer, "contexts": contexts,
            "ground_truth": gt, "confidence": confidence,
            "sources": data.get("sources", []),
        })
        print(f"         → {answer[:80]}...")
        print(f"           chunks={len(contexts)}  confidence={confidence:.2f}")

    except Exception as e:
        print(f"         ERROR: {e}")
        records.append({
            "question": q, "answer": f"ERROR: {e}",
            "contexts": ["ERROR"], "ground_truth": gt,
            "confidence": 0.0, "sources": [],
        })

print("─" * 65)


# ── Step 3: RAGAS evaluation ───────────────────────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from datasets import Dataset
    from langchain_groq import ChatGroq
except ImportError as e:
    print(f"\n[Eval] Missing dep: {e}")
    print("Run: pip install ragas==0.1.21 datasets pandas tabulate langchain-groq")
    sys.exit(1)

print(f"\n[Eval] Scoring {len(records)} records via RAGAS + Groq (~1-3 min)...")

# llama-3.1-8b-instant has much higher free-tier TPM/TPD than the 70b model --
# avoids the rate-limit cascade that corrupted the previous run.
_llm = LangchainLLMWrapper(ChatGroq(
    model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY,
    max_retries=5
))
for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
    metric.llm = _llm

# Wire fastembed for answer_relevancy if available
try:
    from fastembed import TextEmbedding as _FE
    from langchain.embeddings.base import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    class _FEW(Embeddings):
        def __init__(self): self._m = _FE("sentence-transformers/all-MiniLM-L6-v2")
        def embed_documents(self, texts): return [e.tolist() for e in self._m.embed(texts)]
        def embed_query(self, text): return list(self._m.embed([text]))[0].tolist()

    answer_relevancy.embeddings = LangchainEmbeddingsWrapper(_FEW())
    print("[Eval] fastembed wired for answer_relevancy embeddings.")
except Exception as e:
    print(f"[Eval] fastembed not available locally ({e}). answer_relevancy uses Groq embeddings.")

valid = [r for r in records if not r["answer"].startswith("ERROR")]
if not valid:
    print("[Eval] No valid records. Fix errors above.")
    sys.exit(1)

dataset = Dataset.from_dict({
    "question":     [r["question"]     for r in valid],
    "answer":       [r["answer"]       for r in valid],
    "contexts":     [r["contexts"]     for r in valid],
    "ground_truth": [r["ground_truth"] for r in valid],
})

from ragas.run_config import RunConfig
# Sequential-ish execution (max_workers=2) with generous retries/timeout
# to survive Groq free-tier rate limits without cascading failures.
run_config = RunConfig(max_workers=2, timeout=120, max_retries=5, max_wait=30)

try:
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        run_config=run_config,
        raise_exceptions=False,
    )
except Exception as e:
    print(f"[Eval] RAGAS failed: {e}")
    sys.exit(1)

scores = {
    "faithfulness":      round(result["faithfulness"],      4),
    "answer_relevancy":  round(result["answer_relevancy"],  4),
    "context_precision": round(result["context_precision"], 4),
    "context_recall":    round(result["context_recall"],    4),
}
overall = round(sum(scores.values()) / len(scores), 4)


# ── Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RAGAS EVALUATION — MindVault × OS_Notes.pdf")
print("=" * 65)

try:
    from tabulate import tabulate
    rows = [[m, s, _grade(s)] for m, s in scores.items()]
    rows += [["─"*22, "─"*6, "─"*12], ["OVERALL", overall, _grade(overall)]]
    print(tabulate(rows, headers=["Metric", "Score", "Grade"], tablefmt="rounded_outline"))
except ImportError:
    for m, s in scores.items():
        print(f"  {m:<25} {s:.4f}  {_grade(s)}")
    print(f"  {'OVERALL':<25} {overall:.4f}  {_grade(overall)}")

print()
print("PER-QUESTION CONFIDENCE:")
for r in records:
    flag = "⚠️ " if r["confidence"] < 0.4 else "✅ "
    print(f"  {flag}[{r['confidence']:.2f}] {r['question'][:65]}")

print()
print("METRIC GUIDE:")
print("  faithfulness      → grounding in docs (anti-hallucination)")
print("  answer_relevancy  → does answer address the question")
print("  context_precision → are retrieved chunks relevant")
print("  context_recall    → did retrieval find all needed chunks")
print("  <0.5 needs work | 0.5-0.7 ok | >0.7 good | >0.9 excellent")
print("=" * 65)

# Save
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"eval_{ts}.json")
with open(out_path, "w") as f:
    json.dump({"timestamp": ts, "scores": scores, "overall": overall,
               "per_question": records}, f, indent=2)
print(f"\n[Eval] Saved → {out_path}")