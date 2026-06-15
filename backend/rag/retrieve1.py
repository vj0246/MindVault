import os
from rag.db import get_supabase
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from graph.store import get_related_nodes
from dotenv import load_dotenv

load_dotenv()

_RERANKER = None

def _get_reranker():
    global _RERANKER
    if _RERANKER is None:
        from flashrank import Ranker
        _RERANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")
    return _RERANKER

def _rerank(query: str, chunks: list, top_k: int) -> list:
    """Cross-encoder re-rank via FlashRank. Falls back to RRF order if unavailable."""
    if not chunks:
        return chunks
    try:
        from flashrank import RerankRequest
        ranker = _get_reranker()
        passages = [{"id": i, "text": c.get("content", "")} for i, c in enumerate(chunks)]
        results = ranker.rerank(RerankRequest(query=query, passages=passages))
        reranked = []
        for r in results[:top_k]:
            chunk = chunks[r["id"]].copy()
            chunk["rerank_score"] = round(r["score"], 4)
            reranked.append(chunk)
        return reranked
    except Exception as e:
        print(f"[Retrieve] Re-rank failed, falling back to RRF order: {e}")
        return chunks[:top_k]

# ── Singletons ──────────────────────────────────────────────────────────────
# Render's free tier is 512MB. Creating new ChatGroq / Supabase clients per
# call opens new httpx connection pools that don't get cleaned up fast enough
# under rapid requests, causing memory to climb until the process is OOM-killed
# (manifests as 502s after a handful of queries). Cache and reuse instead.

_LLM_CACHE = {}

def get_llm(temperature: float = 0.0):
    # Default temperature=0: deterministic, minimal hallucination for factual RAG
    # Creative chains (summarize/compare/test) pass temperature=0.15
    #
    # FALLBACK: if 70b hits a rate limit (separate daily token pool per model
    # on Groq), automatically retry on 8b-instant instead of failing the
    # whole request. Each model has its own TPD quota, so this gives
    # effectively 2x headroom during traffic spikes.
    #
    # Only 2 distinct temperatures are used (0.0, 0.15) -> cache by temperature
    # so each ChatGroq/httpx client pair is created ONCE and reused.
    if temperature not in _LLM_CACHE:
        primary = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"]
        )
        fallback = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"]
        )
        _LLM_CACHE[temperature] = primary.with_fallbacks([fallback])
    return _LLM_CACHE[temperature]


from rag.embedder import EMBED_MODEL

def _rrf_merge(semantic: list, keyword: list, top_k: int, rrf_k: int = 60) -> list:
    """Reciprocal Rank Fusion — merges two ranked lists into one.
    Each returned chunk gets an 'rrf_score' field (raw RRF value,
    0 to 2/(rrf_k+1)) so callers can measure cross-method agreement."""
    scores = {}
    chunk_map = {}

    for rank, chunk in enumerate(semantic):
        key = chunk["content"][:120]
        scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
        chunk_map[key] = chunk

    for rank, chunk in enumerate(keyword):
        key = chunk["content"][:120]
        scores[key] = scores.get(key, 0) + 1 / (rrf_k + rank + 1)
        if key not in chunk_map:
            chunk_map[key] = chunk

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    result = []
    for k in ranked[:top_k]:
        chunk = chunk_map[k].copy()
        chunk["rrf_score"] = scores[k]
        result.append(chunk)
    return result


def retrieve_context(question: str, k: int = 5, user_id: str = None, document_ids: list = None) -> dict:
    supabase = get_supabase()
    embedding = list(EMBED_MODEL.embed([question]))[0].tolist()

    # --- Semantic search (vector similarity) ---
    sem_params = {
        "query_embedding": embedding,
        "match_count": k * 3,       # fetch wider pool for fusion
        "similarity_threshold": 0.2  # lower threshold, RRF handles re-ranking
    }
    if user_id:
        sem_params["p_user_id"] = user_id
    if document_ids:
        sem_params["p_document_ids"] = document_ids

    sem_result = supabase.rpc("match_chunks", sem_params).execute()
    semantic_chunks = sem_result.data or []

    # --- Keyword search (PostgreSQL FTS / BM25-style) ---
    keyword_chunks = []
    try:
        kw_params = {
            "query_text": question,
            "match_count": k * 2
        }
        if user_id:
            kw_params["p_user_id"] = user_id
        if document_ids:
            kw_params["p_document_ids"] = document_ids

        kw_result = supabase.rpc("keyword_search_chunks", kw_params).execute()
        keyword_chunks = kw_result.data or []
    except Exception as e:
        print(f"[Retrieve] Keyword search failed, falling back to semantic only: {e}")

    if not semantic_chunks and not keyword_chunks:
        return {"context": "", "sources": [], "chunks": []}

    # --- RRF fusion ---
    rrf_pool = _rrf_merge(semantic_chunks, keyword_chunks, top_k=k * 2)

    # --- Cross-encoder re-ranking (FlashRank) ---
    # RRF merges by rank position; cross-encoder scores exact query-passage relevance
    fused = _rerank(question, rrf_pool, top_k=k)

    # Label each chunk with its source so LLM can cite inline
    context_parts = []
    for c in fused:
        fname = c.get("filename", "Uploaded Document")
        context_parts.append(f"[Doc: {fname}]\n{c['content']}")
    context = "\n\n---\n\n".join(context_parts)

    sources = list(set([c.get("filename", "Uploaded Document") for c in fused]))
    chunks = [
        {
            "content": c["content"][:200],
            "similarity": round(c.get("similarity", c.get("rank", 0)), 3),
            "filename": c.get("filename", "Uploaded Document")
        }
        for c in fused
    ]
    # Confidence: combines BOTH retrieval signals.
    # - normalized_rrf (0-1): how strongly semantic + keyword AGREE on this chunk.
    #   1.0 = ranked #1 in both lists. 0.5 = ranked #1 in only one list.
    # - similarity (cosine, 0-1): raw semantic relevance, when available.
    # Blended 50/50 so a chunk found by both methods AND highly similar
    # scores higher than one found by either signal alone.
    RRF_K = 60
    max_rrf = 2 / (RRF_K + 1)  # theoretical max: rank 0 in both lists

    chunk_scores = []
    for c in fused:
        normalized_rrf = min(1.0, c.get("rrf_score", 0) / max_rrf)
        sim = c.get("similarity")
        if sim is not None and sim > 0:
            chunk_scores.append(0.5 * sim + 0.5 * normalized_rrf)
        else:
            chunk_scores.append(normalized_rrf)

    top3 = sorted(chunk_scores, reverse=True)[:3]
    if top3:
        avg_score = sum(top3) / len(top3)
        confidence = round(min(1.0, avg_score * (1 + 0.08 * min(len(fused) - 1, 4))), 3)
    else:
        confidence = 0.0
    return {"context": context, "sources": sources, "chunks": chunks, "confidence": confidence}

def format_history(history: list) -> list:
    messages = []
    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

def classify_intent(question: str) -> str:
    q = question.lower()
    
    compare_keywords = ["compare", "difference", "vs", "versus",
                       "contrast", "distinguish", "similarities",
                       "different between", "alike"]
    test_keywords = ["generate questions", "mcq", "quiz", "test me",
                    "generate mcqs", "create questions", "make questions",
                    "question on", "questions on", "questions about"]
    summarize_keywords = ["summarize", "summary", "overview", "revise",
                         "revision", "brief", "outline", "recap",
                         "give me all", "everything about"]

    for keyword in compare_keywords:
        if keyword in q:
            return "compare"
    for keyword in test_keywords:
        if keyword in q:
            return "test"
    for keyword in summarize_keywords:
        if keyword in q:
            return "summarize"

    llm = get_llm()
    intent_prompt = ChatPromptTemplate.from_template("""
Read the question and reply with exactly one word only.
No explanation. No punctuation. Just one word.

If comparing two things → compare
If generating questions or quiz → test  
If summarizing a topic → summarize
If asking a direct question → answer

Question: {question}

One word:
""")
    intent_chain = intent_prompt | llm | StrOutputParser()
    intent_raw = intent_chain.invoke({"question": question})
    intent = intent_raw.strip().lower().split()[0]
    
    valid_intents = ["compare", "test", "summarize", "answer"]
    return intent if intent in valid_intents else "answer"

def resolve_and_classify(question: str, history: list) -> dict:
    llm = get_llm()
    history_str = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-4:]
    ]) if history else "No history"

    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Do two things:

1. Rewrite the question to be fully explicit using history if needed.
   If already explicit, return it unchanged.

2. Classify the intent as exactly one word:
   compare / test / summarize / answer

History:
{history}

Question: {question}

Reply in this exact format:
RESOLVED: [rewritten question]
INTENT: [one word]
""")
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"history": history_str, "question": question})

    lines = result.strip().split("\n")
    resolved = question
    intent = "answer"

    for line in lines:
        if line.startswith("RESOLVED:"):
            resolved = line.replace("RESOLVED:", "").strip()
        elif line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip().lower()

    q = question.lower()
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast"]):
        intent = "compare"
    elif any(w in q for w in ["quiz", "mcq", "test me", "generate questions", "questions on"]):
        intent = "test"
    elif any(w in q for w in ["summarize", "summary", "overview", "revise", "revision"]):
        intent = "summarize"

    return {"resolved": resolved, "intent": intent}

def build_router_chain(history: list):
    llm = get_llm()
    router_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and the new question, decide if the question
refers to something already discussed in the history or needs new document retrieval.

History:
{history}

Question: {question}

Reply with ONLY one word:
- "history" if question refers to previous conversation
- "retrieve" if question needs document search
""")
    return router_prompt | llm | StrOutputParser()

def build_retrieval_chain(mode: str = "default", user_id: str = None, document_ids: list = None):
    prompts = {
        "student": """You are helping a student prepare for exams.
STRICT RULE: Use ONLY the context below. Never use outside knowledge. If the answer is not in context, respond EXACTLY: "This isn't in your uploaded documents." Cite the source file after each fact in brackets e.g. [notes.pdf].
You are helping a student. Use bullet points. Bold key terms.
Maximum 150 words unless detail is specifically asked for.

Context: {context}
Question: {question}
Answer:""",
        "lawyer": """You are a legal research assistant.
STRICT RULE: Use ONLY the context below. Never use outside knowledge. If the answer is not in context, respond EXACTLY: "This isn't in your uploaded documents." Cite the source file after each fact in brackets e.g. [notes.pdf].
You are a legal research assistant. Answer formally. Flag ambiguities.
Maximum 150 words unless detail is specifically asked for.

Context: {context}
Question: {question}
Answer:""",
        "developer": """You are a technical assistant.
STRICT RULE: Use ONLY the context below. Never use outside knowledge. If the answer is not in context, respond EXACTLY: "This isn't in your uploaded documents." Cite the source file after each fact in brackets e.g. [notes.pdf].
You are a technical assistant. Include implementation details if present.
Maximum 150 words unless detail is specifically asked for.

Context: {context}
Question: {question}
Answer:""",
        "default": """STRICT RULE: Use ONLY the context below. Never use outside knowledge. If the answer is not in context, respond EXACTLY: "This isn't in your uploaded documents." Cite the source file after each fact in brackets e.g. [notes.pdf].
Be clear and concise. Maximum 150 words.

Context: {context}
Question: {question}
Answer:"""
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.get(mode, prompts["default"])),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    llm = get_llm()

    def run_chain(input: dict) -> dict:
        retrieved = retrieve_context(input["question"], k=5, user_id=user_id, document_ids=document_ids)
        result = (prompt | llm | StrOutputParser()).invoke({
            "context": retrieved["context"],
            "question": input["question"],
            "history": input.get("history", [])
        })
        return {"answer": result, "sources": retrieved["sources"], "chunks": retrieved.get("chunks", []), "confidence": retrieved.get("confidence", 0.0)}

    return run_chain

def summarize_chain(question: str, mode: str = "default", user_id: str = None, document_ids: list = None) -> dict:
    llm = get_llm(temperature=0.15)
    retrieved = retrieve_context(question, k=8, user_id=user_id, document_ids=document_ids)
    if not retrieved["context"]:
        return {"answer": "Nothing relevant found in your documents.", "sources": [], "chunks": []}

    prompts = {
        "student": """Summarize as exam revision notes using only the context below.
Group by topic. Use bullet points. Bold key terms.
Context:\n{context}\nRequest: {question}\nRevision Summary:""",
        "lawyer": """Summarize as an executive brief using only the context below.
Include key points, obligations, and risks.
Context:\n{context}\nRequest: {question}\nExecutive Brief:""",
        "default": """Summarize using ONLY the context below. Never add outside knowledge.
Use headers and bullet points where helpful.
Context:\n{context}\nRequest: {question}\nSummary:"""
    }

    prompt = ChatPromptTemplate.from_template(prompts.get(mode, prompts["default"]))
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": retrieved["context"],
        "question": question
    })
    return {"answer": answer, "sources": retrieved["sources"], "chunks": retrieved.get("chunks", []), "confidence": retrieved.get("confidence", 0.0)}

def comparison_chain(question: str, mode: str = "default", user_id: str = None, document_ids: list = None) -> dict:
    llm = get_llm(temperature=0.15)
    retrieved = retrieve_context(question, k=8, user_id=user_id, document_ids=document_ids)
    if not retrieved["context"]:
        return {"answer": "Nothing relevant found in your documents.", "sources": [], "chunks": []}

    prompt = ChatPromptTemplate.from_template("""
Compare using ONLY the context below. Never use outside knowledge or assumptions.
Structure your response as:

**Similarities:**
- point 1

**Differences:**
- point 1

**Key Insight:**
One sentence summarizing the most important distinction.

Context:\n{context}\nComparison request: {question}\nComparison:
""")
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": retrieved["context"],
        "question": question
    })
    return {"answer": answer, "sources": retrieved["sources"], "chunks": retrieved.get("chunks", []), "confidence": retrieved.get("confidence", 0.0)}

def test_generator_chain(question: str, user_id: str = None, document_ids: list = None) -> dict:
    llm = get_llm(temperature=0.15)
    retrieved = retrieve_context(question, k=8, user_id=user_id, document_ids=document_ids)
    if not retrieved["context"]:
        return {"answer": "Nothing relevant found in your documents.", "sources": [], "chunks": []}

    prompt = ChatPromptTemplate.from_template("""
Generate questions based STRICTLY on the content below. Every question and answer must be traceable to the context.
Mix of 3 MCQs and 2 short answer questions.
Give answers at the end.

For MCQs:
Q1. [question]
A) option  B) option  C) option  D) option
Answer: [correct option]

For short answer:
Q4. [question]
Answer: [expected answer]

Content:\n{context}\nTopic: {question}\nQuestions:
""")
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": retrieved["context"],
        "question": question
    })
    return {"answer": answer, "sources": retrieved["sources"], "chunks": retrieved.get("chunks", []), "confidence": retrieved.get("confidence", 0.0)}

def query_with_attachment(question: str, attachment_text: str, attachment_name: str,
                           history: list = [], mode: str = "default",
                           user_id: str = None, document_ids: list = None) -> dict:
    """One-off query with an attached file (image/PDF/TXT/MD/DOCX) for THIS
    message only -- not stored in the vector DB. Combines the attachment's
    extracted content with normal RAG retrieval from the user's vault."""
    llm = get_llm()

    # Normal RAG retrieval still runs, so the attachment is *additional*
    # context on top of the user's existing documents, not a replacement.
    retrieved = retrieve_context(question, k=5, user_id=user_id, document_ids=document_ids)

    # Cap attachment text -- vision descriptions are already capped at the
    # source (max_tokens=2000 in load_image_via_groq); for text documents,
    # truncate to keep the combined prompt a reasonable size.
    attachment_text = (attachment_text or "").strip()[:4000]

    # Build two SEPARATE, clearly-labeled sections rather than merging into
    # one blob. Mixing them caused the model to answer meta-questions about
    # the attachment ("which document is this from?") using vault content
    # instead of the attachment itself.
    attachment_section = (
        f"=== ATTACHED FILE: {attachment_name} ===\n{attachment_text}"
        if attachment_text else
        f"=== ATTACHED FILE: {attachment_name} ===\n(No readable content could be extracted from this file.)"
    )
    vault_section = (
        f"=== YOUR VAULT DOCUMENTS ===\n{retrieved['context']}"
        if retrieved["context"] else
        "=== YOUR VAULT DOCUMENTS ===\n(No relevant content found in your vault.)"
    )

    history_str = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-4:]
    ]) if history else "No history"

    prompt = ChatPromptTemplate.from_template("""You have two sources of information below: an ATTACHED FILE the user just shared, and YOUR VAULT DOCUMENTS (the user's existing knowledge base).

PRIORITY RULE: If the question asks about the attachment itself -- e.g. "what does this show", "describe this image", "what is this file", "which document does this belong to/match" -- answer ONLY from the ATTACHED FILE section. Do not guess a vault filename for content that is actually IN the attachment.

For general knowledge questions, prefer YOUR VAULT DOCUMENTS, and use the attachment only as supporting context if relevant.

STRICT RULE: Use ONLY the information in these two sections. Never use outside knowledge. If the answer isn't in either section, respond EXACTLY: "This isn't in your uploaded documents or attached file." Cite the source after each fact in brackets -- use the attached file's name for content from it, or the vault document's filename for content from there. Never cite a vault filename for something that came from the attachment.

{attachment_section}

{vault_section}

Recent conversation:
{history}

Question: {question}

Be clear and concise. Maximum 200 words.
Answer:""")

    answer = (prompt | llm | StrOutputParser()).invoke({
        "attachment_section": attachment_section,
        "vault_section": vault_section,
        "history": history_str,
        "question": question
    })

    sources = list(retrieved.get("sources", []))
    if attachment_text:
        sources.insert(0, f"{attachment_name} (attached)")

    return {
        "answer": answer,
        "sources": sources,
        "chunks": retrieved.get("chunks", []),
        "confidence": retrieved.get("confidence", 0.0),
        "intent": "answer",
        "related_concepts": []
    }

def query_rag(question: str, history: list = [], mode: str = "default", user_id: str = None, document_ids: list = None) -> dict:
    print(f"[Debug] question={question}, history={len(history)}")

    supabase = get_supabase()
    check = supabase.table("chunks").select("id")\
        .eq("user_id", user_id).limit(1).execute() if user_id else \
        supabase.table("chunks").select("id").limit(1).execute()
    
    if not check.data:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "sources": [],
            "intent": "answer",
            "related_concepts": []
        }

    formatted_history = format_history(history)
    llm = get_llm()

    if history:
        combined = resolve_and_classify(question, history)
        resolved_question = combined["resolved"]
        intent = combined["intent"]
    else:
        resolved_question = question
        intent = classify_intent(question)

    REFERENCE_WORDS = ["above", "that", "it", "previous", "you mentioned",
                       "first point", "second point", "elaborate", "expand",
                       "more detail", "explain more", "go deeper", "what about"]

    needs_router = any(w in resolved_question.lower() for w in REFERENCE_WORDS)
    decision = "retrieve"

    if history and intent == "answer" and needs_router:
        router_chain = build_router_chain(history)
        history_str = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ])
        decision = router_chain.invoke({
            "history": history_str,
            "question": resolved_question
        }).strip().lower()

    if decision == "history":
        history_str = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ])
        history_prompt = ChatPromptTemplate.from_template("""
Answer using only the conversation history below.
Do not make up information.

Conversation History:\n{history}
Question: {question}
Answer:""")
        answer = (history_prompt | llm | StrOutputParser()).invoke({
            "history": history_str,
            "question": resolved_question
        })
        return {
            "answer": answer,
            "sources": ["conversation history"],
            "intent": intent,
            "related_concepts": get_related_nodes(resolved_question, user_id=user_id).get("nodes", [])
        }

    elif intent == "compare":
        result = comparison_chain(resolved_question, mode, user_id=user_id, document_ids=document_ids)
    elif intent == "test":
        result = test_generator_chain(resolved_question, user_id=user_id, document_ids=document_ids)
    elif intent == "summarize":
        result = summarize_chain(resolved_question, mode, user_id=user_id, document_ids=document_ids)
    else:
        chain_fn = build_retrieval_chain(mode, user_id=user_id, document_ids=document_ids)
        result = chain_fn({
            "question": resolved_question,
            "history": formatted_history
        })

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks": result.get("chunks", []),
        "confidence": result.get("confidence", 0.0),
        "intent": intent,
        "related_concepts": get_related_nodes(resolved_question, user_id=user_id).get("nodes", [])
    }