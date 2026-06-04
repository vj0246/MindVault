import os
#from sentence_transformers import SentenceTransformer
from supabase import create_client
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from graph.store import get_related_nodes
from dotenv import load_dotenv

load_dotenv()

#EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
from fastembed import TextEmbedding
def get_embed_model():
    return TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    
def get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"]
    )
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.environ["GROQ_API_KEY"]
    )


def retrieve_context(question: str, k: int = 5) -> dict:
    supabase = get_supabase()
    embed_model = get_embed_model()
    embedding = list(embed_model.embed([question]))[0].tolist()
    result = supabase.rpc("match_chunks", {
        "query_embedding": embedding,
        "match_count": k
    }).execute()
    
    chunks = result.data or []
    if not chunks:
        return {"context": "", "sources": []}
    
    context = "\n\n---\n\n".join([c["content"] for c in chunks])
    sources = list(set([c.get("filename", "Uploaded Document") for c in chunks]))
    
    return {"context": context, "sources": sources}

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

def build_retrieval_chain(mode: str = "default"):
    prompts = {
        "student": """You are helping a student prepare for exams.
Give direct, exam-ready answers only from the context below.
Use bullet points. Bold key terms.
Maximum 150 words unless detail is specifically asked for.
If not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}
Answer:""",

        "lawyer": """You are a legal research assistant.
Answer precisely and formally using only the context below.
Flag ambiguities. Cite specific sections where possible.
Maximum 150 words unless detail is specifically asked for.
If not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}
Answer:""",

        "developer": """You are a technical assistant.
Answer precisely using only the context below.
Include implementation details if present in context.
Maximum 150 words unless detail is specifically asked for.
If not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}
Answer:""",

        "default": """Answer using only the context below.
Be clear and concise. Maximum 150 words.
If not in context say: "This isn't in your uploaded documents."

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
        retrieved = retrieve_context(input["question"], k=5)
        result = (prompt | llm | StrOutputParser()).invoke({
            "context": retrieved["context"],
            "question": input["question"],
            "history": input.get("history", [])
        })
        return {"answer": result, "sources": retrieved["sources"]}

    return run_chain

def summarize_chain(question: str, mode: str = "default") -> dict:
    llm = get_llm()
    retrieved = retrieve_context(question, k=8)
    if not retrieved["context"]:
        return {"answer": "Nothing relevant found in your documents.", "sources": []}

    prompts = {
        "student": """Summarize as exam revision notes using only the context below.
Group by topic. Use bullet points. Bold key terms.
Context:\n{context}\nRequest: {question}\nRevision Summary:""",
        "lawyer": """Summarize as an executive brief using only the context below.
Include key points, obligations, and risks.
Context:\n{context}\nRequest: {question}\nExecutive Brief:""",
        "default": """Summarize clearly and concisely using only the context below.
Use headers and bullet points where helpful.
Context:\n{context}\nRequest: {question}\nSummary:"""
    }

    prompt = ChatPromptTemplate.from_template(prompts.get(mode, prompts["default"]))
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": retrieved["context"],
        "question": question
    })
    return {"answer": answer, "sources": retrieved["sources"]}

def comparison_chain(question: str, mode: str = "default") -> dict:
    llm = get_llm()
    retrieved = retrieve_context(question, k=8)
    if not retrieved["context"]:
        return {"answer": "Nothing relevant found in your documents.", "sources": []}

    prompt = ChatPromptTemplate.from_template("""
Compare the concepts asked about using ONLY the context below.
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
    return {"answer": answer, "sources": retrieved["sources"]}

def test_generator_chain(question: str) -> dict:
    llm = get_llm()
    retrieved = retrieve_context(question, k=8)
    if not retrieved["context"]:
        return {"answer": "Nothing relevant found in your documents.", "sources": []}

    prompt = ChatPromptTemplate.from_template("""
Generate questions based ONLY on the content below.
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
    return {"answer": answer, "sources": retrieved["sources"]}

def query_rag(question: str, history: list = [], mode: str = "default") -> dict:
    print(f"[Debug] question={question}, history={len(history)}")

    # Check if any chunks exist in Supabase
    supabase = get_supabase()
    check = supabase.table("chunks").select("id").limit(1).execute()
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
            "related_concepts": get_related_nodes(resolved_question).get("nodes", [])
        }

    elif intent == "compare":
        result = comparison_chain(resolved_question, mode)
    elif intent == "test":
        result = test_generator_chain(resolved_question)
    elif intent == "summarize":
        result = summarize_chain(resolved_question, mode)
    else:
        chain_fn = build_retrieval_chain(mode)
        result = chain_fn({
            "question": resolved_question,
            "history": formatted_history
        })

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "intent": intent,
        "related_concepts": get_related_nodes(resolved_question).get("nodes", [])
    }