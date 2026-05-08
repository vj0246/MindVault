import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from graph.store import get_related_nodes

def get_llm():
    return ChatOllama(model="llama3.2:3b", temperature=0.2)

def classify_intent(question: str) -> str:
    q = question.lower()
    
    # Keyword detection first — fast and reliable
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
            print(f"[Intent] Keyword match: compare")
            return "compare"
    
    for keyword in test_keywords:
        if keyword in q:
            print(f"[Intent] Keyword match: test")
            return "test"
    
    for keyword in summarize_keywords:
        if keyword in q: 
            print(f"[Intent] Keyword match: summarize")
            return "summarize"
    
    # No keyword match — use LLM as fallback
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
    print(f"[Intent Raw]: '{intent_raw}'")
    intent = intent_raw.strip().lower().split()[0]
    
    valid_intents = ["compare", "test", "summarize", "answer"]
    if intent not in valid_intents:
        intent = "answer"
    
    print(f"[Intent] LLM classified: {intent}")
    return intent

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore_path = "vectorstore1"
    
    if not os.path.exists(vectorstore_path):
        return None
    
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def format_history(history: list) -> list:
    messages = []
    for msg in history[-6:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


def build_retrieval_chain(vectorstore, mode: str = "default"):
    
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

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def retrieve_context(input: dict) -> dict:
        docs = retriever.get_relevant_documents(input["question"])
        context = "\n\n---\n\n".join([
            f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in docs
        ])
        sources = list(set([
            f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} - Page {doc.metadata.get('page', '?')}"\
            for doc in docs
        ]))
        return {
            "context": context,
            "question": input["question"],
            "history": input.get("history", []),
            "sources": sources 
        }

    llm = get_llm()

    chain = (
        RunnableLambda(retrieve_context)
        | RunnablePassthrough.assign(
            answer = prompt | llm | StrOutputParser()
        )
    )

    return chain

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

    router_chain = router_prompt | llm | StrOutputParser()
    
    return router_chain

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
    result = chain.invoke({
        "history": history_str,
        "question": question
    })
    
    lines = result.strip().split("\n")
    resolved = question
    intent = "answer"
    
    for line in lines:
        if line.startswith("RESOLVED:"):
            resolved = line.replace("RESOLVED:", "").strip()
        elif line.startswith("INTENT:"):
            intent = line.replace("INTENT:", "").strip().lower()
    
    # Safety — keyword override
    q = question.lower()
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast"]):
        intent = "compare"
    elif any(w in q for w in ["quiz", "mcq", "test me", "generate questions", "questions on"]):
        intent = "test"
    elif any(w in q for w in ["summarize", "summary", "overview", "revise", "revision"]):
        intent = "summarize"
    
    print(f"[Combined] Resolved: {resolved}")
    print(f"[Combined] Intent: {intent}")
    
    return {"resolved": resolved, "intent": intent}

def resolve_context_from_history(question: str, history: list) -> str:
    llm = get_llm()
    
    resolve_prompt = ChatPromptTemplate.from_template("""
Given the conversation history and a question, rewrite the question to be 
fully explicit and self contained. If the question is already explicit 
return it unchanged.

History:
{history}

Question: {question}

Rewritten question:
""")
    
    resolve_chain = resolve_prompt | llm | StrOutputParser()
    
    history_str = "\n".join([
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]
    ])
    
    return resolve_chain.invoke({
        "history": history_str,
        "question": question
    }).strip()

def summarize_chain(question: str, vectorstore, mode: str = "default") -> dict:
    llm = get_llm()
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return {
            "answer": "Nothing relevant found in your documents.",
            "sources": []
        }
    
    context = "\n\n---\n\n".join([
        f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    sources = list(set([
        f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} - Page {doc.metadata.get('page', '?')}"\
        for doc in docs
    ]))
    
    prompts = {
        "student": """Summarize as exam revision notes using only the context below.
Group by topic. Use bullet points. Bold key terms.
Be concise — prioritize high value content only.
Only use information from the context below.

Context:
{context}

Request: {question}

Revision Summary:""",

        "lawyer": """Summarize as an executive brief using only the context below.
Include key points, obligations, and risks.
Only use information from the context below.

Context:
{context}

Request: {question}

Executive Brief:""",

        "default": """Summarize clearly and concisely using only the context below.
Use headers and bullet points where helpful.
Only use information from the context below.

Context:
{context}

Request: {question}

Summary:"""
    }
    
    prompt = ChatPromptTemplate.from_template(
        prompts.get(mode, prompts["default"])
    )
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return {"answer": answer, "sources": sources}

def comparison_chain(question: str, vectorstore, mode: str = "default") -> dict:
    llm = get_llm()
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return {
            "answer": "Nothing relevant found in your documents.",
            "sources": []
        }
    
    context = "\n\n---\n\n".join([
        f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    sources = list(set([
        f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} - Page {doc.metadata.get('page', '?')}"\
        for doc in docs
    ]))
    
    compare_prompt = ChatPromptTemplate.from_template("""
Compare the concepts asked about using ONLY the context below.
Structure your response as:

**Similarities:**
- point 1
- point 2

**Differences:**
- point 1
- point 2

**Key Insight:**
One sentence summarizing the most important distinction.

If insufficient information exists say so explicitly.
Only use information from the context below.

Context:
{context}

Comparison request: {question}

Comparison:
""")
    
    compare = compare_prompt | llm | StrOutputParser()
    
    answer = compare.invoke({
        "context": context,
        "question": question
    })
    
    return {"answer": answer, "sources": sources}


def test_generator_chain(question: str, vectorstore) -> dict:
    llm = get_llm()
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return {
            "answer": "Nothing relevant found in your documents.",
            "sources": []
        }
    
    context = "\n\n---\n\n".join([
        f"[Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    sources = list(set([
        f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} - Page {doc.metadata.get('page', '?')}"\
        for doc in docs
    ]))
    
    test_prompt = ChatPromptTemplate.from_template("""
Generate questions based ONLY on the content below.
Mix of 3 MCQs and 2 short answer questions.
Strictly from the documents and context you have.
Also give ansers at the end.

For MCQs use this format:
Q1. [question]
A) option
B) option  
C) option
D) option
Answer: [correct option]

For short answer use this format:
Q4. [question]
Answer: [expected answer]

Only use information from the context below.
Do not use any outside knowledge.

Content:
{context}

Topic requested: {question}

Questions:
""")
    
    test = test_prompt | llm | StrOutputParser()
    
    answer = test.invoke({
        "context": context,
        "question": question
    })
    
    return {"answer": answer, "sources": sources}



def query_rag(question: str, history: list = [], mode: str = "default") -> dict:
    print(f"[Debug] question={question}")
    print(f"[Debug] history length={len(history)}")
    
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "sources": [],
        }
    
    formatted_history = format_history(history)
    llm = get_llm()
    #step 1+ step 2
    if history:
        combined = resolve_and_classify(question, history)
        resolved_question = combined["resolved"]
        intent = combined["intent"]
    else:
        resolved_question = question
        intent = classify_intent(question)
    # Step 3 — for answer intent check if history can answer directly
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
        print(f"[Router] Decision: {decision}")
    else:
        print(f"[Router] Skipped — no reference words detected")
    
    # Step 4 — route to right chain
    if decision == "history":
        # Answer from conversation history using LLM
        history_str = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ])
        history_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the conversation history below.
Do not make up information. If the answer is not in the history, say so.

Conversation History:
{history}

Question: {question}

Answer:""")
        history_chain = history_prompt | llm | StrOutputParser()
        answer = history_chain.invoke({
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
        result = comparison_chain(resolved_question, vectorstore, mode)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent,
            "related_concepts": get_related_nodes(resolved_question).get("nodes", [])
        }
    
    elif intent == "test":
        result = test_generator_chain(resolved_question, vectorstore)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent,
            "related_concepts": get_related_nodes(resolved_question).get("nodes", [])
        }
    
    elif intent == "summarize":
        result = summarize_chain(resolved_question, vectorstore, mode)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent,
            "related_concepts": get_related_nodes(resolved_question).get("nodes", [])
        }
    
    else:
        retrieval_chain = build_retrieval_chain(vectorstore, mode)
        result = retrieval_chain.invoke({
            "question": resolved_question,
            "history": formatted_history,
        })
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "intent": intent,
            "related_concepts": get_related_nodes(resolved_question).get("nodes", [])
        }