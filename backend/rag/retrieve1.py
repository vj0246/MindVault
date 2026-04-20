import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

def get_llm():
    return ChatOllama(model="llama3.2:3b", temperature=0.2)

#not in retrieve.py
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
        "student": """You are a study assistant. Answer using only the context below.
Use simple language and bullet points where helpful.
If answer is not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}""",

        "lawyer": """You are a legal research assistant. Answer using only the context below.
Be precise and formal. Flag any ambiguities.
If answer is not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}""",

        "developer": """You are a technical assistant. Answer using only the context below.
Be precise. Include implementation details if present.
If answer is not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}""",

        "default": """Answer using only the context below.
If answer is not in context say: "This isn't in your uploaded documents."

Context: {context}
Question: {question}"""
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

'''def query_rag(question: str, history: list = [], mode: str = "default") -> dict:
    vectorstore = load_vectorstore()
    print(f"[Debug] question={question}")
    print(f"[Debug] history length={len(history)}")
    print(f"[Debug] first history item={history[0] if history else 'EMPTY'}")
    if vectorstore is None:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "sources": [],
        }
    
    formatted_history = format_history(history)
    llm = get_llm()
    
    # Step 1 — Route the question
    if history:
        print(f"[Debug] History exists, running router...")
        router_chain = build_router_chain(history)
        history_str = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ])
        print(f"[Debug] Invoking router with question: {question}")
        decision = router_chain.invoke({
            "history": history_str,
            "question": question
        }).strip().lower()
        print(f"[Router] Decision: {decision}")
    else:
        decision = "retrieve"
    print(f"[Router] Decision: {decision}")
    # Step 2 — Route to right chain
    if decision == "history":
        history_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer based on conversation history only."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        history_chain = history_prompt | llm | StrOutputParser()
        
        answer = history_chain.invoke({
            "history": formatted_history,
            "question": question
        })
        
        return {
            "answer": answer,
            "sources": ["conversation history"],
        }
    
    else:
        retrieval_chain = build_retrieval_chain(vectorstore, mode)
        
        result = retrieval_chain.invoke({
            "question": question,
            "history": formatted_history,
        })
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }'''

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
    
    # Step 1 — resolve vague references using history
    resolved_question = question
    if history:
        resolved_question = resolve_context_from_history(question, history)
        print(f"[Resolver] Original: {question}")
        print(f"[Resolver] Resolved: {resolved_question}")
    
    # Step 2 — classify intent on resolved question
    intent = classify_intent(resolved_question)
    print(f"[Intent] {intent}")
    
    # Step 3 — for answer intent check if history can answer directly
    decision = "retrieve"
    if history and intent == "answer":
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
    
    # Step 4 — route to right chain
    if decision == "history":
        history_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer based on conversation history only."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        history_chain = history_prompt | llm | StrOutputParser()
        answer = history_chain.invoke({
            "history": formatted_history,
            "question": resolved_question
        })
        return {
            "answer": answer,
            "sources": ["conversation history"],
            "intent": intent
        }
    
    elif intent == "compare":
        result = comparison_chain(resolved_question, vectorstore, mode)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent
        }
    
    elif intent == "test":
        result = test_generator_chain(resolved_question, vectorstore)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent
        }
    
    elif intent == "summarize":
        result = summarize_chain(resolved_question, vectorstore, mode)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "intent": intent
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
            "intent": intent
        }