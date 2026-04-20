import os
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from rag.memory import get_history_for_prompt
def load_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore_path = "vectorstore1"
    
    if not os.path.exists(vectorstore_path):
        return None
    
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def retrieve_chunks(question: str, vectorstore, k=5):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    docs = retriever.get_relevant_documents(question)
    return docs

#result=retrieve_chunks(question="main functions of operating system in 5 words?",vectorstore=load_vectorstore())

def generate_answer(question: str, docs, mode: str = "default", history: list = []):
    llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
    history_str = get_history_for_prompt(history, llm)
    #if history:
    #    history_str = "\n".join([
    #        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
    #        for m in history[-6:]
    #    ])
    context = "\n\n---\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    prompts = {
        "student": f"""You are a study assistant with memory of the conversation.
Answer using only the context below.
Use simple language and bullet points where helpful.

CONVERSATION HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

IMPORTANT: If the question refers to something from the conversation history 
like "first point", "you mentioned", "elaborate on that" — answer based on 
the conversation history first, then use document context to expand.
If the answer is not in context or history say: "This isn't in your uploaded documents."

Question: {question}

Answer:""",

        "lawyer": f"""You are a legal research assistant. Answer using only the context below.
Be precise and formal. Flag any ambiguities.
Use simple language and bullet points where helpful.

CONVERSATION HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

IMPORTANT: If the question refers to something from the conversation history 
like "first point", "you mentioned", "elaborate on that" — answer based on 
the conversation history first, then use document context to expand.
If the answer is not in context or history say: "This isn't in your uploaded documents."

Question: {question}

Answer:""",

        "developer": f"""You are a technical assistant. Answer using only the context below.
Be precise. Include implementation details if present in context.
Use simple language and bullet points where helpful.

CONVERSATION HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

IMPORTANT: If the question refers to something from the conversation history 
like "first point", "you mentioned", "elaborate on that" — answer based on 
the conversation history first, then use document context to expand.
If the answer is not in context or history say: "This isn't in your uploaded documents."

Question: {question}

Answer:""",

        "default": f"""Answer using only the context below.Use simple language and bullet points where helpful.

CONVERSATION HISTORY:
{history_str}

DOCUMENT CONTEXT:
{context}

IMPORTANT: If the question refers to something from the conversation history 
like "first point", "you mentioned", "elaborate on that" — answer based on 
the conversation history first, then use document context to expand.
If the answer is not in context or history say: "This isn't in your uploaded documents."

Question: {question}

Answer:"""
    }
    
    prompt = prompts.get(mode, prompts["default"])
    #llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
    response = llm.invoke(prompt)
    print(f"[Debug] LLM response: {response}")
    return response.content if response.content else "LLM returned empty response."
    #response = llm.invoke(prompt)
    #return response.content

def query_rag(question: str, history: list = [], mode: str = "default"):
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return {
            "answer": "No documents uploaded yet. Please upload a document first.",
            "sources": [],
        }
    
    docs = retrieve_chunks(question, vectorstore)
    
    if not docs:
        return {
            "answer": "Nothing relevant found in your documents for this question.",
            "sources": [],
        }
    
    answer = generate_answer(question, docs, mode)
    
    sources = list(set([
        f"{doc.metadata.get('source', 'Unknown')} - Page {doc.metadata.get('page', '?')}"
        for doc in docs
    ]))
    
    return {
        "answer": answer,
        "sources": sources,
    }



#from rag.retrieve import query_rag
#result = query_rag("what is an operating system", mode="student")
#print(result["answer"])
#print("---")
#print(result["sources"])


    