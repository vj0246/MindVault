from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
from rag.memory import (
    get_session_history,
    save_session_message,
    get_history_for_prompt
)

from rag.ingest import ingest_document
from rag.retrieve1 import query_rag
#from rag.retrieve import query_rag
from metadata.tracker import log_document, get_all_documents

app = FastAPI(title="MindVault API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "MindVault is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    allowed_types = ["application/pdf", "text/plain", "text/markdown"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF, TXT and MD files supported."
        )
    
    file_path = f"data/docs/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    chunks = ingest_document(file_path)
    log_document(
    filename=file.filename,
    path=file_path,
    chunk_count=len(chunks))

    return {
        "message": f"{file.filename} ingested successfully.",
        "chunks": len(chunks),
        "filename": file.filename
    }

class QueryRequest(BaseModel):
    question: str
    mode: str = "default"
    session_id: str = "default_session"

@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )
    history = get_session_history(req.session_id)
    result = query_rag(
        question=req.question,
        history=history,
        mode=req.mode
    )
    save_session_message(req.session_id, role="user", content=req.question)
    save_session_message(req.session_id, role="assistant", content=result["answer"])
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "mode": req.mode
    }
@app.get("/documents")
def list_documents():
    return {"documents": get_all_documents()}

#@app.get("/documents")
#def list_documents():
#    docs_path = "data/docs"
#    
#    if not os.path.exists(docs_path):
#        return {"documents": []}
#    
#    files = [
#        f for f in os.listdir(docs_path)
#        if f.endswith((".pdf", ".txt", ".md"))
#    ]
    
    return {"documents": files}

@app.delete("/session/{session_id}")
def clear_session_route(session_id: str):
    from rag.memory import clear_session
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}