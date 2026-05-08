from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
from rag.ingest import ingest_document
from rag.retrieve1 import query_rag
from rag.memory import get_session_history, save_session_message, clear_session
from metadata.tracker import log_document, get_all_documents
from graph.extractor import extract_entities_and_relations
from graph.store import add_to_graph, get_related_nodes, get_full_graph

app = FastAPI(title="MindVault API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "MindVault is running"}

# ─── Upload ───────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        allowed_types = ["application/pdf", "text/plain", "text/markdown"]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT and MD files supported."
            )

        file_path = f"data/docs/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[Upload] File saved: {file_path}")
        
        chunks = ingest_document(file_path)
        print(f"[Upload] Chunks created: {len(chunks)}")

        log_document(
            filename=file.filename,
            path=file_path,
            chunk_count=len(chunks)
        )
        print(f"[Upload] Document logged.")

        print(f"[Graph] Building knowledge graph from {min(10, len(chunks))} chunks...")
        for chunk in chunks[:10]:
            extracted = extract_entities_and_relations(
    chunk.page_content,
    source=file.filename
)
            add_to_graph(extracted)
        print(f"[Graph] Knowledge graph updated.")

        return {
            "message": f"{file.filename} ingested successfully.",
            "chunks": len(chunks),
            "filename": file.filename
        }
    except Exception as e:
        import traceback
        print(f"[Upload ERROR] {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ─── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    mode: str = "default"
    session_id: str = "default_session"

class ExportRequest(BaseModel):
    session_id: str
    format: str = "markdown"

@app.post("/query")
def query(req: QueryRequest):
    try:
        if not req.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        history = get_session_history(req.session_id)
        print(f"[Debug] Session: {req.session_id}, History: {len(history)}")

        result = query_rag(
            question=req.question,
            history=history,
            mode=req.mode
        )

        save_session_message(req.session_id, role="user", content=req.question)
        save_session_message(req.session_id, role="assistant", content=result["answer"])
        # Get related graph nodes for the question
        related = get_related_nodes(req.question.split()[0])
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "mode": req.mode,
            "intent": result.get("intent", "answer"),
            "related_concepts": related.get("nodes", [])
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ─── Documents ────────────────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    return {"documents": get_all_documents()}

# ─── Export ───────────────────────────────────────────────────────────────────

@app.post("/export")
def export_session(req: ExportRequest):
    try:
        history = get_session_history(req.session_id)
        if not history:
            return {"report": "# MindVault Session Export\n\n_No messages in this session._"}

        lines = ["# MindVault Session Export\n"]
        lines.append(f"**Session ID:** `{req.session_id}`\n")
        lines.append(f"**Messages:** {len(history)}\n")
        lines.append("---\n")

        for msg in history:
            role = "**You**" if msg["role"] == "user" else "**MindVault**"
            ts = msg.get("timestamp", "")
            lines.append(f"### {role}  \n*{ts}*\n")
            lines.append(msg["content"] + "\n")
            lines.append("---\n")

        report = "\n".join(lines)
        return {"report": report, "session_id": req.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── Session ──────────────────────────────────────────────────────────────────

@app.delete("/session/{session_id}")
def clear_session_route(session_id: str):
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared."}

# ─── Graph ────────────────────────────────────────────────────────────────────
@app.get("/graph/{topic}")
def get_graph_topic(topic: str):
    return get_related_nodes(topic)

@app.get("/graph")
def get_graph_all():
    return get_full_graph()

# ─── Preflight handler ────────────────────────────────────────────────────────

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )