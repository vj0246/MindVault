from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import uuid
from rag.db import get_supabase
from rag.ingest import ingest_document
from rag.retrieve1 import query_rag
from rag.memory import (
    get_session_history, save_session_message, clear_session_messages,
    list_chat_sessions, create_chat_session, rename_chat_session, delete_chat_session
)
from metadata.tracker import log_document, get_all_documents
from graph.extractor import extract_entities_and_relations
from graph.store import add_to_graph, get_related_nodes, get_full_graph
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="MindVault API")
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    sb = get_supabase()
    try:
        user = sb.auth.get_user(credentials.credentials)
        return user.user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/")
def root():
    return {"status": "MindVault is running"}

def _build_graph_for_chunks(chunks, filename: str, user_id: str):
    """Runs in the background after /upload responds. Extracts entities and
    relationships from up to 10 chunks and adds them to the knowledge graph."""
    for chunk in chunks:
        try:
            extracted = extract_entities_and_relations(chunk.page_content, source=filename)
            add_to_graph(extracted, user_id=user_id)
        except Exception as e:
            print(f"[Graph] Background extraction failed for a chunk: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, user=Depends(get_current_user)):
    try:
        sb = get_supabase()
        ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        os.makedirs("data/docs", exist_ok=True)
        file_path = f"data/docs/{file.filename}"
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        document_id = str(uuid.uuid4())
        actual_document_id = log_document(
            filename=file.filename,
            path=file_path,
            chunk_count=0,
            document_id=document_id,
            user_id=str(user.id)
        )

        try:
            chunks = ingest_document(file_path, document_id=actual_document_id, user_id=str(user.id))
        except ValueError as e:
            # Clean, user-facing error (empty/unreadable document) -- remove
            # the orphaned chunk_count=0 row created by log_document above
            # so it doesn't clutter the user's sidebar with a useless doc.
            sb.table("documents").delete().eq("id", actual_document_id).execute()
            raise HTTPException(status_code=400, detail=str(e))

        sb.table("documents").update(
            {"chunk_count": len(chunks)}
        ).eq("id", actual_document_id).execute()

        # Graph extraction runs AFTER the response is sent -- up to 10
        # sequential Groq calls here were adding 10-30s to upload latency
        # and risking Render's proxy timeout (502s on larger documents).
        if background_tasks is not None:
            background_tasks.add_task(
                _build_graph_for_chunks, chunks[:10], file.filename, str(user.id)
            )

        return {
            "message": f"{file.filename} ingested successfully.",
            "chunks": len(chunks),
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class RenameRequest(BaseModel):
    name: str

class QueryRequest(BaseModel):
    question: str
    mode: str = "default"
    session_id: str = "default_session"
    document_ids: list = []

class ExportRequest(BaseModel):
    session_id: str
    format: str = "markdown"

@app.post("/query")
def query(req: QueryRequest, user=Depends(get_current_user)):
    try:
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        history = get_session_history(req.session_id, user_id=str(user.id))
        result = query_rag(
            question=req.question,
            history=history,
            mode=req.mode,
            user_id=str(user.id),
            document_ids=req.document_ids or None
        )

        save_session_message(req.session_id, role="user", content=req.question, user_id=str(user.id))
        save_session_message(req.session_id, role="assistant", content=result["answer"], user_id=str(user.id))

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": result.get("chunks", []),
            "confidence": result.get("confidence", 0.0),
            "mode": req.mode,
            "intent": result.get("intent", "answer"),
            "related_concepts": result.get("related_concepts", [])
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents(user=Depends(get_current_user)):
    return {"documents": get_all_documents(user_id=str(user.id))}

@app.post("/export")
def export_session(req: ExportRequest, user=Depends(get_current_user)):
    try:
        history = get_session_history(req.session_id, user_id=str(user.id))
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

# ── Session management ────────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions_route(user=Depends(get_current_user)):
    return {"sessions": list_chat_sessions(str(user.id))}

@app.post("/sessions")
def create_session_route(user=Depends(get_current_user)):
    session_id = str(uuid.uuid4())
    session = create_chat_session(session_id, str(user.id))
    return {"session_id": session_id, "name": session["name"]}

@app.patch("/sessions/{session_id}/rename")
def rename_session_route(session_id: str, body: RenameRequest, user=Depends(get_current_user)):
    rename_chat_session(session_id, str(user.id), body.name)
    return {"ok": True}

@app.get("/sessions/{session_id}/history")
def get_history_route(session_id: str, user=Depends(get_current_user)):
    history = get_session_history(session_id, str(user.id))
    return {"history": history}

@app.delete("/sessions/{session_id}")
def delete_session_route(session_id: str, user=Depends(get_current_user)):
    delete_chat_session(session_id, str(user.id))
    return {"ok": True}

@app.get("/graph/{topic}")
def get_graph_topic(topic: str, user=Depends(get_current_user)):
    return get_related_nodes(topic, user_id=str(user.id))

@app.get("/graph")
def get_graph_all(user=Depends(get_current_user)):
    return get_full_graph(user_id=str(user.id))

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