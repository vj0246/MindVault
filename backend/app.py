from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
import uuid
from rag.db import get_supabase
from rag.ingest import ingest_document, load_document, load_image_via_groq, IMAGE_EXTENSIONS
from rag.retrieve1 import query_rag, query_with_attachment, stream_rag
from rag.memory import (
    get_session_history, save_session_message, clear_session_messages,
    list_chat_sessions, create_chat_session, rename_chat_session, delete_chat_session,
    generate_share_token, get_shared_session, revoke_share_token
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

@app.post("/query/stream")
async def query_stream(req: QueryRequest, user=Depends(get_current_user)):
    """Streaming version of /query. Returns SSE (text/event-stream).
    Events:
      {"type":"meta",  "sources":[...], "intent":"...", "confidence":0.8, "chunks":[...]}
      {"type":"token", "text":"hello "}
      {"type":"done",  "related_concepts":[...], "full_answer":"..."}
    Frontend should accumulate tokens, then save full_answer from done event.
    """
    history = get_session_history(req.session_id, user_id=str(user.id))

    async def event_generator():
        full_answer = ""
        try:
            for event in stream_rag(
                question=req.question,
                history=history,
                mode=req.mode,
                user_id=str(user.id),
                document_ids=req.document_ids or None
            ):
                if '"type": "done"' in event:
                    import json as _json
                    data = _json.loads(event.replace("data: ", "").strip())
                    full_answer = data.get("full_answer", "")
                yield event
        except Exception as e:
            import json as _json
            yield f'data: {_json.dumps({"type": "error", "message": str(e)})}\n\n'

        # Save to session history after stream completes
        if full_answer:
            save_session_message(req.session_id, role="user", content=req.question, user_id=str(user.id))
            save_session_message(req.session_id, role="assistant", content=full_answer, user_id=str(user.id))

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/query-with-attachment")
async def query_with_attachment_route(
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form("default_session"),
    mode: str = Form("default"),
    document_ids: str = Form("[]"),
    user=Depends(get_current_user)
):
    """One-off query with an attached image/document. Extracted content is
    used only for this single answer -- not stored in the user's vault."""
    import json
    import tempfile

    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        ext = os.path.splitext(file.filename)[1].lower()
        ALLOWED = {".pdf", ".txt", ".md", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        if ext not in ALLOWED:
            raise HTTPException(status_code=400, detail=f"Unsupported attachment type: {ext}")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Attached file is empty.")

        # Save to a temp file -- load_document / load_image_via_groq need a path
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            if ext in IMAGE_EXTENSIONS:
                docs = load_image_via_groq(tmp_path)
            else:
                docs = load_document(tmp_path)
            # query_with_attachment only uses the first ~4000 chars anyway, so
            # stop accumulating once we have enough instead of joining every
            # page of a large document first. A "Complete Course" OCR PDF can
            # be hundreds of pages -- extracting and joining all of it just to
            # truncate afterward wastes CPU/memory on Render's 512MB tier and
            # risks the request timing out or the process being OOM-killed.
            ATTACHMENT_CHAR_BUDGET = 6000
            parts = []
            total_len = 0
            for d in docs:
                parts.append(d.page_content)
                total_len += len(d.page_content)
                if total_len >= ATTACHMENT_CHAR_BUDGET:
                    break
            attachment_text = "\n\n".join(parts)
        finally:
            os.unlink(tmp_path)

        try:
            doc_ids = json.loads(document_ids) if document_ids else []
        except Exception:
            doc_ids = []

        history = get_session_history(session_id, user_id=str(user.id))
        result = query_with_attachment(
            question=question,
            attachment_text=attachment_text,
            attachment_name=file.filename,
            history=history,
            mode=mode,
            user_id=str(user.id),
            document_ids=doc_ids or None
        )

        # Store the question with a marker so history shows the attachment was used
        user_msg = f"[Attached: {file.filename}] {question}"
        save_session_message(session_id, role="user", content=user_msg, user_id=str(user.id))
        save_session_message(session_id, role="assistant", content=result["answer"], user_id=str(user.id))

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": result.get("chunks", []),
            "confidence": result.get("confidence", 0.0),
            "mode": mode,
            "intent": result.get("intent", "answer"),
            "related_concepts": result.get("related_concepts", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents(user=Depends(get_current_user)):
    return {"documents": get_all_documents(user_id=str(user.id))}

def _build_session_pdf(history: list, session_id: str) -> bytes:
    """Builds a human-readable PDF transcript of a chat session."""
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
        _new_api = True
    except ImportError:
        from fpdf import FPDF
        _new_api = False

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    def safe(text: str) -> str:
        return (text or "").encode("latin-1", "replace").decode("latin-1")

    def cell(w, h, txt, bold=False):
        """Wrapper that handles both old and new fpdf2 API."""
        if _new_api:
            pdf.cell(w, h, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.cell(w, h, txt, ln=True)

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(40, 40, 40)
    cell(0, 12, "MindVault Session Export")

    # Meta
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(120, 120, 120)
    cell(0, 6, safe(f"Session: {session_id[:16]}"))
    cell(0, 6, f"Messages: {len(history)}")
    pdf.ln(4)
    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    for msg in history:
        is_user = msg["role"] == "user"
        label = "You" if is_user else "MindVault"
        ts = msg.get("timestamp") or ""
        content_raw = msg.get("content") or ""

        # Role label
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(40, 90, 200 if is_user else 70)
        if not is_user:
            pdf.set_text_color(70, 140, 110)
        cell(0, 7, label)

        # Timestamp
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        cell(0, 5, safe(ts[:19] if ts else ""))

        # Body
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        # Strip markdown symbols that look ugly in plain PDF
        body = safe(content_raw.replace("**", "").replace("__", "").replace("# ", "").replace("## ", ""))
        pdf.multi_cell(0, 6, body)
        pdf.ln(4)

    return bytes(pdf.output())


@app.post("/export")
def export_session(req: ExportRequest, user=Depends(get_current_user)):
    try:
        history = get_session_history(req.session_id, user_id=str(user.id))

        if req.format == "pdf":
            if not history:
                history = [{"role": "assistant", "content": "No messages in this session.", "timestamp": ""}]
            try:
                pdf_bytes = _build_session_pdf(history, req.session_id)
            except Exception as pdf_err:
                import traceback
                print(f"[PDF Export] ERROR: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(pdf_err)}")
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="mindvault-{req.session_id[:8]}.pdf"'}
            )

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

@app.delete("/sessions/{session_id}/messages")
def clear_session_messages_route(session_id: str, user=Depends(get_current_user)):
    """Clears message history but keeps the session itself (used by 'Clear History')."""
    clear_session_messages(session_id, str(user.id))
    return {"ok": True}

@app.delete("/sessions/{session_id}")
def delete_session_route(session_id: str, user=Depends(get_current_user)):
    delete_chat_session(session_id, str(user.id))
    return {"ok": True}

@app.post("/sessions/{session_id}/share")
def share_session(session_id: str, user=Depends(get_current_user)):
    try:
        token = generate_share_token(session_id, str(user.id))
        base_url = os.environ.get("FRONTEND_URL", "").rstrip("/")
        if not base_url:
            # Fallback: derive from request or use default
            base_url = "https://mind-vault-psi.vercel.app"
        share_url = f"{base_url}/share/{token}"
        print(f"[Share] Generated share URL: {share_url}")
        return {"share_url": share_url, "token": token}
    except Exception as e:
        import traceback
        print(f"[Share] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Share failed: {str(e)}")

@app.delete("/sessions/{session_id}/share")
def unshare_session(session_id: str, user=Depends(get_current_user)):
    revoke_share_token(session_id, str(user.id))
    return {"ok": True}

@app.get("/share/{token}")
def get_shared_session_route(token: str):
    """Public endpoint — no auth required."""
    data = get_shared_session(token)
    if not data:
        raise HTTPException(status_code=404, detail="Shared session not found or access revoked.")
    return data

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