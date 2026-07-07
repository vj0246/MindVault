from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import os
import re
import uuid
from rag.db import get_supabase
from rag.ingest import ingest_document, load_document, load_image_via_groq, IMAGE_EXTENSIONS
from rag.retrieve1 import query_rag, query_with_attachment, stream_rag, stream_with_attachment
from rag.cache import invalidate_has_chunks
from rag.memory import (
    get_session_history, save_session_message, clear_session_messages,
    list_chat_sessions, create_chat_session, rename_chat_session, delete_chat_session,
    generate_share_token, get_shared_session, revoke_share_token,
    get_user_preferences, save_user_preferences,
    list_memory_notes, add_memory_note, delete_memory_note,
    search_messages
)
from metadata.tracker import log_document, get_all_documents, set_document_folder, delete_document
from graph.extractor import extract_entities_and_relations
from graph.store import add_to_graph, get_related_nodes, get_full_graph
from security.rate_limit import enforce_rate_limit
from security.guardrails import moderate_input
from dotenv import load_dotenv

# Per-user/hour caps. IP cap is 3x this (see security/rate_limit.py) as a
# secondary backstop against one IP running many accounts.
QUERY_RATE_LIMIT = 100
UPLOAD_RATE_LIMIT = 20
SESSION_RATE_LIMIT = 30
EXPORT_RATE_LIMIT = 30
MEMORY_RATE_LIMIT = 60
SHARE_RATE_LIMIT = 30
SEARCH_RATE_LIMIT = 30
FOLDER_MAX_CHARS = 60

# Client-controlled input caps -- everything here ends up either in an LLM
# prompt (cost/latency abuse) or a DB row (storage abuse) if left unbounded.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB
QUESTION_MAX_CHARS = 8000
MEMORY_NOTE_MAX_CHARS = 500
PREFERENCE_NAME_MAX_CHARS = 100
PREFERENCE_TONE_MAX_CHARS = 100
PREFERENCE_SYSTEM_PROMPT_MAX_CHARS = 1000
PREFERENCE_MAX_PRIORITIES = 10

load_dotenv()

# Known frontend origins, plus ALLOWED_ORIGINS (comma-separated) for extra
# deploy previews/custom domains without a code change. Auth is Bearer-token
# (not cookies) so a wildcard wasn't a credential-theft vector, but pinning
# to known origins is still tighter than accepting any site.
_DEFAULT_ORIGINS = [
    "https://mind-vault-psi.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]
frontend_url = os.environ.get("FRONTEND_URL", "").rstrip("/")
ALLOWED_ORIGINS = list(dict.fromkeys(_DEFAULT_ORIGINS + _extra_origins + ([frontend_url] if frontend_url else [])))

app = FastAPI(title="MindVault API")
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


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
    """Runs in the background after ingestion finishes. Extracts entities and
    relationships from up to 10 chunks and adds them to the knowledge graph."""
    for chunk in chunks:
        try:
            extracted = extract_entities_and_relations(chunk.page_content, source=filename)
            add_to_graph(extracted, user_id=user_id)
        except Exception as e:
            print(f"[Graph] Background extraction failed for a chunk: {e}")

def _ingest_and_finalize(file_path: str, document_id: str, filename: str, user_id: str):
    """Runs entirely in the background, after /upload has already responded.
    Parsing + chunking + embedding is CPU-bound and was previously done
    inline inside the request -- on Render's free tier that could exceed
    the platform's own request timeout on a slow/heavy document. When that
    happened the connection was killed before Python's exception handling
    ever ran, leaving the chunk_count=-1 row from log_document() as a
    silent zombie with no chunks and no error surfaced anywhere. Moving
    the work here means the HTTP response is never at risk of that timeout,
    and every outcome (success or any exception) is handled explicitly:
    the row either gets a real chunk_count or gets deleted -- never left
    stuck mid-way."""
    sb = get_supabase()
    try:
        chunks = ingest_document(
            file_path, document_id=document_id, user_id=user_id,
            langsmith_extra={"metadata": {"user_id": user_id, "filename": filename, "document_id": document_id}}
        )
        sb.table("documents").update({"chunk_count": len(chunks)}).eq("id", document_id).execute()

        # A successful upload just changed the answer to "does this user have
        # any documents" -- clear the short-TTL cache immediately so the
        # very next query never sees a stale "no documents" result.
        invalidate_has_chunks(user_id)

        _build_graph_for_chunks(chunks[:10], filename, user_id)
    except Exception as e:
        # Logging must never be able to block the cleanup delete below --
        # e.g. a non-ASCII character in a caught error message raising
        # UnicodeEncodeError on a non-UTF-8 stdout would otherwise skip
        # straight past the delete and leave the exact zombie row this
        # function exists to prevent.
        try:
            import traceback
            print(f"[Ingest] Background ingestion failed for '{filename}' (document_id={document_id}): {traceback.format_exc()}")
        except Exception:
            print(f"[Ingest] Background ingestion failed for document_id={document_id} (error message could not be printed)")
        try:
            sb.table("documents").delete().eq("id", document_id).execute()
        except Exception as cleanup_err:
            print(f"[Ingest] Cleanup delete also failed for document_id={document_id}: {cleanup_err}")

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), background_tasks: BackgroundTasks = None, user=Depends(get_current_user)):
    try:
        enforce_rate_limit("upload", str(user.id), request.client.host, UPLOAD_RATE_LIMIT)
        ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        # Client-supplied filename -- take the basename only (drops any
        # ../ or absolute-path components) and strip to safe characters
        # before it ever touches a filesystem path.
        safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(file.filename))
        if not safe_stem or safe_stem in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid filename.")

        # Content-Length is client-supplied and spoofable -- it's a cheap
        # early rejection, not the real guard. The real guard is the capped
        # read below, so a lying/missing header can never bypass the limit.
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 25MB).")

        os.makedirs("data/docs", exist_ok=True)
        file_path = f"data/docs/{uuid.uuid4()}_{safe_stem}"
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 25MB).")
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        document_id = str(uuid.uuid4())
        # chunk_count=-1 is the "still processing" sentinel -- _ingest_and_finalize
        # (below) replaces it with the real count on success, or deletes the
        # row entirely on failure. It is never left at this value.
        actual_document_id = log_document(
            filename=file.filename,
            path=file_path,
            chunk_count=-1,
            document_id=document_id,
            user_id=str(user.id)
        )

        if background_tasks is not None:
            background_tasks.add_task(
                _ingest_and_finalize, file_path, actual_document_id, file.filename, str(user.id)
            )
        else:
            # BackgroundTasks is always provided by FastAPI in practice --
            # this is just a safe fallback, not the expected path.
            _ingest_and_finalize(file_path, actual_document_id, file.filename, str(user.id))

        return {
            "status": "processing",
            "document_id": actual_document_id,
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

class RenameRequest(BaseModel):
    name: str = Field(..., max_length=60)

class PreferencesRequest(BaseModel):
    name: str = Field("", max_length=PREFERENCE_NAME_MAX_CHARS)
    tone: str = Field("", max_length=PREFERENCE_TONE_MAX_CHARS)
    priorities: list[str] = Field(default_factory=list, max_length=PREFERENCE_MAX_PRIORITIES)
    system_prompt: str = Field("", max_length=PREFERENCE_SYSTEM_PROMPT_MAX_CHARS)
    theme: str = "Light"

class MemoryNoteRequest(BaseModel):
    content: str = Field(..., max_length=MEMORY_NOTE_MAX_CHARS)

class QueryRequest(BaseModel):
    question: str = Field(..., max_length=QUESTION_MAX_CHARS)
    mode: str = "default"
    session_id: str = "default_session"
    document_ids: list = []

class ExportRequest(BaseModel):
    session_id: str
    format: str = "markdown"

@app.post("/query")
def query(req: QueryRequest, request: Request, user=Depends(get_current_user)):
    try:
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        enforce_rate_limit("query", str(user.id), request.client.host, QUERY_RATE_LIMIT)

        mod = moderate_input(req.question)
        if mod["flagged"]:
            refusal = f"I can't help with that request. {mod['reason']}".strip()
            save_session_message(req.session_id, role="user", content=req.question, user_id=str(user.id))
            save_session_message(req.session_id, role="assistant", content=refusal, user_id=str(user.id))
            return {
                "answer": refusal, "sources": [], "chunks": [], "confidence": 0.0,
                "mode": req.mode, "intent": "blocked", "related_concepts": []
            }

        history = get_session_history(req.session_id, user_id=str(user.id))
        result = query_rag(
            question=req.question,
            history=history,
            mode=req.mode,
            user_id=str(user.id),
            document_ids=req.document_ids or None,
            # @traceable functions accept a reserved langsmith_extra kwarg --
            # stripped before the call, used to attach metadata/tags to the
            # trace. Without this, every trace in LangSmith looks identical;
            # with it, you can filter/search by user_id, session_id, or mode
            # in the dashboard instead of opening traces one by one.
            langsmith_extra={"metadata": {"user_id": str(user.id), "session_id": req.session_id, "mode": req.mode}}
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
            "related_concepts": result.get("related_concepts", []),
            "answer_type": result.get("answer_type", "grounded"),
            "tokens": result.get("tokens", {"message": None, "daily_used": 0, "daily_pct": 0})
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request, user=Depends(get_current_user)):
    """Streaming version of /query. Returns SSE (text/event-stream).
    Events:
      {"type":"meta",  "sources":[...], "intent":"...", "confidence":0.8, "chunks":[...]}
      {"type":"token", "text":"hello "}
      {"type":"done",  "related_concepts":[...], "full_answer":"..."}
    Frontend should accumulate tokens, then save full_answer from done event.
    """
    enforce_rate_limit("query", str(user.id), request.client.host, QUERY_RATE_LIMIT)
    history = get_session_history(req.session_id, user_id=str(user.id))

    async def event_generator():
        import json as _json
        full_answer = ""

        mod = moderate_input(req.question)
        if mod["flagged"]:
            refusal = f"I can't help with that request. {mod['reason']}".strip()
            yield f'data: {_json.dumps({"type": "meta", "sources": [], "intent": "blocked", "confidence": 0.0, "chunks": []})}\n\n'
            yield f'data: {_json.dumps({"type": "token", "text": refusal})}\n\n'
            yield f'data: {_json.dumps({"type": "done", "related_concepts": [], "full_answer": refusal})}\n\n'
            save_session_message(req.session_id, role="user", content=req.question, user_id=str(user.id))
            save_session_message(req.session_id, role="assistant", content=refusal, user_id=str(user.id))
            return

        try:
            for event in stream_rag(
                question=req.question,
                history=history,
                mode=req.mode,
                user_id=str(user.id),
                document_ids=req.document_ids or None,
                langsmith_extra={"metadata": {"user_id": str(user.id), "session_id": req.session_id, "mode": req.mode}}
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
    request: Request,
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

        enforce_rate_limit("query", str(user.id), request.client.host, QUERY_RATE_LIMIT)

        # Check the question text before doing any file parsing -- cheap
        # fail-fast on a flagged request avoids burning CPU on the attachment.
        mod = moderate_input(question)
        if mod["flagged"]:
            refusal = f"I can't help with that request. {mod['reason']}".strip()
            user_msg = f"[Attached: {file.filename}] {question}"
            save_session_message(session_id, role="user", content=user_msg, user_id=str(user.id))
            save_session_message(session_id, role="assistant", content=refusal, user_id=str(user.id))
            return {
                "answer": refusal, "sources": [], "chunks": [], "confidence": 0.0,
                "mode": mode, "intent": "blocked", "related_concepts": []
            }

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
            document_ids=doc_ids or None,
            langsmith_extra={"metadata": {"user_id": str(user.id), "session_id": session_id, "mode": mode, "has_attachment": True}}
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

@app.post("/query-with-attachment/stream")
async def query_with_attachment_stream_route(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    session_id: str = Form("default_session"),
    mode: str = Form("default"),
    document_ids: str = Form("[]"),
    user=Depends(get_current_user)
):
    """Streaming (SSE) counterpart to /query-with-attachment -- same
    validation/extraction, progressive token rendering instead of one
    blocking response like /query/stream is to /query."""
    import json
    import tempfile

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    enforce_rate_limit("query", str(user.id), request.client.host, QUERY_RATE_LIMIT)

    ext = os.path.splitext(file.filename)[1].lower()
    ALLOWED = {".pdf", ".txt", ".md", ".docx", ".doc", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported attachment type: {ext}")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Attached file is empty.")

    try:
        doc_ids = json.loads(document_ids) if document_ids else []
    except Exception:
        doc_ids = []

    history = get_session_history(session_id, user_id=str(user.id))

    async def event_generator():
        full_answer = ""
        mod = moderate_input(question)
        if mod["flagged"]:
            refusal = f"I can't help with that request. {mod['reason']}".strip()
            user_msg = f"[Attached: {file.filename}] {question}"
            yield f'data: {json.dumps({"type": "meta", "sources": [], "intent": "blocked", "confidence": 0.0, "chunks": []})}\n\n'
            yield f'data: {json.dumps({"type": "token", "text": refusal})}\n\n'
            yield f'data: {json.dumps({"type": "done", "related_concepts": [], "full_answer": refusal})}\n\n'
            save_session_message(session_id, role="user", content=user_msg, user_id=str(user.id))
            save_session_message(session_id, role="assistant", content=refusal, user_id=str(user.id))
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            if ext in IMAGE_EXTENSIONS:
                docs = load_image_via_groq(tmp_path)
            else:
                docs = load_document(tmp_path)
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
            for event in stream_with_attachment(
                question=question,
                attachment_text=attachment_text,
                attachment_name=file.filename,
                history=history,
                mode=mode,
                user_id=str(user.id),
                document_ids=doc_ids or None,
                langsmith_extra={"metadata": {"user_id": str(user.id), "session_id": session_id, "mode": mode, "has_attachment": True}}
            ):
                if '"type": "done"' in event:
                    data = json.loads(event.replace("data: ", "").strip())
                    full_answer = data.get("full_answer", "")
                yield event
        except Exception as e:
            yield f'data: {json.dumps({"type": "error", "message": str(e)})}\n\n'

        if full_answer:
            user_msg = f"[Attached: {file.filename}] {question}"
            save_session_message(session_id, role="user", content=user_msg, user_id=str(user.id))
            save_session_message(session_id, role="assistant", content=full_answer, user_id=str(user.id))

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/documents")
def list_documents(user=Depends(get_current_user)):
    return {"documents": get_all_documents(user_id=str(user.id))}

class FolderRequest(BaseModel):
    folder: str | None = Field(None, max_length=FOLDER_MAX_CHARS)

@app.patch("/documents/{document_id}/folder")
def set_document_folder_route(document_id: str, body: FolderRequest, user=Depends(get_current_user)):
    folder = body.folder.strip() if body.folder and body.folder.strip() else None
    set_document_folder(document_id, str(user.id), folder)
    return {"ok": True, "folder": folder}

@app.delete("/documents/{document_id}")
def delete_document_route(document_id: str, user=Depends(get_current_user)):
    found = delete_document(document_id, str(user.id))
    if not found:
        raise HTTPException(status_code=404, detail="Document not found")
    invalidate_has_chunks(str(user.id))
    return {"ok": True}

def _build_session_pdf(history: list, session_label: str) -> bytes:
    """Builds a human-readable PDF transcript of a chat session.
    session_label is a clean display string like 'Session #4' -- never the
    raw UUID, which means nothing to the user."""
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
    cell(0, 6, safe(session_label))
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
def export_session(req: ExportRequest, request: Request, user=Depends(get_current_user)):
    try:
        enforce_rate_limit("export", str(user.id), request.client.host, EXPORT_RATE_LIMIT)
        history = get_session_history(req.session_id, user_id=str(user.id))

        # Look up the session's clean sequential number + name for display.
        # Falls back to a generic label if the session row can't be found
        # (e.g. it was deleted) -- exports should never show a raw UUID.
        all_sessions = list_chat_sessions(str(user.id))
        matching = next((s for s in all_sessions if s["id"] == req.session_id), None)
        session_label = f"Session #{matching['number']}" if matching else "Session"
        session_name = matching["name"] if matching else "Untitled"

        if req.format == "pdf":
            if not history:
                history = [{"role": "assistant", "content": "No messages in this session.", "timestamp": ""}]
            try:
                pdf_bytes = _build_session_pdf(history, session_label)
            except Exception as pdf_err:
                import traceback
                print(f"[PDF Export] ERROR: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(pdf_err)}")
            fname_num = matching["number"] if matching else "x"
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="mindvault-session-{fname_num}.pdf"'}
            )

        if not history:
            return {"report": "# MindVault Session Export\n\n_No messages in this session._"}

        lines = ["# MindVault Session Export\n"]
        lines.append(f"**{session_label}** \u2014 {session_name}\n")
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Session management ────────────────────────────────────────────────────

@app.get("/sessions")
def list_sessions_route(user=Depends(get_current_user)):
    return {"sessions": list_chat_sessions(str(user.id))}

@app.get("/sessions/search")
def search_messages_route(q: str, request: Request, user=Depends(get_current_user)):
    if not q.strip():
        return {"results": []}
    enforce_rate_limit("search", str(user.id), request.client.host, SEARCH_RATE_LIMIT)
    return {"results": search_messages(str(user.id), q.strip()[:200])}

@app.post("/sessions")
def create_session_route(request: Request, user=Depends(get_current_user)):
    enforce_rate_limit("session_create", str(user.id), request.client.host, SESSION_RATE_LIMIT)
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

@app.get("/preferences")
def get_preferences_route(user=Depends(get_current_user)):
    return {"preferences": get_user_preferences(str(user.id))}

@app.post("/preferences")
def save_preferences_route(body: PreferencesRequest, request: Request, user=Depends(get_current_user)):
    enforce_rate_limit("preferences_save", str(user.id), request.client.host, MEMORY_RATE_LIMIT)
    # Custom system_prompt is spliced into the LLM's system role on every
    # future query (see rag/retrieve1.py:_preference_hint) -- it must clear
    # the same moderation bar as a live question, or a user could plant a
    # one-time jailbreak that silently applies to every answer afterward.
    if body.system_prompt.strip():
        mod = moderate_input(body.system_prompt)
        if mod["flagged"]:
            raise HTTPException(status_code=400, detail=f"Custom instructions rejected: {mod['reason']}")
    return save_user_preferences(str(user.id), body.name, body.tone, body.priorities, body.system_prompt, body.theme)

@app.get("/memory")
def list_memory_route(user=Depends(get_current_user)):
    return {"notes": list_memory_notes(str(user.id))}

@app.post("/memory")
def add_memory_route(body: MemoryNoteRequest, request: Request, user=Depends(get_current_user)):
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Note cannot be empty.")
    enforce_rate_limit("memory_add", str(user.id), request.client.host, MEMORY_RATE_LIMIT)
    return add_memory_note(str(user.id), body.content.strip())

@app.delete("/memory/{note_id}")
def delete_memory_route(note_id: str, user=Depends(get_current_user)):
    delete_memory_note(note_id, str(user.id))
    return {"ok": True}

@app.post("/sessions/{session_id}/share")
def share_session(session_id: str, request: Request, user=Depends(get_current_user)):
    try:
        enforce_rate_limit("share_create", str(user.id), request.client.host, SHARE_RATE_LIMIT)
        token = generate_share_token(session_id, str(user.id))
        base_url = os.environ.get("FRONTEND_URL", "").rstrip("/")
        if not base_url:
            # Fallback: derive from request or use default
            base_url = "https://mind-vault-psi.vercel.app"
        share_url = f"{base_url}/share/{token}"
        print(f"[Share] Generated share URL: {share_url}")
        return {"share_url": share_url, "token": token}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Share] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Share failed: {str(e)}")

@app.delete("/sessions/{session_id}/share")
def unshare_session(session_id: str, user=Depends(get_current_user)):
    revoke_share_token(session_id, str(user.id))
    return {"ok": True}

@app.get("/share/{token}")
def get_shared_session_route(token: str, request: Request):
    """Public endpoint — no auth required. Rate-limited per-IP since there's
    no user_id to key on (reuses the same helper as authed routes by passing
    the IP as both identifiers -- both checks collapse to one per-IP limit)."""
    enforce_rate_limit("share_view", request.client.host, request.client.host, SHARE_RATE_LIMIT)
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
    origin = request.headers.get("origin", "")
    headers = {
        "Access-Control-Allow-Methods": "POST, GET, DELETE, PATCH, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    }
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
    return JSONResponse(content={}, headers=headers)