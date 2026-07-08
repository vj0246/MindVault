import os
import re
import base64
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from rag.embedder import EMBED_MODEL
from rag.db import get_supabase
from security.groq_keys import call_with_key_fallback, get_client
from dotenv import load_dotenv

try:
    from langsmith import traceable
except ImportError:
    # Tracing is observability, not core functionality -- same fallback
    # pattern as retrieve1.py, so ingestion never breaks if langsmith is
    # ever unavailable.
    def traceable(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator if not (args and callable(args[0])) else args[0]

load_dotenv()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".webp": "image/webp", ".tiff": "image/tiff"
}

# --- Section heading detection ---
# Matches: markdown headers, ALL-CAPS titles, numbered sections (1. / 1.1 / 1.1.1)
HEADING_RE = re.compile(
    r'^(#{1,4}\s+\S.{0,80}|[A-Z][A-Z\s\d\-:]{4,60}$|(?:\d+\.)+\d*\s+[A-Z].{0,60})$',
    re.MULTILINE
)

# Fenced code blocks
CODE_RE = re.compile(r'```[\s\S]*?```', re.MULTILINE)





# ─────────────────────────────────────────────
# Document Loaders
# ─────────────────────────────────────────────

def load_document(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    elif ext in (".docx", ".doc"):
        return Docx2txtLoader(file_path).load()
    elif ext in IMAGE_EXTENSIONS:
        return load_image_via_groq(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


@traceable(name="ingest_image_ocr", run_type="llm")
def load_image_via_groq(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = IMAGE_MIME.get(ext, "image/jpeg")
    with open(file_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    def _call(key):
        client = get_client(key)
        return client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                {"type": "text", "text": (
                    "Extract all content from this image in full detail. "
                    "If it contains text or notes, transcribe exactly. "
                    "If it contains diagrams, charts, or tables, describe them thoroughly."
                )}
            ]}],
            max_tokens=2000
        )
    response = call_with_key_fallback(_call)
    extracted = response.choices[0].message.content
    print(f"[Ingest] Image extracted via Groq: {len(extracted)} chars")
    return [Document(page_content=extracted, metadata={"source": os.path.basename(file_path), "type": "image"})]


# ─────────────────────────────────────────────
# Smart Chunking
# ─────────────────────────────────────────────

def _cosine_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]


def _semantic_split(text: str, header: str = "", max_chars=600, min_chars=150, threshold=0.65) -> list:
    """
    Splits text into semantically coherent chunks.
    Embeds sentences, splits where similarity drops below threshold or chunk exceeds max_chars.
    Prepends section header to each chunk so retrieval always has context.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(text) <= max_chars or len(sentences) <= 2:
        chunk = f"{header}\n{text}".strip() if header else text.strip()
        return [chunk]

    # Batch-embed all sentences at once — efficient single call to fastembed
    embeddings = list(EMBED_MODEL.embed(sentences))

    chunks, current, cur_len = [], [sentences[0]], len(sentences[0])

    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        new_len = cur_len + len(sentences[i]) + 1

        # Split when semantic break detected + enough content, or hard size limit hit
        if (sim < threshold and cur_len >= min_chars) or new_len > max_chars:
            chunk_text = " ".join(current)
            chunks.append(f"{header}\n{chunk_text}".strip() if header else chunk_text)
            current, cur_len = [sentences[i]], len(sentences[i])
        else:
            current.append(sentences[i])
            cur_len = new_len

    if current:
        chunk_text = " ".join(current)
        chunks.append(f"{header}\n{chunk_text}".strip() if header else chunk_text)

    return [c for c in chunks if len(c.strip()) > 30]


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return len(s) >= 2 and s.startswith('|') and s.endswith('|')


def _is_table_separator(line: str) -> bool:
    s = line.strip()
    return bool(s) and _is_table_row(s) and set(s) <= set('|-: \t')


def _find_table_blocks(text: str) -> list:
    """Line-based table detection -- deliberately not a regex. The
    previous TABLE_RE (`(?:\\|.+\\|\\s*\\n)+`) nests an unbounded
    quantifier inside a repeated group over pipe-delimited lines --
    classic catastrophic-backtracking shape. A document with lots of
    '|' characters (shell pipes, ASCII diagrams -- exactly what OS/CS
    notes tend to have) could make it run for a very long time, and
    since CPython's re engine doesn't release the GIL mid-match, it
    can stall the whole process, not just the one request. Line
    scanning is O(n) and can't blow up regardless of content."""
    lines = text.split('\n')
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        if _is_table_row(lines[i]) and i + 1 < n and _is_table_separator(lines[i + 1]):
            start = i
            i += 2
            while i < n and _is_table_row(lines[i]):
                i += 1
            blocks.append('\n'.join(lines[start:i]))
        else:
            i += 1
    return blocks


def _protect_blocks(text: str):
    """
    Replaces tables and code blocks with placeholders.
    Returns (modified_text, {placeholder: original_text}).
    Protected blocks are stored as-is (not split).
    """
    store = {}

    def _sub(m, prefix):
        key = f"__{prefix}_{len(store)}__"
        store[key] = m.group(0).strip()
        return f"\n{key}\n"

    for block in _find_table_blocks(text):
        key = f"__TABLE_{len(store)}__"
        store[key] = block.strip()
        text = text.replace(block, f"\n{key}\n", 1)

    text = CODE_RE.sub(lambda m: _sub(m, "CODE"), text)
    return text, store


def _parse_sections(text: str) -> list:
    """
    Splits text into [(heading, body)] pairs.
    Sections with no heading get an empty string as heading.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections = []
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append(("", pre))

    for i, m in enumerate(matches):
        heading = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))

    return sections


def _process_block(text: str, header: str, store: dict) -> list:
    """
    Processes one section body:
    - Protected placeholders → restored, kept intact as single chunk
    - Short text (≤200 chars) → single chunk
    - Long text → semantic split
    """
    chunks = []
    parts = re.split(r'(__(?:TABLE|CODE)_\d+__)', text)

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in store:
            # Table or code block — restore, never split
            content = store[part]
            chunks.append(f"{header}\n{content}".strip() if header else content)
        elif len(part) <= 200:
            chunks.append(f"{header}\n{part}".strip() if header else part)
        else:
            chunks.extend(_semantic_split(part, header=header))

    return [c for c in chunks if len(c.strip()) > 30]


@traceable(name="ingest_chunk_documents", run_type="chain")
def chunk_documents(pages, filename="") -> list:
    """
    Smart chunking pipeline:
    1. Combine all loaded pages into one text
    2. Protect tables + code blocks (kept intact)
    3. Split by section headings
    4. Semantically split each section body
    5. Prepend section header to every chunk
    Returns list of LangChain Document objects.
    """
    full_text = "\n\n".join([p.page_content for p in pages])
    print(f"[Ingest] Smart chunking: {len(full_text)} chars, file={filename}")

    text_clean, protected = _protect_blocks(full_text)
    sections = _parse_sections(text_clean)

    raw_chunks = []
    for heading, body in sections:
        if not body.strip():
            # Section heading with no body — skip or attach to next (skip for now)
            continue
        raw_chunks.extend(_process_block(body, heading, protected))

    # Fallback: if zero chunks produced, use simple fixed splitter
    if not raw_chunks:
        print("[Ingest] Smart chunker produced 0 chunks — falling back to fixed split")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(pages)

    docs = [
        Document(page_content=chunk, metadata={"source": filename, "chunk_index": i})
        for i, chunk in enumerate(raw_chunks)
    ]
    print(f"[Ingest] Smart chunker -> {len(docs)} chunks")
    return docs


# ─────────────────────────────────────────────
# Embed + Store
# ─────────────────────────────────────────────

@traceable(name="ingest_embed_and_store", run_type="chain")
def embed_and_store(chunks, document_id: str, filename: str, user_id: str):
    if not chunks:
        # fastembed.embed([]) on an empty list can misbehave -- guard explicitly.
        # Caller (ingest_document) already raises before reaching here, but
        # this keeps the function safe if called directly.
        return

    supabase = get_supabase()
    texts = [c.page_content for c in chunks]
    embeddings = list(EMBED_MODEL.embed(texts))

    rows = [
        {
            "document_id": document_id,
            "content": chunk.page_content,
            "embedding": emb.tolist(),
            "chunk_index": i,
            "page_number": chunk.metadata.get("page", 0),  # PyPDFLoader sets "page"; others default 0
            "filename": filename,
            "user_id": user_id
        }
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    batch_size = 50
    for i in range(0, len(rows), batch_size):
        supabase.table("chunks").insert(rows[i:i + batch_size]).execute()

    print(f"[Ingest] Stored {len(rows)} chunks")


@traceable(name="ingest_document", run_type="chain")
def ingest_document(file_path: str, document_id: str, user_id: str, **kwargs):
    print(f"[Ingest] Loading: {file_path}")
    pages = load_document(file_path)
    print(f"[Ingest] Loaded {len(pages)} pages/sections")

    # Detect empty/unreadable documents early (e.g. scanned PDFs with no
    # extractable text -- PyPDFLoader returns pages with empty page_content).
    total_chars = sum(len(p.page_content.strip()) for p in pages)
    if total_chars == 0:
        raise ValueError(
            "No extractable text found in this document. "
            "If this is a scanned PDF, try uploading a text-based PDF or "
            "an image of the page instead."
        )

    filename = os.path.basename(file_path)
    chunks = chunk_documents(pages, filename=filename)

    if not chunks:
        raise ValueError(
            "Document text was too short or could not be split into chunks. "
            "Try uploading a document with more content."
        )

    embed_and_store(chunks, document_id, filename, user_id)
    print(f"[Ingest] Done.")
    return chunks