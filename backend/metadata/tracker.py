import json
import os
from datetime import datetime


DOCS_FILE = "data/documents.json"


def _load_docs() -> list:
    if not os.path.exists(DOCS_FILE):
        return []
    with open(DOCS_FILE, "r") as f:
        return json.load(f)


def _save_docs(docs: list):
    with open(DOCS_FILE, "w") as f:
        json.dump(docs, f, indent=2)


def log_document(filename: str, path: str, chunk_count: int):
    docs = _load_docs()
    
    for doc in docs:
        if doc["filename"] == filename:
            doc["chunk_count"] = chunk_count
            doc["last_updated"] = datetime.utcnow().isoformat()
            _save_docs(docs)
            return
    
    docs.append({
        "filename": filename,
        "path": path,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.utcnow().isoformat(),
    })
    
    _save_docs(docs)


def get_all_documents() -> list:
    return _load_docs()


def get_document(filename: str) -> dict | None:
    docs = _load_docs()
    for doc in docs:
        if doc["filename"] == filename:
            return doc
    return None