## What is actually built and working

Week 1:
- ingest.py: PDF → chunks → embeddings → FAISS (local, zero cost)
- retrieve_manual.py: hand-built RAG without LangChain chains
- retrieve_lcel.py: proper LangChain LCEL with RouterChain
- app.py: FastAPI with upload, query, documents routes

Week 2:
- memory.py: persistent session history with summarization for long sessions
- tracker.py: document metadata registry

## Key decisions made
- Fully local stack: Ollama + llama3.2:3b + nomic-embed-text
- Built RAG manually first, then rebuilt with LCEL (intentional learning)
- Router chain decides between history and retrieval per query
- chunk_size=500 optimized for typed student notes
