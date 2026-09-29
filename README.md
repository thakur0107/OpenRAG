# Document Q&A (RAG)

Upload TXT, PDF or DOCX files and ask questions about them. Answers are grounded in the documents and cite their sources.
Built with Python, LangChain + LangGraph, Streamlit and Chroma.

## Quick start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app creates the vector store in `./chroma_db` if it doesn't already exist. To use a different location, set `CHROMA_DIR`:
```bash
CHROMA_DIR=/data/rag_db streamlit run app.py
```

## Features
- **One-time setup per session:**
  - LLM provider: Anthropic or OpenAI.
  - Model and API key. Keys are kept in memory only.
  - Embedding model: OpenAI (`text-embedding-3-small` or `-large`), or Chroma's default local model (all-MiniLM-L6-v2, needs no key; about 80 MB, downloaded to `~/.cache/chroma` on first use).
- **Chunking:** Recursive (the default) or Fixed size. Sizes are measured in tokens; defaults are 600 per chunk with 200 overlap, and you can change both.
- **Metadata on every chunk:** file name, file id (SHA-256), file type, upload time, page (for PDFs), chunk index, and the chunking settings used.
- **Persistent storage:** one Chroma collection per embedding model, so vectors from different models never mix. Documents can be deleted, and re-uploading a file replaces its old chunks.
- **Retrieval:** searches only the documents you've selected; top-k defaults to 6 and can be changed.
- **Grounded answers:** each claim is cited as `[file, p. N]`, and a "Sources" panel shows the retrieved chunks. If the answer isn't in the documents, the app says so.
- **Memory:** two layers. Streamlit `session_state` holds the chat shown in the UI, including the sources for each answer. A LangGraph `InMemorySaver` checkpointer holds the conversation the model sees, under one `thread_id` per session; the graph uses the last N messages (default 6) to rewrite follow-up questions. "Clear chat" starts a new thread.

## Layout
| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI (setup screen, sidebar, chat) |
| `rag/config.py` | Defaults and constants |
| `rag/loaders.py` | Turns files into Documents with metadata |
| `rag/chunking.py` | Recursive and fixed-size token splitters |
| `rag/embeddings.py` | OpenAI and Chroma-default embeddings |
| `rag/vectorstore.py` | Chroma setup, add, list, delete and retrieval |
| `rag/llm.py` | Builds the ChatAnthropic or ChatOpenAI model |
| `rag/chain.py` | LangGraph pipeline: condense → retrieve → generate / not_found |
| `rag/memory.py` | Sliding-window chat history |

## Tests
```bash
pytest -q
```
