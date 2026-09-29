"""Streamlit UI for document Q&A (RAG)."""

import uuid

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from rag import config as C
from rag.chain import build_graph, source_label, stream_answer
from rag.chunking import split, validate
from rag.embeddings import build_embeddings, collection_name_for
from rag.llm import build_llm
from rag.loaders import LoaderError, load_file
from rag.vectorstore import (
    VectorStoreError,
    add_chunks,
    delete_document,
    ensure_db,
    get_retriever,
    get_store,
    list_documents,
)

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Preparing vector store…")
def _db():
    return ensure_db(C.CHROMA_DIR)


@st.cache_resource(show_spinner="Loading embedding model…")
def _embeddings(model: str, api_key: str | None):
    return build_embeddings(model, api_key)


def _init_state():
    defaults = {
        "configured": False,
        "cfg": {},
        "messages": [],  # UI display only: dicts of role, content, sources
        "checkpointer": InMemorySaver(),  # LangGraph conversation memory (per session)
        "thread_id": str(uuid.uuid4()),
        "uploader_key": 0,
        "db_notice_shown": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _reset_config():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ---------- setup (asked once per session) ----------
def setup_screen():
    st.title("📄 Document Q&A — Setup")
    st.caption("Configure once for this session. API keys are kept in memory only, never written to disk.")

    provider = st.selectbox("LLM provider", C.PROVIDERS)
    model_choice = st.selectbox("Model", C.SUGGESTED_MODELS[provider] + [C.CUSTOM_MODEL_LABEL])
    model = st.text_input("Custom model name") if model_choice == C.CUSTOM_MODEL_LABEL else model_choice
    llm_key = st.text_input(f"{provider} API key", type="password")

    st.divider()
    emb_option = st.radio("Embedding model", C.EMBEDDING_OPTIONS, horizontal=True)
    emb_model, emb_key = C.CHROMA_DEFAULT_EMBEDDING, None
    if emb_option == "OpenAI":
        emb_model = st.selectbox("OpenAI embedding model", C.OPENAI_EMBEDDING_MODELS)
        emb_key = llm_key if provider == "OpenAI" else st.text_input("OpenAI API key (embeddings)", type="password")
        if provider == "OpenAI":
            st.caption("Using the OpenAI key above for embeddings.")
    else:
        st.caption("Uses Chroma's built-in all-MiniLM-L6-v2 (runs locally; downloaded on first use).")

    if st.button("Save & continue", type="primary"):
        if not model or not llm_key:
            st.error("Model and API key are required.")
            return
        if emb_option == "OpenAI" and not emb_key:
            st.error("An OpenAI API key is required for OpenAI embeddings.")
            return
        with st.spinner("Validating credentials…"):
            try:
                build_llm(provider, model, llm_key).invoke("ping")
                if emb_option == "OpenAI":
                    _embeddings(emb_model, emb_key).embed_query("ping")
            except Exception as exc:
                st.error(f"Validation failed: {exc}")
                return
        st.session_state.cfg = {
            "provider": provider,
            "model": model,
            "llm_key": llm_key,
            "emb_model": emb_model,
            "emb_key": emb_key,
        }
        st.session_state.configured = True
        st.rerun()


# ---------- sidebar ----------
def sidebar(store) -> dict:
    cfg = st.session_state.cfg
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(
            f"**LLM:** {cfg['provider']} · `{cfg['model']}`  \n"
            f"**Embeddings:** `{cfg['emb_model']}`  \n"
            f"**Collection:** `{collection_name_for(cfg['emb_model'])}`"
        )
        if st.button("Reset configuration"):
            _reset_config()
            st.rerun()

        st.header("📤 Upload & chunking")
        strategy = st.radio(
            "Chunking strategy",
            C.CHUNKING_STRATEGIES,
            index=C.CHUNKING_STRATEGIES.index(C.DEFAULT_CHUNKING_STRATEGY),
            horizontal=True,
        )
        c1, c2 = st.columns(2)
        chunk_size = c1.number_input(
            "Chunk size (tokens)", C.MIN_CHUNK_SIZE, C.MAX_CHUNK_SIZE, C.DEFAULT_CHUNK_SIZE, step=50
        )
        chunk_overlap = c2.number_input(
            "Overlap (tokens)", 0, C.MAX_CHUNK_SIZE - 1, C.DEFAULT_CHUNK_OVERLAP, step=25
        )
        files = st.file_uploader(
            "Upload files",
            type=C.SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if files and st.button("Index files", type="primary"):
            _index_files(store, files, strategy, int(chunk_size), int(chunk_overlap))

        st.header("📚 Documents")
        docs = list_documents(store)
        selected = []
        if not docs:
            st.caption("No documents indexed yet.")
        for d in docs:
            col_chk, col_del = st.columns([5, 1])
            label = f"{d['file_name']}"
            help_ = (
                f"Uploaded {d['upload_time']} · {d['n_chunks']} chunks · "
                f"{d['strategy']} {d['chunk_size']}/{d['chunk_overlap']}"
            )
            if col_chk.checkbox(label, value=True, key=f"sel_{d['file_id']}", help=help_):
                selected.append(d["file_id"])
            if col_del.button("🗑", key=f"del_{d['file_id']}", help=f"Delete {d['file_name']}"):
                delete_document(store, d["file_id"])
                st.rerun()

        st.header("🔎 Retrieval & memory")
        top_k = st.slider("Top-k chunks", 1, C.MAX_TOP_K, C.DEFAULT_TOP_K)
        mem = st.slider("Memory (past messages)", C.MIN_MEMORY_WINDOW, C.MAX_MEMORY_WINDOW, C.DEFAULT_MEMORY_WINDOW)
        temperature = st.slider("Temperature", 0.0, 1.0, C.DEFAULT_TEMPERATURE, 0.1)
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())  # fresh LangGraph thread
            st.rerun()

    return {"selected": selected, "top_k": top_k, "memory": mem, "temperature": temperature}


def _index_files(store, files, strategy, chunk_size, chunk_overlap):
    try:
        validate(chunk_size, chunk_overlap)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return
    total = 0
    with st.sidebar.status("Indexing…", expanded=True) as status:
        for f in files:
            try:
                docs = load_file(f.name, f.getvalue())
                chunks = split(docs, strategy, chunk_size, chunk_overlap)
                n = add_chunks(store, chunks)
                total += n
                st.write(f"✅ {f.name}: {n} chunks")
            except LoaderError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.warning(f"Failed to index {f.name}: {exc}")
        status.update(label=f"Indexed {total} chunks", state="complete")
    st.session_state.uploader_key += 1  # clear uploader
    st.rerun()


# ---------- chat ----------
def _render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**{i}. {s['label']}** · chunk {s['chunk_index']}")
            st.caption(s["preview"])


def chat(store, opts: dict):
    cfg = st.session_state.cfg
    st.title("📄 Document Q&A")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            _render_sources(m.get("sources", []))

    disabled = not opts["selected"]
    prompt = st.chat_input(
        "Select at least one document" if disabled else "Ask a question about your documents…",
        disabled=disabled,
    )
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    llm = build_llm(cfg["provider"], cfg["model"], cfg["llm_key"], opts["temperature"])
    retriever = get_retriever(store, opts["top_k"], opts["selected"])

    graph = build_graph(llm, retriever, st.session_state.checkpointer)

    with st.chat_message("assistant"):
        result: dict = {}
        try:
            with st.spinner("Searching documents…"):
                tokens = stream_answer(graph, prompt, st.session_state.thread_id, opts["memory"], result)
                answer = st.write_stream(tokens)
        except Exception as exc:
            answer = f"⚠️ Error: {exc}"
            st.error(answer)
        docs = result.get("docs", [])
        sources = [
            {
                "label": source_label(d),
                "chunk_index": d.metadata.get("chunk_index"),
                "preview": d.page_content[:300] + ("…" if len(d.page_content) > 300 else ""),
            }
            for d in docs
        ]
        _render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# ---------- main ----------
def main():
    _init_state()
    try:
        db = _db()
    except VectorStoreError as exc:
        st.error(str(exc))
        st.stop()
    if db.created and not st.session_state.db_notice_shown:
        st.toast(f"Created new vector store at {db.path}", icon="🆕")
        st.session_state.db_notice_shown = True

    if not st.session_state.configured:
        setup_screen()
        return

    cfg = st.session_state.cfg
    try:
        store = get_store(
            db,
            _embeddings(cfg["emb_model"], cfg["emb_key"]),
            collection_name_for(cfg["emb_model"]),
            cfg["emb_model"],
        )
    except VectorStoreError as exc:
        st.error(str(exc))
        st.stop()

    opts = sidebar(store)
    chat(store, opts)


main()
