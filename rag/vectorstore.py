"""Persistent Chroma vector store: bootstrap, add, list, delete, retrieve."""

from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreError(Exception):
    pass


@dataclass
class DBHandle:
    client: chromadb.ClientAPI
    path: Path
    created: bool  # True if the store didn't exist before this run


def ensure_db(path: Path) -> DBHandle:
    """Create the Chroma directory/SQLite store if missing (e.g. fresh clone) and open it."""
    path = Path(path)
    created = not (path / "chroma.sqlite3").exists()
    try:
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
    except (OSError, PermissionError) as exc:
        raise VectorStoreError(f"Cannot create or open vector store at {path}: {exc}") from exc
    return DBHandle(client=client, path=path, created=created)


def get_store(db: DBHandle, embeddings: Embeddings, collection_name: str, embedding_model: str) -> Chroma:
    collection = db.client.get_or_create_collection(
        collection_name, metadata={"embedding_model": embedding_model}
    )
    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model and existing_model != embedding_model:
        raise VectorStoreError(
            f"Collection '{collection_name}' was built with '{existing_model}', "
            f"not '{embedding_model}'. Refusing to mix embeddings."
        )
    return Chroma(client=db.client, collection_name=collection_name, embedding_function=embeddings)


def add_chunks(store: Chroma, chunks: list[Document]) -> int:
    if not chunks:
        return 0
    file_ids = {c.metadata["file_id"] for c in chunks}
    for file_id in file_ids:  # re-index cleanly (e.g. new chunk settings)
        delete_document(store, file_id)
    ids = [f"{c.metadata['file_id']}:{c.metadata['chunk_index']}" for c in chunks]
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def list_documents(store: Chroma) -> list[dict]:
    """Unique indexed files with chunk counts, newest first."""
    metadatas = store.get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for m in metadatas:
        entry = docs.setdefault(
            m["file_id"],
            {
                "file_id": m["file_id"],
                "file_name": m.get("file_name", "?"),
                "upload_time": m.get("upload_time", ""),
                "strategy": m.get("chunking_strategy", ""),
                "chunk_size": m.get("chunk_size"),
                "chunk_overlap": m.get("chunk_overlap"),
                "n_chunks": 0,
            },
        )
        entry["n_chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["upload_time"], reverse=True)


def delete_document(store: Chroma, file_id: str) -> None:
    store.delete(where={"file_id": file_id})


def get_retriever(store: Chroma, k: int, file_ids: list[str]):
    filter_ = {"file_id": file_ids[0]} if len(file_ids) == 1 else {"file_id": {"$in": file_ids}}
    return store.as_retriever(search_kwargs={"k": k, "filter": filter_})
