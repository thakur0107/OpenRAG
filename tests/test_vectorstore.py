from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.vectorstore import add_chunks, delete_document, ensure_db, get_retriever, get_store, list_documents


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(t)), float(t.count("a")), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _chunk(file_id, name, i, text):
    return Document(
        page_content=text,
        metadata={"file_id": file_id, "file_name": name, "chunk_index": i, "page": -1,
                  "upload_time": "2026-01-01T00:00:00+00:00", "chunking_strategy": "Recursive",
                  "chunk_size": 600, "chunk_overlap": 200},
    )


def test_ensure_db_creates_missing_directory(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    assert not target.exists()
    db = ensure_db(target)
    assert db.created and target.is_dir()
    assert ensure_db(target).created is False  # second open finds existing store


def test_add_list_filter_delete(tmp_path):
    db = ensure_db(tmp_path / "db")
    store = get_store(db, FakeEmbeddings(), "docs__test", "fake")
    add_chunks(store, [_chunk("f1", "a.txt", 0, "aaa"), _chunk("f1", "a.txt", 1, "aab")])
    add_chunks(store, [_chunk("f2", "b.txt", 0, "bbb")])
    add_chunks(store, [_chunk("f1", "a.txt", 0, "aaa")])  # re-index replaces old chunks

    docs = {d["file_id"]: d for d in list_documents(store)}
    assert docs["f1"]["n_chunks"] == 1 and docs["f2"]["n_chunks"] == 1

    hits = get_retriever(store, 6, ["f2"]).invoke("bbb")
    assert {h.metadata["file_id"] for h in hits} == {"f2"}

    delete_document(store, "f1")
    assert [d["file_id"] for d in list_documents(store)] == ["f2"]
