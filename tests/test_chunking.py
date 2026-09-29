import pytest
import tiktoken
from langchain_core.documents import Document

from rag.chunking import split

ENC = tiktoken.get_encoding("cl100k_base")
TEXT = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(800))


def _doc():
    return Document(page_content=TEXT, metadata={"file_name": "a.txt", "file_id": "x", "page": -1})


@pytest.mark.parametrize("strategy", ["Recursive", "Fixed size"])
def test_chunks_respect_token_limit_and_keep_metadata(strategy):
    chunks = split([_doc()], strategy, 600, 200)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert len(ENC.encode(c.page_content)) <= 600
        assert c.metadata["file_name"] == "a.txt"
        assert c.metadata["chunk_index"] == i
        assert c.metadata["chunking_strategy"] == strategy
        assert (c.metadata["chunk_size"], c.metadata["chunk_overlap"]) == (600, 200)


def test_fixed_chunks_overlap():
    chunks = split([_doc()], "Fixed size", 300, 50)
    tail = ENC.encode(chunks[0].page_content)[-50:]
    head = ENC.encode(chunks[1].page_content)[:50]
    assert tail == head


@pytest.mark.parametrize("size,overlap", [(600, 600), (50, 0), (600, -1)])
def test_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        split([_doc()], "Recursive", size, overlap)
