"""Token-based chunking strategies."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from rag.config import MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, TOKEN_ENCODING


def validate(chunk_size: int, chunk_overlap: int) -> None:
    if not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError(f"Chunk size must be between {MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE} tokens.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Chunk overlap must be ≥ 0 and smaller than the chunk size.")


def _splitter(strategy: str, chunk_size: int, chunk_overlap: int):
    if strategy == "Fixed size":
        return TokenTextSplitter(
            encoding_name=TOKEN_ENCODING, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=TOKEN_ENCODING, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def split(docs: list[Document], strategy: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split documents, preserving parent metadata and adding chunk metadata."""
    validate(chunk_size, chunk_overlap)
    chunks = _splitter(strategy, chunk_size, chunk_overlap).split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            chunk_index=i,
            chunking_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    return chunks
