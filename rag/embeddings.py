"""Embedding model selection: OpenAI or Chroma's built-in default."""

import re

from langchain_core.embeddings import Embeddings

from rag.config import CHROMA_DEFAULT_EMBEDDING, COLLECTION_PREFIX


class ChromaDefaultEmbeddings(Embeddings):
    """LangChain adapter around Chroma's default local embedding (all-MiniLM-L6-v2, ONNX)."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(model: str, api_key: str | None = None) -> Embeddings:
    if model == CHROMA_DEFAULT_EMBEDDING:
        return ChromaDefaultEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model, api_key=api_key)


def collection_name_for(model: str) -> str:
    """One collection per embedding model, so vector dimensions never mix."""
    return COLLECTION_PREFIX + re.sub(r"[^a-zA-Z0-9_-]", "-", model)
