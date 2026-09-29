"""Application defaults and constants."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", PROJECT_ROOT / "chroma_db")).expanduser().resolve()

SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx"]

# Chunking
CHUNKING_STRATEGIES = ["Recursive", "Fixed size"]
DEFAULT_CHUNKING_STRATEGY = "Recursive"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4000
TOKEN_ENCODING = "cl100k_base"

# Retrieval / memory
DEFAULT_TOP_K = 6
MAX_TOP_K = 20
DEFAULT_MEMORY_WINDOW = 6
MIN_MEMORY_WINDOW = 2
MAX_MEMORY_WINDOW = 10
DEFAULT_TEMPERATURE = 0.0

# Models
PROVIDERS = ["Anthropic", "OpenAI"]
SUGGESTED_MODELS = {
    "Anthropic": ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
}
CUSTOM_MODEL_LABEL = "Custom…"

CHROMA_DEFAULT_EMBEDDING = "chroma-default"
EMBEDDING_OPTIONS = ["Chroma default (local, no key)", "OpenAI"]
OPENAI_EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

COLLECTION_PREFIX = "docs__"
