"""Load uploaded files into LangChain Documents with base metadata."""

import hashlib
import io
import os
from datetime import datetime, timezone

import docx2txt
from langchain_core.documents import Document
from pypdf import PdfReader

from rag.config import SUPPORTED_FILE_TYPES


class LoaderError(Exception):
    """Raised when a file can't be loaded or yields no text."""


def file_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decode_text(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _extract_pages(data: bytes, file_type: str) -> list[tuple[int, str]]:
    """Return (page, text) pairs; page is 1-based for PDFs, -1 otherwise (Chroma rejects None)."""
    if file_type == "pdf":
        reader = PdfReader(io.BytesIO(data))
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    if file_type == "docx":
        return [(-1, docx2txt.process(io.BytesIO(data)))]
    return [(-1, _decode_text(data))]


def load_file(file_name: str, data: bytes, upload_time: str | None = None) -> list[Document]:
    """Parse raw file bytes into Documents (one per PDF page, one per TXT/DOCX file)."""
    file_type = os.path.splitext(file_name)[1].lower().lstrip(".")
    if file_type not in SUPPORTED_FILE_TYPES:
        raise LoaderError(f"Unsupported file type: .{file_type}")

    upload_time = upload_time or datetime.now(timezone.utc).isoformat(timespec="seconds")
    file_id = file_id_for(data)

    try:
        pages = _extract_pages(data, file_type)
    except Exception as exc:  # corrupted / unreadable files
        raise LoaderError(f"Could not read {file_name}: {exc}") from exc

    docs = [
        Document(
            page_content=text,
            metadata={
                "file_name": file_name,
                "file_id": file_id,
                "file_type": file_type,
                "upload_time": upload_time,
                "page": page,
            },
        )
        for page, text in pages
        if text.strip()
    ]
    if not docs:
        raise LoaderError(f"No extractable text found in {file_name} (scanned or empty?)")
    return docs
