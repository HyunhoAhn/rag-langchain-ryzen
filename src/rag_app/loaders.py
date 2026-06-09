"""Document loading helpers for local source files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_documents(data_dir: Path) -> list[Document]:
    """Recursively load supported documents from a data directory."""
    if not data_dir.exists():
        return []

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")

        documents.extend(loader.load())

    return documents
