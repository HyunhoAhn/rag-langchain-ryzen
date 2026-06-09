"""Document ingestion into a persistent Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import AppConfig
from rag_app.loaders import load_documents


CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TEXT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "다. ",
    "요. ",
    "니다. ",
    " ",
    "",
]


@dataclass(frozen=True)
class IngestResult:
    source_document_count: int
    chunk_count: int
    chroma_dir: str
    collection_name: str


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=TEXT_SEPARATORS,
    )


def build_embeddings(config: AppConfig) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = build_text_splitter()
    return splitter.split_documents(documents)


def ingest_documents(
    config: AppConfig,
    data_dir: str | Path,
    *,
    reset: bool = False,
    document_loader: Callable[[Path], list[Document]] = load_documents,
    embedding_factory: Callable[[AppConfig], HuggingFaceEmbeddings] = build_embeddings,
    vector_store_cls: type[Chroma] = Chroma,
) -> IngestResult:
    source_documents = document_loader(Path(data_dir))
    if not source_documents:
        raise ValueError(f"No supported documents found in {data_dir}. Add .txt, .md, or .pdf files.")

    chunks = chunk_documents(source_documents)
    if not chunks:
        raise ValueError(f"No text chunks were created from documents in {data_dir}.")

    embeddings = embedding_factory(config)
    vector_store = vector_store_cls(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_dir,
    )

    if reset:
        vector_store.reset_collection()

    vector_store.add_documents(chunks)

    return IngestResult(
        source_document_count=len(source_documents),
        chunk_count=len(chunks),
        chroma_dir=config.chroma_dir,
        collection_name=config.collection_name,
    )
