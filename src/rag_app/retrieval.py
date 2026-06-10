"""Retrieve documents from the persistent Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from rag_app.config import AppConfig
from rag_app.ingest import build_embeddings


SearchType = Literal["similarity", "mmr"]
CHUNK_PREVIEW_CHARS = 800


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    source: str
    page: int | str | None
    text: str
    score: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    chroma_dir: str
    collection_name: str
    store_ready: bool = True


def _metadata_value(metadata: dict, key: str) -> str | int | None:
    value = metadata.get(key)
    if value is None or value == "":
        return None
    return value


def _build_chunk(rank: int, document: Document, score: float | None = None) -> RetrievedChunk:
    metadata = document.metadata or {}
    source = str(_metadata_value(metadata, "source") or "unknown")
    page = _metadata_value(metadata, "page")
    return RetrievedChunk(
        rank=rank,
        source=source,
        page=page,
        score=score,
        text=document.page_content[:CHUNK_PREVIEW_CHARS],
    )


def format_retrieved_chunks(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        lines = [
            f"Rank: {chunk.rank}",
            f"Source: {chunk.source}",
        ]
        if chunk.page is not None:
            lines.append(f"Page: {chunk.page}")
        if chunk.score is not None:
            lines.append(f"Score: {chunk.score:.4f}")
        lines.extend(["Text:", chunk.text])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def empty_store_message(config: AppConfig) -> str:
    return (
        f"No indexed documents found in Chroma collection '{config.collection_name}' "
        f"at '{config.chroma_dir}'. Run ingest first:\n"
        r".\.venv\Scripts\python.exe -m rag_app ingest --data-dir .\data --reset"
    )


def _collection_count(vector_store: Chroma) -> int:
    return int(vector_store._collection.count())


def _stored_collection_count(config: AppConfig) -> int:
    chroma_path = Path(config.chroma_dir)
    if not chroma_path.exists():
        return 0

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection(config.collection_name)
    except Exception:
        return 0
    return int(collection.count())


def retrieve_documents(
    config: AppConfig,
    question: str,
    *,
    top_k: int = 5,
    search_type: SearchType = "mmr",
    embedding_factory: Callable[[AppConfig], HuggingFaceEmbeddings] = build_embeddings,
    vector_store_cls: type[Chroma] = Chroma,
) -> RetrievalResult:
    if vector_store_cls is Chroma and _stored_collection_count(config) == 0:
        return RetrievalResult(
            chunks=[],
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
            store_ready=False,
        )

    embeddings = embedding_factory(config)
    vector_store = vector_store_cls(
        collection_name=config.collection_name,
        embedding_function=embeddings,
        persist_directory=config.chroma_dir,
    )

    if _collection_count(vector_store) == 0:
        return RetrievalResult(
            chunks=[],
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
            store_ready=False,
        )

    if search_type == "similarity":
        results = vector_store.similarity_search_with_score(question, k=top_k)
        chunks = [_build_chunk(rank, document, score) for rank, (document, score) in enumerate(results, start=1)]
    elif search_type == "mmr":
        documents = vector_store.max_marginal_relevance_search(question, k=top_k)
        chunks = [_build_chunk(rank, document) for rank, document in enumerate(documents, start=1)]
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

    return RetrievalResult(
        chunks=chunks,
        chroma_dir=config.chroma_dir,
        collection_name=config.collection_name,
    )
