from langchain_core.documents import Document

from rag_app.config import AppConfig
from rag_app.ingest import TEXT_SEPARATORS, build_embeddings, chunk_documents, ingest_documents


def _config(tmp_path):
    return AppConfig(
        lemonade_base_url="http://localhost:13305/v1",
        lemonade_chat_model="Qwen3-8B-GGUF",
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_model="test-model",
    )


def test_chunk_documents_uses_expected_splitter_settings():
    document = Document(page_content="가" * 900 + "\n\n" + "English text.", metadata={"source": "notes.md"})

    chunks = chunk_documents([document])

    assert len(chunks) >= 2
    assert all(len(chunk.page_content) <= 800 for chunk in chunks)
    assert "다. " in TEXT_SEPARATORS
    assert ". " in TEXT_SEPARATORS


def test_build_embeddings_uses_cpu_and_normalization(monkeypatch, tmp_path):
    captured = {}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("rag_app.ingest.HuggingFaceEmbeddings", FakeEmbeddings)

    embeddings = build_embeddings(_config(tmp_path))

    assert isinstance(embeddings, FakeEmbeddings)
    assert captured["model_name"] == "test-model"
    assert captured["model_kwargs"] == {"device": "cpu"}
    assert captured["encode_kwargs"] == {"normalize_embeddings": True}


def test_ingest_documents_resets_and_adds_chunks(tmp_path):
    calls = {"reset": 0, "added": []}

    class FakeVectorStore:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def reset_collection(self):
            calls["reset"] += 1

        def add_documents(self, documents):
            calls["added"] = documents

    result = ingest_documents(
        _config(tmp_path),
        tmp_path / "data",
        reset=True,
        document_loader=lambda path: [Document(page_content="Local RAG notes.", metadata={"source": str(path)})],
        embedding_factory=lambda config: "embeddings",
        vector_store_cls=FakeVectorStore,
    )

    assert calls["init"] == {
        "collection_name": "test_collection",
        "embedding_function": "embeddings",
        "persist_directory": str(tmp_path / "chroma"),
    }
    assert calls["reset"] == 1
    assert len(calls["added"]) == result.chunk_count == 1
    assert result.source_document_count == 1


def test_ingest_documents_fails_when_no_documents(tmp_path):
    try:
        ingest_documents(
            _config(tmp_path),
            tmp_path / "empty",
            document_loader=lambda path: [],
            embedding_factory=lambda config: "embeddings",
        )
    except ValueError as exc:
        assert "No supported documents found" in str(exc)
    else:
        raise AssertionError("Expected ingest_documents to fail without documents.")
