from langchain_core.documents import Document

from rag_app.config import AppConfig
from rag_app.retrieval import RetrievedChunk, format_retrieved_chunks, retrieve_documents


def _config(tmp_path):
    return AppConfig(
        lemonade_base_url="http://localhost:13305/v1",
        lemonade_chat_model="Qwen3-8B-GGUF",
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_model="test-model",
    )


def test_format_retrieved_chunks_includes_metadata_score_and_preview():
    output = format_retrieved_chunks(
        [
            RetrievedChunk(
                rank=1,
                source="notes.md",
                page=3,
                score=0.123456,
                text="x" * 800,
            )
        ]
    )

    assert "Rank: 1" in output
    assert "Source: notes.md" in output
    assert "Page: 3" in output
    assert "Score: 0.1235" in output
    assert output.endswith("x" * 800)


def test_retrieve_documents_uses_similarity_scores(tmp_path):
    config = _config(tmp_path)
    calls = {}

    class FakeCollection:
        def count(self):
            return 1

    class FakeVectorStore:
        _collection = FakeCollection()

        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def similarity_search_with_score(self, question, k):
            calls["similarity"] = {"question": question, "k": k}
            return [
                (
                    Document(
                        page_content="Local RAG chunk.",
                        metadata={"source": "guide.md", "page": 2},
                    ),
                    0.42,
                )
            ]

    result = retrieve_documents(
        config,
        "What is RAG?",
        top_k=3,
        search_type="similarity",
        embedding_factory=lambda received_config: "embeddings",
        vector_store_cls=FakeVectorStore,
    )

    assert calls["init"] == {
        "collection_name": "test_collection",
        "embedding_function": "embeddings",
        "persist_directory": str(tmp_path / "chroma"),
    }
    assert calls["similarity"] == {"question": "What is RAG?", "k": 3}
    assert result.chunks[0].source == "guide.md"
    assert result.chunks[0].page == 2
    assert result.chunks[0].score == 0.42


def test_retrieve_documents_defaults_to_mmr_without_scores(tmp_path):
    config = _config(tmp_path)
    calls = {}

    class FakeCollection:
        def count(self):
            return 1

    class FakeVectorStore:
        _collection = FakeCollection()

        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def max_marginal_relevance_search(self, question, k):
            calls["mmr"] = {"question": question, "k": k}
            return [Document(page_content="MMR chunk.", metadata={"source": "guide.md"})]

    result = retrieve_documents(
        config,
        "What is RAG?",
        top_k=2,
        embedding_factory=lambda received_config: "embeddings",
        vector_store_cls=FakeVectorStore,
    )

    assert calls["mmr"] == {"question": "What is RAG?", "k": 2}
    assert result.chunks[0].source == "guide.md"
    assert result.chunks[0].page is None
    assert result.chunks[0].score is None


def test_retrieve_documents_reports_missing_chroma_without_embeddings(tmp_path):
    config = _config(tmp_path)

    def fail_embedding_factory(received_config):
        raise AssertionError("Embeddings should not load when the Chroma directory is missing.")

    result = retrieve_documents(config, "What is RAG?", embedding_factory=fail_embedding_factory)

    assert result.store_ready is False
    assert result.chunks == []
