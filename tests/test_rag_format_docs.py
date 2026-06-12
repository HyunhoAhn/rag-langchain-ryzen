from langchain_core.runnables import RunnableLambda

from rag_app.config import AppConfig
from rag_app.rag_chain import (
    UNSUPPORTED_ANSWER,
    answer_question,
    finalize_answer,
    format_docs_for_prompt,
    format_source_list,
)
from rag_app.retrieval import RetrievedChunk, RetrievalResult


def _config(tmp_path):
    return AppConfig(
        lemonade_base_url="http://localhost:13305/v1",
        lemonade_chat_model="Qwen3-8B-GGUF",
        chroma_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_model="test-model",
    )


def test_format_docs_for_prompt_labels_chunks_and_metadata():
    output = format_docs_for_prompt(
        [
            RetrievedChunk(
                rank=1,
                source="notes.md",
                page=3,
                text="Local RAG uses Chroma.",
            )
        ]
    )

    assert "[chunk 1]" in output
    assert "source: notes.md" in output
    assert "page: 3" in output
    assert "text:\nLocal RAG uses Chroma." in output


def test_format_source_list_includes_chunk_numbers():
    output = format_source_list(
        [
            RetrievedChunk(rank=1, source="notes.md", page=3, text="one"),
            RetrievedChunk(rank=2, source="guide.md", page=None, text="two"),
        ]
    )

    assert output == "chunk 1: notes.md (page 3)\nchunk 2: guide.md"


def test_finalize_answer_keeps_supported_answer_and_source_line():
    answer = finalize_answer(
        f"LangChain과 Chroma를 사용합니다.\n\n{UNSUPPORTED_ANSWER}",
        [RetrievedChunk(rank=1, source="notes.md", page=None, text="one")],
    )

    assert answer == "LangChain과 Chroma를 사용합니다.\n\n출처: chunk 1"


def test_finalize_answer_does_not_duplicate_inline_source_marker():
    answer = finalize_answer(
        "LangChain과 Chroma를 사용합니다. 출처: chunk 1",
        [RetrievedChunk(rank=1, source="notes.md", page=None, text="one")],
    )

    assert answer == "LangChain과 Chroma를 사용합니다. 출처: chunk 1"


def test_finalize_answer_preserves_exact_unsupported_answer():
    answer = finalize_answer(
        UNSUPPORTED_ANSWER,
        [RetrievedChunk(rank=1, source="notes.md", page=None, text="one")],
    )

    assert answer == UNSUPPORTED_ANSWER


def test_answer_question_uses_retriever_and_lcel_chain(tmp_path):
    calls = {}

    def fake_retriever(config, question, *, top_k, search_type):
        calls["retriever"] = {"question": question, "top_k": top_k, "search_type": search_type}
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    rank=1,
                    source="notes.md",
                    page=None,
                    text="RAG components include retrieval and generation.",
                )
            ],
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
        )

    llm = RunnableLambda(lambda prompt_value: "검색과 생성을 사용합니다.\n\n출처: chunk 1")

    result = answer_question(
        _config(tmp_path),
        "What are the RAG components?",
        top_k=3,
        retriever=fake_retriever,
        llm_factory=lambda config: llm,
    )

    assert calls["retriever"] == {
        "question": "What are the RAG components?",
        "top_k": 3,
        "search_type": "mmr",
    }
    assert result.answer == "검색과 생성을 사용합니다.\n\n출처: chunk 1"
    assert result.chunks[0].source == "notes.md"


def test_answer_question_returns_unsupported_answer_without_chunks(tmp_path):
    def fake_retriever(config, question, *, top_k, search_type):
        return RetrievalResult(
            chunks=[],
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
        )

    def fail_llm_factory(config):
        raise AssertionError("LLM should not run when retrieval returns no chunks.")

    result = answer_question(
        _config(tmp_path),
        "Unknown?",
        retriever=fake_retriever,
        llm_factory=fail_llm_factory,
    )

    assert result.answer == UNSUPPORTED_ANSWER
    assert result.generation_time_seconds == 0.0
