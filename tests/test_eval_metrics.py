import json

import pytest

from rag_app.config import AppConfig
from rag_app.eval_retrieval import (
    evaluate_retrieval,
    format_evaluation_report,
    load_gold_questions,
    reciprocal_rank,
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


def _write_gold(tmp_path, items):
    gold_file = tmp_path / "gold.json"
    gold_file.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return gold_file


def test_load_gold_questions_requires_retrieval_schema(tmp_path):
    gold_file = _write_gold(
        tmp_path,
        [{"question": "question", "expected_source_contains": "sample.md"}],
    )

    questions = load_gold_questions(gold_file)

    assert questions[0].question == "question"
    assert questions[0].expected_source_contains == "sample.md"


def test_load_gold_questions_rejects_missing_expected_source(tmp_path):
    gold_file = _write_gold(tmp_path, [{"question": "question"}])

    with pytest.raises(ValueError, match="expected_source_contains"):
        load_gold_questions(gold_file)


def test_load_gold_questions_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="Could not read gold file"):
        load_gold_questions(tmp_path / "missing.json")


def test_reciprocal_rank_uses_first_matching_source():
    assert reciprocal_rank(["other.md", "data/sample.md", "sample.md"], "sample.md") == 0.5
    assert reciprocal_rank(["other.md"], "sample.md") == 0.0


def test_evaluate_retrieval_computes_hit_at_k_and_mrr(tmp_path):
    gold_file = _write_gold(
        tmp_path,
        [
            {"question": "first", "expected_source_contains": "sample.md"},
            {"question": "second", "expected_source_contains": "missing.md"},
        ],
    )
    calls = []

    def fake_retriever(config, question, *, top_k, search_type):
        calls.append({"question": question, "top_k": top_k, "search_type": search_type})
        chunks = [
            RetrievedChunk(rank=1, source="other.md", page=None, text="other"),
            RetrievedChunk(rank=2, source="data/sample.md", page=None, text="sample"),
        ]
        return RetrievalResult(chunks=chunks, chroma_dir=config.chroma_dir, collection_name=config.collection_name)

    result = evaluate_retrieval(_config(tmp_path), gold_file, top_k=5, retriever=fake_retriever)

    assert calls == [
        {"question": "first", "top_k": 5, "search_type": "mmr"},
        {"question": "second", "top_k": 5, "search_type": "mmr"},
    ]
    assert result.hit_count == 1
    assert result.total_count == 2
    assert result.hit_at_k == 0.5
    assert result.mrr == 0.25
    assert result.items[0].reciprocal_rank == 0.5
    assert result.items[1].reciprocal_rank == 0.0


def test_format_evaluation_report_lists_sources_and_metrics(tmp_path):
    gold_file = _write_gold(
        tmp_path,
        [{"question": "question", "expected_source_contains": "sample.md"}],
    )

    def fake_retriever(config, question, *, top_k, search_type):
        return RetrievalResult(
            chunks=[RetrievedChunk(rank=1, source="data/sample.md", page=None, text="sample")],
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
        )

    result = evaluate_retrieval(_config(tmp_path), gold_file, top_k=5, retriever=fake_retriever)
    report = format_evaluation_report(result, top_k=5)

    assert "Question 1: question" in report
    assert "  1. data/sample.md" in report
    assert "Hit: yes" in report
    assert "Hit@5: 1/1 = 1.0000" in report
    assert "MRR: 1.0000" in report
