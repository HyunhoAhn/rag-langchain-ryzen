"""Simple retrieval evaluation for the local RAG CLI."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

from rag_app.config import AppConfig
from rag_app.retrieval import RetrievalResult, retrieve_documents


@dataclass(frozen=True)
class GoldQuestion:
    question: str
    expected_source_contains: str


@dataclass(frozen=True)
class EvaluationItem:
    question: str
    expected_source_contains: str
    retrieved_sources: list[str]
    hit: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationResult:
    items: list[EvaluationItem]
    hit_at_k: float
    mrr: float
    hit_count: int
    total_count: int
    store_ready: bool = True


Retriever = Callable[..., RetrievalResult]


def load_gold_questions(gold_file: str | Path) -> list[GoldQuestion]:
    path = Path(gold_file)
    try:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read gold file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gold file is not valid JSON: {path}") from exc

    if not isinstance(raw_items, list):
        raise ValueError("Gold file must contain a JSON list.")
    if not raw_items:
        raise ValueError("Gold file must contain at least one item.")

    gold_questions: list[GoldQuestion] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            raise ValueError(f"Gold item {index} must be an object.")

        question = raw_item.get("question")
        expected_source_contains = raw_item.get("expected_source_contains")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Gold item {index} must include a non-empty question.")
        if not isinstance(expected_source_contains, str) or not expected_source_contains.strip():
            raise ValueError(f"Gold item {index} must include a non-empty expected_source_contains.")

        gold_questions.append(
            GoldQuestion(
                question=question,
                expected_source_contains=expected_source_contains,
            )
        )

    return gold_questions


def reciprocal_rank(retrieved_sources: list[str], expected_source_contains: str) -> float:
    for index, source in enumerate(retrieved_sources, start=1):
        if expected_source_contains in source:
            return 1.0 / index
    return 0.0


def evaluate_retrieval(
    config: AppConfig,
    gold_file: str | Path,
    *,
    top_k: int = 5,
    retriever: Retriever = retrieve_documents,
) -> EvaluationResult:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    gold_questions = load_gold_questions(gold_file)
    items: list[EvaluationItem] = []

    for gold in gold_questions:
        retrieval_result = retriever(config, gold.question, top_k=top_k, search_type="mmr")
        if not retrieval_result.store_ready:
            return EvaluationResult(
                items=items,
                hit_at_k=0.0,
                mrr=0.0,
                hit_count=0,
                total_count=len(gold_questions),
                store_ready=False,
            )

        sources = [chunk.source for chunk in retrieval_result.chunks]
        rank_score = reciprocal_rank(sources, gold.expected_source_contains)
        items.append(
            EvaluationItem(
                question=gold.question,
                expected_source_contains=gold.expected_source_contains,
                retrieved_sources=sources,
                hit=rank_score > 0.0,
                reciprocal_rank=rank_score,
            )
        )

    hit_count = sum(1 for item in items if item.hit)
    total_count = len(items)
    return EvaluationResult(
        items=items,
        hit_at_k=hit_count / total_count,
        mrr=sum(item.reciprocal_rank for item in items) / total_count,
        hit_count=hit_count,
        total_count=total_count,
    )


def format_evaluation_report(result: EvaluationResult, top_k: int) -> str:
    lines: list[str] = []
    for index, item in enumerate(result.items, start=1):
        lines.extend(
            [
                f"Question {index}: {item.question}",
                "Retrieved sources:",
            ]
        )
        if item.retrieved_sources:
            for rank, source in enumerate(item.retrieved_sources, start=1):
                lines.append(f"  {rank}. {source}")
        else:
            lines.append("  (none)")
        lines.extend(
            [
                f"Expected source contains: {item.expected_source_contains}",
                f"Hit: {'yes' if item.hit else 'no'}",
                f"Reciprocal rank: {item.reciprocal_rank:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "Summary:",
            f"Hit@{top_k}: {result.hit_count}/{result.total_count} = {result.hit_at_k:.4f}",
            f"MRR: {result.mrr:.4f}",
        ]
    )
    return "\n".join(lines)
