"""Source-grounded RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from rag_app.config import AppConfig
from rag_app.retrieval import RetrievedChunk, RetrievalResult, SearchType, retrieve_documents


UNSUPPORTED_ANSWER = "문서에서 근거를 찾지 못했습니다."


@dataclass(frozen=True)
class RagAnswerResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    retrieval_time_seconds: float
    generation_time_seconds: float
    total_time_seconds: float
    store_ready: bool = True


class LemonadeGenerationError(RuntimeError):
    """Raised when Lemonade Server cannot generate a chat completion."""


def format_docs_for_prompt(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        lines = [
            f"[chunk {chunk.rank}]",
            f"source: {chunk.source}",
        ]
        if chunk.page is not None:
            lines.append(f"page: {chunk.page}")
        lines.extend(["text:", chunk.text])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def format_source_list(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No chunks retrieved."

    lines: list[str] = []
    for chunk in chunks:
        source = f"chunk {chunk.rank}: {chunk.source}"
        if chunk.page is not None:
            source = f"{source} (page {chunk.page})"
        lines.append(source)
    return "\n".join(lines)


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "\n".join(
                    [
                        "제공된 Context만 사용해서 답변하세요.",
                        "답변은 한국어로 작성하세요.",
                        "추측하지 마세요.",
                        "Context에 답변 근거가 있으면 답변 마지막 줄에 `출처: chunk 1` 형식으로 근거 chunk 번호를 쓰세요.",
                        f'Context에 답변 근거가 없으면 다른 말 없이 정확히 "{UNSUPPORTED_ANSWER}"만 쓰세요.',
                        f'근거가 있는 답변에는 "{UNSUPPORTED_ANSWER}" 문장을 쓰지 마세요.',
                    ]
                ),
            ),
            ("human", "Question:\n{question}\n\nContext:\n{context}"),
        ]
    )


def _default_llm_factory(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=config.lemonade_base_url,
        model=config.lemonade_chat_model,
        api_key="not-needed",
        temperature=0,
    )


def build_rag_chain(llm: Runnable) -> Runnable:
    return _build_prompt() | llm | StrOutputParser()


def _source_line(chunks: list[RetrievedChunk]) -> str:
    return "출처: " + ", ".join(f"chunk {chunk.rank}" for chunk in chunks)


def finalize_answer(answer: str, chunks: list[RetrievedChunk]) -> str:
    stripped = answer.strip()
    if stripped == UNSUPPORTED_ANSWER:
        return stripped

    lines = [line.rstrip() for line in stripped.splitlines() if line.strip() != UNSUPPORTED_ANSWER]
    while lines and not lines[-1].strip():
        lines.pop()
    if not any(line.strip().startswith("출처:") for line in lines):
        if lines:
            lines.append("")
        lines.append(_source_line(chunks))
    return "\n".join(lines).strip()


def answer_question(
    config: AppConfig,
    question: str,
    *,
    top_k: int = 5,
    search_type: SearchType = "mmr",
    retriever: Callable[..., RetrievalResult] = retrieve_documents,
    llm_factory: Callable[[AppConfig], Runnable] = _default_llm_factory,
    clock: Callable[[], float] = time.perf_counter,
) -> RagAnswerResult:
    total_start = clock()

    retrieval_start = clock()
    retrieval_result = retriever(config, question, top_k=top_k, search_type=search_type)
    retrieval_time = clock() - retrieval_start

    if not retrieval_result.store_ready:
        return RagAnswerResult(
            question=question,
            answer="",
            chunks=[],
            retrieval_time_seconds=retrieval_time,
            generation_time_seconds=0.0,
            total_time_seconds=clock() - total_start,
            store_ready=False,
        )

    if not retrieval_result.chunks:
        return RagAnswerResult(
            question=question,
            answer=UNSUPPORTED_ANSWER,
            chunks=[],
            retrieval_time_seconds=retrieval_time,
            generation_time_seconds=0.0,
            total_time_seconds=clock() - total_start,
        )

    chain = build_rag_chain(llm_factory(config))
    context = format_docs_for_prompt(retrieval_result.chunks)

    generation_start = clock()
    try:
        answer = chain.invoke({"question": question, "context": context})
    except Exception as exc:
        raise LemonadeGenerationError(str(exc)) from exc

    generation_time = clock() - generation_start

    return RagAnswerResult(
        question=question,
        answer=finalize_answer(answer, retrieval_result.chunks),
        chunks=retrieval_result.chunks,
        retrieval_time_seconds=retrieval_time,
        generation_time_seconds=generation_time,
        total_time_seconds=clock() - total_start,
    )
