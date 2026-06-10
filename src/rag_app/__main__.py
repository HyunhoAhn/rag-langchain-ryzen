"""Command line entry point for the local RAG verification project."""

from __future__ import annotations

import argparse
import sys

from rag_app.config import load_config
from rag_app.ingest import ingest_documents
from rag_app.lemonade_client import LemonadeCheckResult, check_lemonade_models
from rag_app.retrieval import empty_store_message, format_retrieved_chunks, retrieve_documents


def _format_model_found(result: LemonadeCheckResult) -> str:
    if result.model_found is True:
        return "yes"
    if result.model_found is False:
        return "no"
    return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_app",
        description="CLI skeleton for local RAG verification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Check local configuration and services.")

    ingest = subparsers.add_parser("ingest", help="Ingest local documents.")
    ingest.add_argument("--data-dir", default="./data", help="Directory containing source documents.")
    ingest.add_argument("--reset", action="store_true", help="Reset the vector store before ingesting.")

    retrieve = subparsers.add_parser("retrieve", help="Inspect retrieved documents for a question.")
    retrieve.add_argument("question", help="Question to retrieve context for.")
    retrieve.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    retrieve.add_argument(
        "--search-type",
        choices=("similarity", "mmr"),
        default="mmr",
        help="Chroma search type to use.",
    )

    ask = subparsers.add_parser("ask", help="Generate a source-grounded answer.")
    ask.add_argument("question", help="Question to answer.")

    evaluate = subparsers.add_parser("eval", help="Run a basic retrieval evaluation.")
    evaluate.add_argument("--gold-file", default="./eval/gold_qa.example.json", help="Gold QA JSON file.")
    evaluate.add_argument("--top-k", type=int, default=5, help="Number of retrieved documents to evaluate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check":
        config = load_config()
        result = check_lemonade_models(config)

        print(f"Lemonade base URL: {result.base_url}")
        print(f"Lemonade chat model: {result.model_name}")
        print(f"Models endpoint: {result.models_url}")
        print(f"Server reachable: {'yes' if result.reachable else 'no'}")
        print(f"Configured model listed: {_format_model_found(result)}")
        if result.error:
            print(f"Note: {result.error}")

        if not result.reachable or result.model_found is False:
            return 1
        return 0

    if args.command == "ingest":
        config = load_config()
        try:
            result = ingest_documents(config, args.data_dir, reset=args.reset)
        except ValueError as exc:
            print(f"Ingest failed: {exc}")
            return 1

        print(f"Ingested {result.source_document_count} source document(s).")
        print(f"Stored {result.chunk_count} chunk(s) in Chroma.")
        print(f"Chroma directory: {result.chroma_dir}")
        print(f"Collection: {result.collection_name}")
        return 0

    if args.command == "retrieve":
        config = load_config()
        result = retrieve_documents(
            config,
            args.question,
            top_k=args.top_k,
            search_type=args.search_type,
        )
        if not result.store_ready:
            print(empty_store_message(config))
            return 1
        if not result.chunks:
            print("No chunks matched the query.")
            return 0
        print(format_retrieved_chunks(result.chunks))
        return 0

    print("not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
