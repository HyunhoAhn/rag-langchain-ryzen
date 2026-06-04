"""Command line entry point for the local RAG verification project."""

from __future__ import annotations

import argparse


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

    ask = subparsers.add_parser("ask", help="Generate a source-grounded answer.")
    ask.add_argument("question", help="Question to answer.")

    evaluate = subparsers.add_parser("eval", help="Run a basic retrieval evaluation.")
    evaluate.add_argument("--gold-file", default="./eval/gold_qa.example.json", help="Gold QA JSON file.")
    evaluate.add_argument("--top-k", type=int, default=5, help="Number of retrieved documents to evaluate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print("not implemented yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
