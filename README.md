# rag-langchain-ryzen

This repository is a small, developer-focused CLI project for verifying a local RAG flow on a Windows 11 Ryzen AI MAX machine.

It is not a product, chatbot UI, frontend, deployment project, or benchmark harness. The goal is to keep the project small while checking that Lemonade Server, LangChain orchestration, Chroma persistence, local ingestion, retrieval inspection, grounded answers, and basic evaluation can work together.

## Setup

```powershell
conda deactivate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
copy .env.example .env
```

## CLI

The intended interface is `python -m rag_app`.

```powershell
python -m rag_app check
python -m rag_app ingest --data-dir .\data --reset
python -m rag_app retrieve "Ask Question"
python -m rag_app ask "Ask Question"
python -m rag_app eval --gold-file .\eval\gold_qa.example.json --top-k 5
```

The commands are currently skeletons and print `not implemented yet`.

## Tests

```powershell
python -m pytest
```
