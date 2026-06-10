# rag-langchain-ryzen

This repository is a small, developer-focused CLI project for verifying a local RAG flow on a Windows 11 Ryzen AI MAX machine.

It is not a product, chatbot UI, frontend, deployment project, or benchmark harness. The goal is to keep the project small while checking that Lemonade Server, LangChain orchestration, Chroma persistence, local ingestion, retrieval inspection, grounded answers, and basic evaluation can work together.

## Setup

```powershell
conda deactivate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

## CLI

The intended interface is `python -m rag_app`, run through the repository virtual environment.

```powershell
.\.venv\Scripts\python.exe -m rag_app check
.\.venv\Scripts\python.exe -m rag_app ingest --data-dir .\data --reset
.\.venv\Scripts\python.exe -m rag_app retrieve "Ask Question" --top-k 5
.\.venv\Scripts\python.exe -m rag_app ask "Ask Question"
.\.venv\Scripts\python.exe -m rag_app eval --gold-file .\eval\gold_qa.example.json --top-k 5
```

Currently implemented:

```powershell
.\.venv\Scripts\python.exe -m rag_app check
.\.venv\Scripts\python.exe -m rag_app ingest --data-dir .\data --reset
.\.venv\Scripts\python.exe -m rag_app retrieve "Ask Question" --top-k 5
```

The check command calls the OpenAI-compatible Lemonade Server `/models` endpoint, prints the configured `LEMONADE_BASE_URL` and `LEMONADE_CHAT_MODEL`, and reports whether the configured model appears in the returned model list when the response includes one.

The ingest command loads `.txt`, `.md`, and `.pdf` files from `--data-dir`, chunks them with a `RecursiveCharacterTextSplitter`, embeds them with `HuggingFaceEmbeddings`, and writes them to the configured persistent Chroma collection. It uses `EMBEDDING_MODEL` from `.env` or the environment, defaulting to `BAAI/bge-m3`, on CPU with normalized embeddings. The first run may download the embedding model through `sentence-transformers`.

The retrieve command loads the configured persistent Chroma collection with the same embedding settings used by ingest, then prints the rank, source, page when present, score when available, and the first 800 characters of each retrieved chunk. It defaults to MMR search and also supports `--search-type similarity`.

The ask and eval commands are still placeholders.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```
