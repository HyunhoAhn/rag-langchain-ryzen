# AGENTS.md

## Project Identity

This repository is a small, developer-focused local RAG test project.

It is not an end-to-end product.
It is not a polished chatbot app.
It is not a frontend project.
It is not a deployment project.

The goal is to verify that the core RAG components work correctly on a Windows 11 Ryzen AI MAX machine:

1. Local LLM serving through Lemonade Server
2. LangChain-based RAG orchestration
3. Chroma persistent vector search
4. Local document ingestion
5. Retrieval inspection
6. Source-grounded answer generation
7. Basic retrieval evaluation

Keep the project intentionally small.

---

## Non-Goals

Do not implement these unless the user explicitly asks:

- Streamlit UI
- Gradio UI
- React frontend
- web dashboard
- user login
- authentication
- Docker
- cloud deployment
- CI/CD
- LangGraph
- agentic RAG
- multi-agent orchestration
- query rewriting
- rerankers
- hybrid search
- background jobs
- file upload UI
- Lemonade Server installation automation
- full benchmark harness
- LangSmith
- RAGAS
- tracing platforms

Future ideas may be documented in README.md under "Future Work", but they should not be implemented without direct instruction.

---

## Scope Control Rules

Prefer the smallest implementation that proves the feature works.

For each task:

1. Implement only the requested feature.
2. Avoid speculative abstractions.
3. Avoid creating extra layers.
4. Avoid adding dependencies unless necessary.
5. Avoid adding a UI unless explicitly requested.
6. Avoid adding production features.
7. Keep changes reviewable.
8. Keep tests runnable.

A good change usually touches 1 to 4 files.
If a change needs more than 5 files, explain why before editing.

Do not turn this project into a framework.

When uncertain, choose the simpler option.

The default answer to "should we add a UI?" is no.
The default answer to "should we add another framework?" is no.
The default answer to "should we implement this future improvement now?" is no.

---

## Preferred Interface

The primary interface is a developer CLI.

Required commands:

```powershell
python -m rag_app check
python -m rag_app ingest --data-dir .\data --reset
python -m rag_app retrieve "Ask Question"
python -m rag_app ask "Ask Question"
python -m rag_app eval --gold-file .\eval\gold_qa.example.json --top-k 5

The CLI should be sufficient to verify the complete RAG flow.

Do not add Streamlit by default.
Do not add FastAPI by default.
FastAPI may be added later only if explicitly requested.
