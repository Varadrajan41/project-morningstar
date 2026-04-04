# Project Morningstar

A **local-first research assistant** for staying current on AI, cybersecurity, and identity-related work—without sending your reading list or questions to a hosted API by default. It ingests curated sources, stores them in a **local vector database**, and answers questions through a small **Streamlit** UI with optional live web fallback.

## Why this exists

The goal is a personal “morning digest + memory” loop: automate discovery and scoring of noisy feeds (ArXiv, then the open web), keep **only what passes a relevance bar**, and make that corpus **searchable in natural language** offline on your machine (via **Ollama**). The project is intentionally small and scriptable so you can schedule ingestion and browse or query the same memory from a browser.

## What it does (at a glance)

| Piece | Role |
|--------|------|
| **`digest_generator.py`** | Fetches recent **ArXiv** papers, scores them with a local LLM, writes a dated **Markdown digest** (`Morningstar_Digest_YYYY-MM-DD.md`), and embeds high-scoring items into ChromaDB (`daily_research`). |
| **`web_Scout.py`** | A **LangGraph** pipeline: web search → score → optional **full-page extraction** for top hits → writes **snippets** and **deep** chunks into two Chroma collections (`daily_research`, `deep_dive_research`). |
| **`app.py`** | **Streamlit** chat: hybrid retrieval (embeddings + BM25 + fusion) over the vault you select; if local context is thin, it can fall back to live web snippets. |
| **`run_morningstar.sh`** | Example entrypoint: activate venv and run the daily ArXiv digest (e.g. from cron or Task Scheduler). |

**`query_morningstar.py`** is a minimal CLI over the same database if you prefer the terminal over the UI.

## Prerequisites

- **Python 3.10+** (project uses a virtualenv under `venv/` in this repo).
- **Ollama** with models aligned to the scripts (e.g. `qwen2.5:7b-instruct`, `nomic-embed-text`)—see `app.py` / `digest_generator.py` for the exact names.
- Dependencies: `pip install -r requirements.txt` (Streamlit, ChromaDB, Ollama client, ArXiv, web search, LangGraph, trafilatura, BM25, etc.).

## Quick start

```bash
cd project-morningstar
source venv/bin/activate   # Windows: venv\Scripts\activate
streamlit run app.py
```

Run ingestion on demand: `python digest_generator.py` or `python web_Scout.py` (adjust the search topic in `web_Scout.py` as needed).

---

*Internal / personal tooling; adapt prompts, models, and schedules to your environment.*
