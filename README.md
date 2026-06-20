# Travel Copilot

A Retrieval-Augmented Generation (RAG) system that answers travel questions grounded in real travel-guide content, with citations. Built over the [Wikivoyage](https://en.wikivoyage.org) corpus, it combines hybrid retrieval, cross-encoder re-ranking, and tool-using generation, with a clean, provider-agnostic architecture.

> **Status:** The RAG core is complete and working. The project is actively being extended into a deployed, full-stack GenAI web application (FastAPI API + Next.js front-end on Azure) — see [Roadmap](#roadmap).

---

## What it does

Ask a natural-language travel question — *"where can I walk around a historic medieval city centre?"* — and the system retrieves the most relevant passages from the travel-guide corpus, then generates a grounded answer that cites its sources. It can also perform reliable budget/cost calculations via a sandboxed computation tool rather than trusting the language model's arithmetic.

## Architecture

The system separates two flows:

**Ingestion (offline):** `Wikivoyage XML dump → parse & clean → chunk → embed → store in vector DB`

**Query (runtime):** `question → hybrid retrieval → re-ranking → grounded, cited answer`

### Key design decisions

- **Provider-agnostic LLM access.** All model calls (chat + embeddings) go through a single module with a `PROVIDER` switch, so swapping from local Ollama to Azure OpenAI is a one-line change. Implementation details never leak past this boundary.
- **Config as data, behavior in modules.** Configuration holds only settings; logic lives in the modules that act on them. Invalid configuration fails fast with a clear error.
- **Section-aware chunking.** Chunking respects the source documents' own structure (Wikivoyage sections), with a recursive character-splitting fallback for unstructured articles. Each chunk carries metadata (title, section) for citations.
- **Two-stage retrieval.** Hybrid search (dense vector + sparse BM25, fused with Reciprocal Rank Fusion) maximizes recall; a cross-encoder re-ranker (FlashRank) then maximizes precision over the shortlist.
- **Tool-using generation with a sandboxed calculator.** Arithmetic is delegated to a safe expression evaluator (no arbitrary code execution) so numeric answers are correct, not hallucinated.

## Retrieval pipeline — measured improvement

The retrieval stack was built and evaluated in stages, with documented before/after results (see [`results/`](results/)):

| Stage | Method | Fixes |
|-------|--------|-------|
| 1 | Vector search only | Baseline — strong on concepts, weak on proper nouns |
| 2a | Hybrid (BM25 + vector + RRF) | **Recall** — proper-noun queries now retrieve the right content |
| 2b | + Cross-encoder re-ranking | **Precision** — the most relevant chunk is ranked first |

Each stage addresses the previous stage's weakness — a clear recall-then-precision progression.

## Tech stack

- **Language:** Python 3.11 (Poetry for dependency management)
- **LLM / embeddings:** Ollama (local) — `llama3.1`, `nomic-embed-text`; architecture supports Azure OpenAI
- **Vector store:** ChromaDB (cosine similarity, HNSW index)
- **Retrieval:** BM25 (`rank-bm25`) + dense vectors, fused with RRF; FlashRank cross-encoder re-ranking
- **Orchestration:** LangChain components, LangGraph
- **Parsing/chunking:** `mwparserfromhell`, `langchain-text-splitters`

## Project structure

```
src/travel_copilot/
├── config.py              # settings + provider switch (data only)
├── llm.py                 # single boundary for all LLM/embedding calls
├── ingestion/
│   ├── parse.py           # streaming XML parse + markup cleaning
│   ├── chunk.py           # section-aware chunking
│   └── index.py           # embed + store in ChromaDB
├── retrieval/
│   ├── hybrid.py          # BM25 + vector search, RRF fusion
│   └── rerank.py          # FlashRank cross-encoder re-ranking
└── tools.py               # retrieval tool + sandboxed calculator
scripts/
├── ask.py                 # full RAG loop: question → grounded, cited answer
└── peek.py                # inspect the vector store
results/                   # staged retrieval evaluation (before/after)
```

## Running it

```bash
# Install dependencies
poetry install

# Pull local models (requires Ollama)
ollama pull llama3.1
ollama pull nomic-embed-text

# Build the index from a Wikivoyage dump placed in data/
poetry run python -m src.travel_copilot.ingestion.index

# Ask a question
poetry run python -m scripts.ask "where can I walk around a historic medieval city centre?"
```

## Roadmap

The RAG core is complete. The project is being extended into a full-stack, deployable GenAI web application:

- [ ] **FastAPI** backend exposing the copilot as a REST API
- [ ] **Next.js + TypeScript** front-end (chat UI)
- [ ] **Docker** containerization and deployment to **Azure**
- [ ] **PostgreSQL + pgvector** for relational data and vector storage
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Evaluation harness (RAGAS) and observability

---

*Built as a hands-on exploration of production RAG architecture and end-to-end GenAI application engineering.*
