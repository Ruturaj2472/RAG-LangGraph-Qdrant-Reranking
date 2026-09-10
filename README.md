# LangGraph + Qdrant Hybrid Search + Open-Source Reranking

A hands-on, single-notebook reference implementation of a Retrieval-Augmented Generation (RAG) pipeline that combines:

- LangGraph for stateful, graph-based LLM orchestration
- Qdrant for dense + sparse hybrid vector search
- Open-source cross-encoder reranking
- OpenAI (`gpt-4o-mini`) for final answer generation

Runs end-to-end in Google Colab with no external vector database service and no paid embedding/reranking API — only the generation step calls out to OpenAI.

---

## Table of Contents

- Overview
- Architecture / Workflow
- Tech Stack
- Repository Structure
- Notebook Sections
- Setup
- How Hybrid Search Works
- How Reranking Works
- LangGraph State & Nodes
- Running the Pipeline
- Extending This Project
- License

---

## Overview

| Aspect | Details |
|---|---|
| Purpose | Educational reference for hybrid retrieval + reranking + agentic orchestration |
| Environment | Google Colab (CPU-friendly) |
| Generation LLM | OpenAI `gpt-4o-mini` (via API key) |
| Embeddings | Open-source, local (`fastembed`) |
| Reranker | Open-source, local (`sentence-transformers`) |
| Vector Store | Qdrant (in-memory instance, swappable to Qdrant Cloud/server) |
| Orchestration | LangGraph `StateGraph` |

---

## Architecture / Workflow

High-level pipeline executed by the compiled LangGraph graph:

    User Query
        |
        v
    [retrieve_node] ---> Hybrid Search (Qdrant)
        |                   - Dense prefetch (semantic, cosine similarity)
        |                   - Sparse prefetch (lexical, BM25-style)
        |                   - Fusion via Reciprocal Rank Fusion (RRF)
        v
    [rerank_node] ---> CrossEncoder re-scores (query, doc) pairs
        |                 - cross-encoder/ms-marco-MiniLM-L-6-v2
        |                 - Keeps top-N most relevant documents
        v
    [generate_node] ---> Builds context from top-N docs
        |                  - Prompts OpenAI gpt-4o-mini
        v
    Final Answer

Retrieval detail (inside `retrieve_node`, via `hybrid_search`):

    Query
      |--> embed_dense()  --> Prefetch(using="dense",  limit=prefetch_k)
      |--> embed_sparse() --> Prefetch(using="sparse", limit=prefetch_k)
      |
      v
    Qdrant merges both ranked lists via FusionQuery(fusion=RRF)
      |
      v
    Top-K fused results returned

---

## Tech Stack

| Component | Library / Model | Role |
|---|---|---|
| Orchestration | `langgraph`, `langchain` | Stateful graph execution |
| Generation LLM | `langchain-openai` (`gpt-4o-mini`) | Final answer synthesis |
| Vector Database | `qdrant-client` (in-memory) | Storage + hybrid retrieval |
| Dense Embeddings | `fastembed` — `BAAI/bge-small-en-v1.5` | Semantic vector representation (384-dim) |
| Sparse Embeddings | `fastembed` — `Qdrant/bm25` | Lexical/keyword vector representation |
| Reranker | `sentence-transformers` — `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-document relevance scoring |

---

## Repository Structure

    .
    |-- LangGraph_Qdrant_Hybrid_Rerank.ipynb   Main notebook (setup, retrieval, rerank, graph, runs)
    |-- README.md                              This file
    |-- .gitignore                             Ignore rules for caches, secrets, envs

---

## Notebook Sections

| # | Section | What Happens |
|---|---|---|
| 1 | LangGraph Introduction | Explains State, Node, Edge concepts vs. linear LCEL chains |
| 2 | Setup & Environment | Installs dependencies, loads OpenAI key from Colab Secrets |
| 3 | Embedding & Sparse Vector Setup | Initializes dense (`bge-small-en-v1.5`) and sparse (`Qdrant/bm25`) models |
| 4 | Qdrant Collection Setup | Creates in-memory collection with named `dense` + `sparse` vectors |
| 5 | Ingestion Pipeline | Embeds and upserts an 18-document sample corpus |
| 6 | Hybrid Search in Qdrant | Compares dense-only, sparse-only, and RRF-fused hybrid search |
| 7 | Open-Source Reranking | CrossEncoder reranks top hybrid candidates |
| 8 | LangGraph Agent Construction | Defines `RAGState`, three nodes, compiles the graph |
| 9 | Running the Graph | `.invoke()` and `.stream()` examples, batch test queries |

---

## Setup

1. Open the notebook in Google Colab.
2. Add your OpenAI key via Colab Secrets:
   - Click the key icon in the left sidebar
   - Add a secret named `OPENAI_API_KEY`
   - Toggle notebook access ON
3. Run all cells top to bottom. Dependencies install automatically via the first code cell:

       pip install -q langgraph langchain langchain-openai langchain-core qdrant-client langchain-qdrant fastembed sentence-transformers

No local GPU or Qdrant server is required — embeddings/reranking run on CPU, and Qdrant runs in-memory (`QdrantClient(":memory:")`).

---

## How Hybrid Search Works

| Search Type | Captures | Weakness | Query Object Used |
|---|---|---|---|
| Dense | Semantic meaning (`car` ~ `automobile`) | Misses exact keyword/ID matches | Plain `List[float]` |
| Sparse | Exact term/keyword overlap (BM25-style) | Misses semantic similarity | `SparseVector(indices=..., values=...)` |
| Hybrid (RRF) | Best of both, via rank-based fusion | Slightly higher latency (two searches) | `FusionQuery(fusion=Fusion.RRF)` |

Reciprocal Rank Fusion combines the two independently ranked candidate lists using only rank position (not raw scores, which live on incomparable scales):

    RRF_score(doc) = sum over each list containing doc of  1 / (k + rank_in_that_list)

Documents ranked consistently well across both dense and sparse lists rise to the top of the fused result, even if neither list alone ranked them first.

---

## How Reranking Works

1. Retrieve a larger candidate set cheaply (e.g., top 10-20) via hybrid search.
2. Form `(query, document)` pairs for every candidate.
3. Score each pair jointly using the CrossEncoder (`ms-marco-MiniLM-L-6-v2`), which is slower but more accurate than vector similarity alone.
4. Sort by score, descending, and keep only the top-N (default 3).

This two-stage retrieve-then-rerank pattern balances speed (broad recall from vector search) with precision (accurate final ranking from the cross-encoder).

---

## LangGraph State & Nodes

| Field (in `RAGState`) | Type | Set By |
|---|---|---|
| `query` | `str` | Initial input |
| `retrieved_docs` | `List[dict]` | `retrieve_node` |
| `reranked_docs` | `List[dict]` | `rerank_node` |
| `answer` | `str` | `generate_node` |

| Node | Input Used | Output Produced |
|---|---|---|
| `retrieve_node` | `query` | `retrieved_docs` (top-10 hybrid results) |
| `rerank_node` | `retrieved_docs`, `query` | `reranked_docs` (top-3 after CrossEncoder) |
| `generate_node` | `reranked_docs`, `query` | `answer` (LLM-generated response) |

Graph edges: `START -> retrieve -> rerank -> generate -> END` (linear, but structured to make future conditional branching straightforward).

---

## Running the Pipeline

Invoke with a single query:

    result = graph.invoke({"query": "What is the difference between LangGraph and a LangChain LCEL chain?"})
    print(result["answer"])

Stream intermediate state after each node (useful for debugging):

    for step in graph.stream({"query": "How does hybrid search combine dense and sparse retrieval?"}):
        print(step)

---

## Extending This Project

| Idea | How |
|---|---|
| Stronger embeddings | Swap in `BAAI/bge-base-en-v1.5` or `intfloat/e5-large-v2` |
| Stronger reranker | Swap in `BAAI/bge-reranker-base` or `-large` (GPU recommended) |
| Learned sparse retrieval | Replace `Qdrant/bm25` with SPLADE for term expansion |
| Multi-turn memory | Add `chat_history` to `RAGState`; use LangGraph checkpointing (`MemorySaver`) |
| Conditional branching | Route to a `clarify_node` if retrieval returns nothing relevant |
| Corrective RAG loop | Loop back to `retrieve_node` with a reformulated query if the answer is unsupported |
| Production readiness | Swap `QdrantClient(":memory:")` for a persistent Qdrant server/cloud instance; add payload filtering; batch embedding calls |

---

## License

Add your preferred license here (e.g., MIT) before publishing.
