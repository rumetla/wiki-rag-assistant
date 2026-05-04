# Production Deployment Recommendations

This document outlines what would change if the **Local Wikipedia RAG Assistant** moved from an educational localhost demo to a real production service. It covers infrastructure, model serving, retrieval quality, observability, security, and tradeoffs.

---

## 1. Reference deployment topology

```
                    +------------------+
                    |  Load balancer   |
                    +--------+---------+
                             |
              +--------------+---------------+
              |                              |
       +------v-----+                 +------v-----+
       | API server | ... (replicas)  | API server |
       +-----+------+                 +------+-----+
             |                                |
             |  gRPC/HTTP                     |
             v                                v
       +----------------+              +-----------------+
       | LLM inference  |              | Embedding svc   |
       | (vLLM / TGI)   |              | (TEI / vLLM)    |
       +----------------+              +-----------------+
             |
             v
       +-----------------+        +------------------+
       | Vector DB       |<------>| Object storage   |
       | (Qdrant /       |        | (S3 for raw docs)|
       |  pgvector)      |        +------------------+
       +-----------------+
             |
       +-----v-----+
       | Postgres  |  (chat history, users, audit)
       +-----------+
```

---

## 2. Infrastructure & runtime

### 2.1 Containerize everything
- One container per service: `api`, `llm`, `embed`, `vector-db`, `ingest-job`.
- Pin model versions into image labels for reproducibility.
- Use a minimal base (e.g., `python:3.12-slim`) for the API; CUDA-enabled images for inference.

### 2.2 Replace Ollama with a production inference server
Ollama is excellent for local development but is single-process and not optimized for concurrent traffic.

| Option | Best for | Notes |
| ------ | -------- | ----- |
| **vLLM** | High throughput | PagedAttention, continuous batching, OpenAI-compatible API. Recommended. |
| **TGI** (Text Generation Inference) | HuggingFace ecosystem | Good streaming + telemetry. |
| **llama.cpp server** | CPU-only / edge | Drop-in for small models. |

For embeddings, use **TEI** (Text Embeddings Inference) or vLLM running a compact embedding model. Run on GPU when QPS justifies it; CPU is fine up to a few QPS.

### 2.3 GPU sizing
- `llama3.2:3b`: fits in 4–6 GB VRAM (FP16). One T4/L4 handles tens of concurrent streams.
- Larger upgrades (Llama 3 8B, Qwen 2.5 7B) need 16–24 GB. Quantize (AWQ / GPTQ / GGUF Q4_K_M) to fit smaller cards.

---

## 3. Vector database upgrades

For >10k chunks or multi-tenant usage, ChromaDB's embedded SQLite backend becomes a bottleneck.

| Database | Why pick it |
| -------- | ----------- |
| **Qdrant** | Filtered HNSW, payload indexes, gRPC API, easy clustering. |
| **pgvector** | Already running Postgres? One service to operate; excellent for hybrid SQL + vector queries. |
| **Weaviate** | Built-in modules and hybrid search (BM25 + vector). |
| **Milvus** | Best when you cross 100M+ vectors. |

**Migration is mostly trivial**: replace `vector_store.VectorStore` with a thin adapter over the new client and re-run ingestion.

---

## 4. Retrieval quality

The v1 system uses cosine similarity with a single embedding model. Production-grade quality usually requires:

1. **Hybrid retrieval** — combine BM25 (keyword) with dense vectors and fuse with Reciprocal Rank Fusion. Catches exact-name and rare-term queries the embedding model misses.
2. **Re-ranking** — pass the top-50 candidates through a cross-encoder (e.g., `bge-reranker-large`) and keep top-5. Big quality jump for ~50 ms extra latency.
3. **Better chunking** — semantic / recursive chunking informed by document structure (headings, paragraphs). Per-section chunks improve precision for long Wikipedia articles.
4. **Query rewriting** — use the LLM to expand short queries into 2–3 paraphrases before retrieval (HyDE). Helps recall on conversational queries.
5. **Metadata-rich filters** — store `era`, `country`, `category` so the router can be even more selective ("famous place in Turkey" → `country=Turkey`).

---

## 5. Ingestion pipeline

Move ingestion from a one-off script to a **scheduled job** with the following properties:

- Triggered by cron (weekly) or by a webhook when entities are added.
- Stores raw article text in object storage (S3) with content hashes.
- Re-embeds only changed articles (compare hash). Saves GPU time.
- Emits an event/log per entity: `{name, n_chunks, duration_ms, status}`.
- Idempotent + resumable on failure.
- Observability: ingestion success rate, time-to-fresh per entity.

For an order of magnitude more entities, parallelize fetch + embed with a queue (e.g., Redis Streams, SQS).

---

## 6. Guardrails and safety

The current system has two basic guardrails:
1. Strict prompt: *"answer only from context, else 'I don't know.'"*
2. Distance threshold to skip the LLM entirely on out-of-scope queries.

In production, layer additional defenses:

- **Output validator** — a second pass (small LLM or regex/JSON-schema) that flags answers containing claims not present in the retrieved chunks.
- **PII / toxicity filter** — Llama Guard or a small classifier.
- **Prompt-injection mitigation** — strip user-provided instructions before injecting context; treat retrieved Wikipedia text as data, not instructions.
- **Rate limiting** at the API edge (per IP / per user).
- **Audit log** of every (query, route, retrieved IDs, answer) tuple for review.

---

## 7. Observability

| Signal | Tool | Why |
| ------ | ---- | --- |
| Request latency, error rate | Prometheus + Grafana | SLOs (e.g. p95 first-token < 2s). |
| LLM token usage | Custom counters | Cost / capacity planning. |
| Retrieval quality | Offline eval set + nightly job | Catch regressions when corpus or model changes. |
| Tracing | OpenTelemetry | End-to-end span: route → embed → search → generate. |
| Logs | Structured JSON to ELK / Loki | Debug specific bad answers. |

---

## 8. Evaluation harness

Build a small offline eval set (50–200 query/answer pairs) and run it on every model or prompt change. Track:
- **Faithfulness** — does the answer use only retrieved context? (Use RAGAS or an LLM-as-judge.)
- **Answer relevancy** — does the answer address the question?
- **Context precision/recall** — are the right chunks retrieved?
- **"I don't know" precision/recall** — does the system correctly abstain?

Block deployment if any metric regresses beyond a threshold.

---

## 9. Caching

- **Semantic cache** — hash normalized queries; if a near-identical embedding exists, return the prior answer. Cuts cost on repeated questions.
- **Embedding cache** — embedding for the same input chunk is computed once.
- **Static page cache** for landing UI / docs.

Be careful with caching grounded answers: invalidate when the underlying corpus changes.

---

## 10. Multi-tenancy and access control

- Add `tenant_id` to every chunk and to the chat sessions table.
- Enforce row-level filters at the vector DB (Qdrant payload filter, pgvector `WHERE`).
- AuthN via OIDC, AuthZ via JWT scopes.
- Per-tenant rate limits and quotas.

---

## 11. Cost & latency tradeoffs

| Lever | Effect | Cost trade |
| ----- | ------ | ---------- |
| Larger LLM (8B → 70B) | Better answers, especially comparisons | 5–10× GPU cost, 2–4× latency |
| Bigger top-k | Better recall | More tokens in prompt → higher cost & latency |
| Re-ranking | Big precision win | +50–150 ms / call |
| Hybrid retrieval | Better recall on rare terms | +1 service to operate |
| Quantization (Q4) | 2–4× smaller VRAM | Slight quality drop |
| Smaller embedding model | Cheaper ingest | Possibly worse retrieval |

In practice, the highest-leverage changes are: **hybrid retrieval + cross-encoder re-ranking + structured chunking**. The LLM upgrade matters less than retrieval quality for grounded QA.

---

## 12. Operational checklist before going live

- [ ] Reproducible CI build of every container, pinned model digests.
- [ ] Healthchecks for LLM, embed, vector DB, API.
- [ ] Backups for vector DB and Postgres.
- [ ] Disaster-recovery runbook (warm standby of vector DB; replay ingestion).
- [ ] Load test at expected peak QPS.
- [ ] Eval suite green; manual spot-check of acceptance queries.
- [ ] Privacy review of logs (do they store user queries?).
- [ ] Incident response on-call rotation.

---

## 13. Migration plan from this v1

1. Extract `embed()` + `query()` into a thin adapter interface so the implementation can swap.
2. Replace Chroma backend with Qdrant; re-run ingestion against the new endpoint.
3. Replace Ollama with vLLM behind an OpenAI-compatible API; only the base URL changes.
4. Add re-ranker after retrieval; gate behind an eval-set improvement.
5. Wrap the API in FastAPI; keep Streamlit as the demo UI, expose REST for other clients.
6. Add OpenTelemetry, Prometheus exporters, and structured logs.
7. Wire ingestion into a scheduled job with hash-based incremental updates.

Each step is independently shippable, which keeps risk low while incrementally hardening the system.
