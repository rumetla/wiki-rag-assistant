# Product Requirements Document — Local Wikipedia RAG Assistant

**Course:** BLG483E — Project 3
**Status:** v1 (educational deliverable)
**Author:** Project owner
**Last updated:** 2026-05-04

---

## 1. Problem statement

Users want a ChatGPT-style assistant that can answer questions about famous **people** and **places** without sending any data outside their machine. The system must be reproducible by an instructor with only the README, run on consumer hardware, and avoid all paid or external APIs.

## 2. Goals

- Provide accurate, grounded answers about a fixed catalog of entities (≥20 people, ≥20 places) sourced from Wikipedia.
- Run end-to-end on a single laptop using only open, local components.
- Refuse to answer when the retrieved context does not contain the answer (no hallucinated facts).
- Make the architecture and trade-offs easy to inspect and explain.

## 3. Non-goals (v1)

- Real-time web crawling or live Wikipedia updates.
- Long-term chat memory across sessions.
- Multi-user authentication / cloud deployment.
- Re-ranking, hybrid (BM25 + dense) retrieval — listed as future work in `recommendation.md`.

## 4. Target users

- Course instructor evaluating the project.
- Students / engineers studying local RAG patterns.
- Anyone curious to chat with their laptop about famous people and places offline.

## 5. Functional requirements

### FR-1 Ingestion
- Fetch the full Wikipedia article for each entity in `entities.py`.
- Clean boilerplate (References, External links, etc.).
- Chunk into ~500-char windows with ~100-char overlap, snapped to sentence boundaries.
- Store each chunk in ChromaDB with metadata: `type` (`person` | `place`), `name`, `title`, `url`, `chunk_idx`.
- Idempotent re-runs (`python ingest.py`); destructive wipe via `--reset`.

### FR-2 Routing
- Given a query, classify as `person`, `place`, or `both`.
- Use canonical entity name matching first, then keyword heuristics.
- Default to `both` on ambiguity to avoid false negatives.

### FR-3 Retrieval
- Embed the query with `mxbai-embed-large` via Ollama.
- Filter precedence: named-entity (`name in [...]`) → type → none.
- Run a cosine-distance similarity search filtered accordingly.
- Top-k = 5 for single-type/single-entity queries, 6 for multi-entity queries, 8 for `both`.

### FR-4 Grounding guard
- If no retrieved chunk has cosine distance ≤ `RAG_DISTANCE_THRESHOLD` (default 0.55), the system returns the literal string `I don't know.` **without invoking the LLM**.
- This is the primary defense against hallucination on out-of-scope queries.

### FR-5 Generation
- Build a strict prompt: system message ("answer only from context, else 'I don't know.'") + numbered context block + user question.
- Stream tokens from Ollama `llama3.2:3b`.
- Display the answer plus an expandable list of retrieved sources with name, type, distance, and Wikipedia URL.

### FR-6 Chat UI
- Streamlit app at `http://localhost:8501` with:
  - Chat history within a session.
  - Streamed assistant response.
  - "Retrieved sources" expander.
  - Sidebar showing chunk counts per type, model names, and a "Clear chat" button.

## 6. Acceptance criteria (mapped to PDF Example Questions)

The system must produce a sensible, grounded answer (or "I don't know.") for each query below:

| # | Query | Expected behavior |
| - | ----- | ----------------- |
| 1 | *Who was Albert Einstein and what is he known for?* | Person route → Einstein chunks → grounded answer about relativity, Nobel Prize. |
| 2 | *What did Marie Curie discover?* | Person route → Curie chunks → polonium, radium. |
| 3 | *Why is Nikola Tesla famous?* | Person route → Tesla chunks → AC current, electromagnetism. |
| 4 | *Compare Lionel Messi and Cristiano Ronaldo.* | Person route → both entity hits → comparative answer using only retrieved context. |
| 5 | *What is Frida Kahlo known for?* | Person route → Kahlo chunks → Mexican painter, self-portraits. |
| 6 | *Where is the Eiffel Tower located?* | Place route → Paris, France. |
| 7 | *Why is the Great Wall of China important?* | Place route → fortifications, historical significance. |
| 8 | *What is Machu Picchu?* | Place route → Inca citadel in Peru. |
| 9 | *What was the Colosseum used for?* | Place route → gladiatorial contests, public spectacles. |
| 10 | *Where is Mount Everest?* | Place route → Himalayas, Nepal/China border. |
| 11 | *Which famous place is located in Turkey?* | Mixed/place route → Hagia Sophia (and/or Topkapı Palace). |
| 12 | *Which person is associated with electricity?* | Mixed/person route → Tesla. |
| 13 | *Compare Albert Einstein and Nikola Tesla.* | Both-entity person route → comparative answer. |
| 14 | *Compare the Eiffel Tower and the Statue of Liberty.* | Both-entity place route → comparative answer. |
| 15 | *Who is the president of Mars?* | Grounding guard → `I don't know.` |
| 16 | *Tell me about a random unknown person John Doe.* | Grounding guard → `I don't know.` |

## 7. Non-functional requirements

| Category | Requirement |
| -------- | ----------- |
| **Locality** | All inference and storage must happen on `localhost`. No external HTTP calls except to the Wikipedia API during ingestion. |
| **Reproducibility** | A user following only `README.md` must be able to install and run the system. |
| **Latency** | First-token latency under ~5 s on a typical laptop with `llama3.2:3b`. Embedding ingestion under ~10 minutes for 40 entities. |
| **Resilience** | Graceful failure when Ollama is down (clear error message in UI), and idempotent re-ingestion. |
| **Hallucination control** | Distance-threshold guard + strict system prompt. |

## 8. Architecture

| Layer | Choice | Rationale |
| ----- | ------ | --------- |
| LLM | Ollama `llama3.2:3b` | Small, fast, capable; runs on CPU/Apple Silicon. |
| Embeddings | Ollama `mxbai-embed-large` | Local, 1024-dim, strong on short generic queries, no task prefixes required. |
| Vector DB | ChromaDB (persistent) | Embedded, file-backed, no server. |
| Vector store layout | **Option B**: one collection with `type` metadata | Simpler than two stores; supports mixed queries naturally; aligns with `.cursorrules`. |
| Chunking | Native ~500/100, sentence-aware | `.cursorrules` forbids LangChain splitters. |
| Routing | Rule-based keyword + entity match | Cheap, deterministic, easy to demo. |
| UI | Streamlit | Fastest path to a polished demo. |

## 9. Constraints

- No paid LLM/embedding APIs (per PDF + `.cursorrules`).
- No LangChain/LlamaIndex for core RAG logic (per `.cursorrules`).
- Mandatory documentation: `README.md`, `product_prd.md`, `recommendation.md`.

## 10. Risks & mitigations

| Risk | Mitigation |
| ---- | ---------- |
| Wikipedia article disambiguation ambiguity | Fall back to first option then auto-suggest. |
| Slow ingestion due to per-prompt Ollama embeddings | Idempotent skipping; one-time cost. |
| Hallucination on out-of-scope queries | Distance threshold guard + strict system prompt. |
| Streamlit cache staleness after re-ingest | `st.cache_resource` only caches the store handle; ChromaDB persists changes. |

## 11. Future work

- Streaming citations highlighted inline.
- Cross-session chat history.
- Re-ranking with a small cross-encoder for better top-k quality.
- Hybrid retrieval (BM25 + dense).
- Latency dashboard.
- Comparative evaluation across Phi3 / Mistral / Llama3.2 (see PDF "Optional Extensions").

## 12. Success metrics

- 100% of acceptance-criteria queries yield a sensible answer or the canonical "I don't know." response.
- A new user can complete setup and ask the first question in under 15 minutes following `README.md`.
- Zero external HTTP calls during chat (verified via local network monitoring).
