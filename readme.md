# Local Wikipedia RAG Assistant

github link: github.com/rumetla/wiki-rag-assistant

A ChatGPT-style question-answering system about famous **people** and **places**, running 100% locally on your laptop. The system ingests Wikipedia articles, embeds them with a local model, stores them in ChromaDB, and uses a local Ollama LLM to generate grounded answers. If the answer isn't in the retrieved context, it replies **"I don't know."**

Built for BLG483E - Project 3.

---

## Features

- 100% local — no external APIs, no cloud calls.
- Ingests 20 people + 20 places (configurable) from Wikipedia.
- Native, hand-written chunker (no LangChain/LlamaIndex for core logic).
- Single ChromaDB collection with `type` and `name` metadata.
- Rule-based router that detects entity names and applies the strongest available filter (entity name → type → none).
- Grounding guard: low-similarity queries short-circuit to "I don't know" without invoking the LLM.
- Streamlit chat UI with streamed responses and a source viewer.

---

## Architecture

```
Wikipedia API
     |
     v
ingest.py  --(clean + chunk)-->  vector_store.py  --(ollama embed)-->  ChromaDB (./chroma_db)
                                                                            ^
                                                                            |
                                                  rag_engine.py  <----------+
                                                  (route + retrieve + ground)
                                                            |
                                                  ollama llama3.2:3b
                                                            |
                                                            v
                                                          app.py (Streamlit UI)
```

| File | Role |
| ---- | ---- |
| `entities.py`     | Catalog of 20 people + 20 places, router keyword lists |
| `vector_store.py` | ChromaDB wrapper + Ollama `mxbai-embed-large` embeddings |
| `ingest.py`       | Wikipedia fetch + native chunker + idempotent storage |
| `rag_engine.py`   | Router, retrieval, grounding guard, Ollama generation |
| `app.py`          | Streamlit chat UI |

---

## Setup (Windows / PowerShell)

### 1. Install Ollama

```powershell
winget install Ollama.Ollama
```

(Or download the installer from <https://ollama.com/download/windows>.)

### 2. Start the Ollama service

The Windows installer registers Ollama as a background app — it usually starts automatically. If `Get-Process ollama` shows nothing, start it manually:

```powershell
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
```

Verify it's reachable: `Invoke-WebRequest http://localhost:11434` should return `Ollama is running`.

### 3. Pull the local models

```powershell
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
```

### 4. Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(If activation is blocked: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.)

### 5. Ingest Wikipedia data (one-time, ~10–15 min)

```powershell
python ingest.py
```

Re-runs are idempotent (entities already indexed are skipped). To wipe and re-ingest:

```powershell
python ingest.py --reset
```

For a fast smoke-test ingest (~3–4 min, smaller article extracts):

```powershell
python ingest.py --reset --quick
```

### 6. Launch the chat UI

```powershell
python -m streamlit run app.py
```

Open the URL Streamlit prints (default: `http://localhost:8501`).

### Setup on Linux / macOS

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
ollama pull mxbai-embed-large
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
streamlit run app.py
```

---

## Example queries

### People
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *What is Frida Kahlo known for?*

### Places
- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

### Mixed
- *Which famous place is located in Turkey?*  → Hagia Sophia
- *Which person is associated with electricity?*  → Tesla
- *Compare Albert Einstein and Nikola Tesla.*
- *Compare the Eiffel Tower and the Statue of Liberty.*

### Failure cases (must say "I don't know")
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe.*

---

## Configuration

Override defaults via environment variables:

| Variable | Default | Purpose |
| -------- | ------- | ------- |
| `OLLAMA_URL` | `http://localhost:11434` | Local Ollama endpoint |
| `LLM_MODEL` | `llama3.2:3b` | Generation model |
| `EMBED_MODEL` | `mxbai-embed-large` | Embedding model (1024-dim) |
| `CHROMA_PATH` | `./chroma_db` | Persistent vector-store directory |
| `RAG_DISTANCE_THRESHOLD` | `0.7` | Max cosine distance still considered "grounded" |
| `EMBED_PARALLELISM` | `6` | Concurrent embedding requests during ingest |

---

## Design choices

- **Single ChromaDB collection with `type` + `name` metadata** instead of two separate stores. Simpler to maintain and lets mixed queries (e.g. *"Compare Einstein and the Eiffel Tower"*) work without union logic.
- **Embedder: `mxbai-embed-large`** (1024-dim) — a strong general-purpose local embedder that does not require task prefixes and ranks the right entity at top-1 reliably for short generic queries.
- **Native chunking** at ~900 chars with ~150 char overlap, snapped to sentence boundaries. Handles long Wikipedia articles without splitting mid-word.
- **Rule-based router** with substring + last-token entity matching ("Einstein" alone matches "Albert Einstein"). When an entity is named, retrieval is **hard-filtered by `name`**, which guarantees correct chunks regardless of embedder ranking.
- **Grounding guard** rejects queries whose nearest chunk distance exceeds `RAG_DISTANCE_THRESHOLD`. Prevents hallucination on out-of-scope questions like *"Who is the president of Mars?"*.

See [`product_prd.md`](product_prd.md) for full requirements and [`recommendation.md`](recommendation.md) for production-deployment guidance.

---

## Troubleshooting

- **`Could not reach Ollama at http://localhost:11434`** → run `Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden` (PowerShell) or `ollama serve` (bash).
- **`Ollama returned no embedding`** → `ollama pull mxbai-embed-large`.
- **Index is empty in the UI** → run `python ingest.py`.
- **A specific entity failed to ingest** → re-run `python ingest.py`; transient network errors are common with the Wikipedia API.

---

## License

Educational project. Wikipedia content is used under the [Creative Commons Attribution-ShareAlike License](https://creativecommons.org/licenses/by-sa/4.0/).
