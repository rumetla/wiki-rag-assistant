# Demo Script — Local Wikipedia RAG Assistant

A 5-minute walkthrough of the system. Each section has copy-pasteable PowerShell commands plus talking points.

---

## 0. Pre-flight (run BEFORE recording)

Verify everything is ready in one command:

```powershell
Write-Host "--- Ollama ---"; (Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 5).Content; Write-Host "--- Models ---"; ollama list; Write-Host "--- Python deps ---"; python -c "import chromadb, requests, streamlit; print('OK')"
```

You should see `Ollama is running`, both `llama3.2:3b` and `mxbai-embed-large` listed, and `OK`.

If Ollama isn't running:

```powershell
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
```

Wipe any old index so ingestion is a real first run:

```powershell
Remove-Item -Path "chroma_db" -Recurse -Force -ErrorAction SilentlyContinue
```

---

## 1. System overview (~45 s)

**Show:** project folder tree, `.cursorrules`, the architecture section in `readme.md`.

```powershell
Get-ChildItem -Force | Format-Table Name, Length, LastWriteTime -AutoSize
```

**Talking points:**

- "This is a fully local RAG assistant. Nothing leaves the laptop — no API keys, no cloud."
- "Goal: answer questions about famous people and places using only Wikipedia content I've indexed myself."
- Five core modules:
  - `ingest.py` — Wikipedia fetch + native chunker
  - `vector_store.py` — ChromaDB wrapper + Ollama embeddings
  - `rag_engine.py` — query router + retrieval + grounded prompt
  - `app.py` — Streamlit chat UI
  - `entities.py` — catalog of 20 people + 20 places
- Three docs: `readme.md`, `product_prd.md`, `recommendation.md`.

---

## 2. Live ingestion (~60 s — pause recording during the slow part)

**Command (start recording, run, then pause):**

```powershell
python ingest.py --reset
```

You'll see per-entity progress like:

```
[person] Albert Einstein
  -> fetching 'Albert Einstein' ... 47 chunks
     embedded 47 chunks in 17.8s
[person] Marie Curie
  -> fetching 'Marie Curie' ... 46 chunks
     embedded 46 chunks in 17.9s
...
Done in 709s. Added 1743 new chunks.
Collection now holds: people=919, places=824
```

**While the first 1–2 entities are ingesting, narrate:**

- "For each entity I hit the Wikipedia MediaWiki API with a custom User-Agent. Wikipedia rejects the default `requests` UA, so I added one."
- "I clean Wikipedia boilerplate (References, External links), cap each article at 30,000 characters to keep ingest tractable, then run a hand-written sentence-aware chunker — 900-char windows with 150-char overlap."
- "Each chunk gets embedded by **`mxbai-embed-large`** running locally in Ollama — 1024-dimensional vectors. I parallelize 6 embedding requests at a time, which cuts ingest from ~50 minutes to ~12."
- "Storage is one ChromaDB collection with `type` and `name` metadata, so I can filter at query time."

**Now pause the recording.** Wait ~12 minutes for ingestion to finish (~1700 chunks). Resume when you see "Done in ... Added ... new chunks."

**Optional fast path for the demo:** if you don't want to pause, run quick mode (~3 minutes, smaller article extracts):

```powershell
python ingest.py --reset --quick
```

---

## 3. Live Q&A (~120 s)

**Launch the UI:**

```powershell
streamlit run app.py
```

Open `http://localhost:8501` in the browser.

Walk through these queries in order, expanding the "Retrieved sources" panel after each so the audience sees grounding in action:

### Person query
```
Who was Albert Einstein and what is he known for?
```
Expect: top source = `Albert Einstein`, distance ~0.21, answer about relativity / Nobel Prize. Show sources panel.

### Place query
```
Where is the Eiffel Tower located?
```
Expect: top source = `Eiffel Tower`, "Champ de Mars in Paris, France."

### Mixed (no entity name)
```
Which famous place is located in Turkey?
```
Expect: route=`place`, retrieves Hagia Sophia and/or Topkapı Palace.

### Comparison
```
Compare Lionel Messi and Cristiano Ronaldo.
```
Expect: route=`person`, both entities retrieved, comparative answer.

### Failure case (the important one)
```
Who is the president of Mars?
```
Expect: `I don't know.` with no LLM call — the grounding guard blocks it.

### Second failure case
```
Tell me about a random unknown person John Doe.
```
Expect: `I don't know.`

**Talking points during the queries:**
- "Notice the sidebar — it shows 919 person chunks and 824 place chunks."
- "Each answer prints `route` and `grounded` flags so you can see what the router decided."
- "For named entities I do a hard metadata filter on `name` — so Einstein's chunks always come back when 'Einstein' appears in the query, regardless of embedder ranking."

---

## 4. Technical decisions (~45 s)

Five points, each one sentence:

1. **Embedder: `mxbai-embed-large`** — 1024-dim, no task prefixes required, much stronger top-1 ranking than `nomic-embed-text` on short generic queries like *"Who was X?"*.
2. **LLM: `llama3.2:3b`** — fits in CPU RAM, fast first-token, good enough at extractive answers from short context.
3. **Vector store: single ChromaDB collection with `type` + `name` metadata** (Option B). Simpler than two stores; lets me filter mixed queries naturally.
4. **Chunker: native, sentence-aware, 900/150** — `.cursorrules` forbids LangChain splitters. The custom chunker keeps sentences intact and overlaps to preserve context across boundaries.
5. **Routing: rule-based with last-token entity match** — *"Where is Everest?"* alone matches *Mount Everest*. When an entity is named, retrieval is hard-filtered by `name` rather than relying on embedding similarity.

---

## 5. Tradeoffs & limitations (~30 s)

- **Embeddings on CPU are slow.** ~0.4 s per chunk → ~12 min full ingest. Acceptable as a one-time cost; production would batch on GPU.
- **No re-ranking.** Top-k is final. Cross-encoder re-ranking would help on ambiguous queries.
- **Fixed corpus.** No live Wikipedia updates — articles are snapshotted at ingest time.
- **Strict refusal can be too strict.** A factual question whose answer happens to sit in a chunk just past the cosine-distance threshold gets rejected. Tunable via `RAG_DISTANCE_THRESHOLD`.
- **Comparison queries depend on retrieval recall** — both entities must make it into the top-k.
- **No long-term chat memory** — each turn is independent.

---

## 6. Possible improvements (~30 s)

Listed in priority order:

1. **Hybrid retrieval (BM25 + dense)** — catches exact-name and rare-term queries the embedder ranks weakly.
2. **Cross-encoder re-ranking** of top-50 → top-5. Big precision win for ~50 ms.
3. **Streaming citation highlighting** — visually link each sentence in the answer back to the source chunk.
4. **Larger LLM (Llama 3 8B / Qwen 2.5 7B)** for stronger comparison and reasoning answers.
5. **Re-embed only changed articles** via content hashing, so refreshes are cheap.
6. **Eval harness** — RAGAS-style faithfulness/relevancy/abstention metrics gating each prompt or model change.
7. **Multi-tenant deployment** with Qdrant + FastAPI + auth (covered in `recommendation.md`).

---

## 7. Wrap-up (~15 s)

- "Code, README, PRD, and recommendation doc are all in the repo."
- "Everything ran on this laptop. No API keys, no cloud."

**Stop the recording.**

---

## Quick-reference: every command in one place

```powershell
# Pre-flight
(Invoke-WebRequest -Uri "http://localhost:11434" -UseBasicParsing -TimeoutSec 5).Content
ollama list

# Reset DB
Remove-Item -Path "chroma_db" -Recurse -Force -ErrorAction SilentlyContinue

# Full ingest (pause demo here)
python ingest.py --reset

# Quick ingest (alternative, ~3 min)
python ingest.py --reset --quick

# Launch UI
streamlit run app.py
```

## If something breaks live

| Symptom | Fix |
| ------- | --- |
| `Could not reach Ollama` | `Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden` |
| Streamlit shows "Index is empty" | Run `python ingest.py` first |
| `Ollama returned no embedding` | `ollama pull mxbai-embed-large` |
| Wikipedia fetch fails repeatedly | Wait 30 s and re-run `python ingest.py` (idempotent — skips done entities) |
| `Internal error: Error finding id` from Chroma | Wipe DB and re-ingest: `Remove-Item chroma_db -Recurse -Force; python ingest.py --reset` |
