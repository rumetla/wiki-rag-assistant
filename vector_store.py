"""Thin ChromaDB wrapper backed by local Ollama embeddings.

Single-collection design (Option B from the PRD): one Chroma collection holds
both people and places, distinguished by a ``type`` metadata field. This keeps
mixed queries (e.g. "Compare Einstein and the Eiffel Tower") trivial — no
union logic required at retrieval time.

Embeddings are computed locally by Ollama's ``mxbai-embed-large`` model
(1024-dim) via the HTTP API, so no external service is contacted.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import chromadb
import requests
from chromadb.config import Settings


OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "mxbai-embed-large")
CHROMA_PATH: str = os.environ.get("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME: str = "wikipedia_rag"
EMBED_PARALLELISM: int = int(os.environ.get("EMBED_PARALLELISM", "6"))


class OllamaEmbeddingError(RuntimeError):
    """Raised when the local Ollama embedding endpoint fails."""


def _embed_one(text: str, timeout: float = 60.0) -> list[float]:
    """Embed a single text via Ollama's ``/api/embeddings`` endpoint.

    ``mxbai-embed-large`` (the default) does not require any task-prefix; it
    produces high-quality embeddings directly from raw text.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise OllamaEmbeddingError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running? ({e})"
        ) from e

    if resp.status_code != 200:
        raise OllamaEmbeddingError(
            f"Ollama embeddings call failed ({resp.status_code}): {resp.text[:200]}"
        )

    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        raise OllamaEmbeddingError(
            f"Ollama returned no embedding. Did you pull '{EMBED_MODEL}'? Response: {data}"
        )
    return embedding


def embed(texts: Iterable[str]) -> list[list[float]]:
    """Embed many texts in parallel.

    Ollama's embedding API takes one prompt per call, but the server happily
    serves concurrent requests. We use a small thread pool to overlap the
    request latency, which roughly cuts ingest time by ``EMBED_PARALLELISM``
    on CPU machines.
    """
    text_list = list(texts)
    if not text_list:
        return []
    workers = max(1, min(EMBED_PARALLELISM, len(text_list)))
    if workers == 1:
        return [_embed_one(t) for t in text_list]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_embed_one, text_list))


class VectorStore:
    """Persistent ChromaDB-backed store with local Ollama embeddings."""

    def __init__(self, path: str = CHROMA_PATH, collection: str = COLLECTION_NAME) -> None:
        self.path = path
        self.collection_name = collection
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection (used by ``ingest.py --reset``)."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def has_entity(self, name: str) -> bool:
        """Return True if any chunks for ``name`` already exist (idempotent ingest)."""
        try:
            res = self.collection.get(where={"name": name}, limit=1)
        except Exception:
            return False
        return bool(res.get("ids"))

    def add_chunks(
        self,
        name: str,
        chunks: list[str],
        base_metadata: dict[str, Any],
    ) -> int:
        """Embed and insert chunks for a single entity.

        Each chunk's ID is ``"{name}::{chunk_idx}"`` so re-ingesting the same
        entity overwrites cleanly via Chroma's upsert semantics.
        """
        if not chunks:
            return 0

        ids = [f"{name}::{i}" for i in range(len(chunks))]
        metadatas = [
            {**base_metadata, "chunk_idx": i, "name": name} for i in range(len(chunks))
        ]
        embeddings = embed(chunks)

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query(
        self,
        text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a similarity query, optionally filtered by metadata.

        Returns a list of dicts with ``text``, ``metadata``, and ``distance``
        (cosine distance — lower is better). Returns an empty list if the
        store is empty.
        """
        if self.count() == 0:
            return []

        embedding = _embed_one(text)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )

        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        return [
            {"text": d, "metadata": m, "distance": float(dist)}
            for d, m, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        """Total number of chunks stored."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def count_by_type(self) -> dict[str, int]:
        """Counts grouped by ``type`` metadata (``person`` / ``place``)."""
        counts: dict[str, int] = {"person": 0, "place": 0}
        for t in ("person", "place"):
            try:
                res = self.collection.get(where={"type": t})
                counts[t] = len(res.get("ids") or [])
            except Exception:
                counts[t] = 0
        return counts
