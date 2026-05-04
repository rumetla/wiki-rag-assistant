"""Retrieval-Augmented Generation engine.

Pipeline:
1. ``route(query)`` decides if the user is asking about a person, a place, or
   both, using the canonical entity names plus light keyword heuristics.
2. ``retrieve(query, route)`` runs metadata-filtered similarity search.
3. A **grounding guard** rejects queries whose top results are too distant
   (cosine distance above a threshold), short-circuiting straight to
   ``"I don't know"`` so we never hallucinate on out-of-scope questions like
   *"Who is the president of Mars?"*.
4. ``generate(...)`` calls the local Ollama LLM with a strict, grounded
   prompt. Streaming is exposed so the UI can render token-by-token.

All HTTP calls hit ``localhost`` only, per project rules.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

import requests

from entities import (
    PERSON_KEYWORDS,
    PLACE_KEYWORDS,
    match_entities,
)
from vector_store import VectorStore


OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "llama3.2:3b")

DISTANCE_THRESHOLD: float = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "0.7"))
TOP_K_SINGLE: int = 5
TOP_K_BOTH: int = 8
TOP_K_NAMED: int = 6

IDK_RESPONSE: str = "I don't know."

SYSTEM_PROMPT: str = (
    "You are a careful assistant that answers questions about famous people "
    "and places using ONLY the provided context. Follow these rules strictly:\n"
    "1. If the answer is not present in the context, reply exactly: I don't know.\n"
    "2. Do not use any outside knowledge.\n"
    "3. Be concise and factual. Cite the source numbers like [1], [2] when useful.\n"
    "4. If the question compares two entities, answer for both using only the context."
)


RouteType = str  # "person" | "place" | "both"


@dataclass
class Source:
    """A retrieved chunk attached to its provenance metadata."""

    name: str
    type: str
    url: str
    text: str
    distance: float


@dataclass
class RagResult:
    """Container for a single end-to-end RAG run."""

    answer: str
    sources: list[Source]
    route: RouteType
    grounded: bool


def route(query: str) -> RouteType:
    """Classify the query as ``"person"``, ``"place"``, or ``"both"``.

    Priority order (matches `.cursorrules` "rule-based approaches are
    acceptable"):
    1. Direct entity-name hits in the query.
    2. Keyword hints (``where`` -> place, ``who`` -> person, etc.).
    3. Fall back to ``"both"`` so we don't accidentally exclude relevant docs.
    """
    found = match_entities(query)
    has_person = bool(found["person"])
    has_place = bool(found["place"])

    if has_person and has_place:
        return "both"
    if has_person:
        return "person"
    if has_place:
        return "place"

    q = query.lower()
    person_hits = sum(1 for kw in PERSON_KEYWORDS if kw in q)
    place_hits = sum(1 for kw in PLACE_KEYWORDS if kw in q)

    if person_hits > place_hits:
        return "person"
    if place_hits > person_hits:
        return "place"
    return "both"


def retrieve(
    store: VectorStore,
    query: str,
    route_type: RouteType,
) -> list[Source]:
    """Run metadata-filtered similarity search and return ``Source`` objects.

    Filter precedence:
    1. **Named-entity filter** — if the router detected one or more specific
       entities in the query (e.g. *"Albert Einstein"*, *"Eiffel Tower"*),
       restrict retrieval to chunks whose ``name`` is in that set. This is
       the strongest signal we have and bypasses any embedder ranking
       weakness for short generic biographical queries.
    2. **Type filter** — fall back to filtering by ``type`` (``person`` /
       ``place``) when only the topic is known.
    3. **No filter** — for ``"both"`` queries with no named entities.
    """
    found = match_entities(query)
    named = found["person"] + found["place"]

    if named:
        k = TOP_K_NAMED if len(named) >= 2 else TOP_K_SINGLE
        if len(named) == 1:
            where: dict[str, Any] = {"name": named[0]}
        else:
            where = {"name": {"$in": named}}
        results = store.query(query, k=k, where=where)
    elif route_type == "both":
        results = store.query(query, k=TOP_K_BOTH, where=None)
    else:
        results = store.query(query, k=TOP_K_SINGLE, where={"type": route_type})

    sources: list[Source] = []
    for r in results:
        meta = r["metadata"] or {}
        sources.append(
            Source(
                name=str(meta.get("name", "unknown")),
                type=str(meta.get("type", "unknown")),
                url=str(meta.get("url", "")),
                text=r["text"],
                distance=r["distance"],
            )
        )
    return sources


def is_grounded(sources: list[Source]) -> bool:
    """True iff at least one retrieved chunk is closer than the threshold."""
    if not sources:
        return False
    return min(s.distance for s in sources) <= DISTANCE_THRESHOLD


def build_prompt(query: str, sources: list[Source]) -> str:
    """Assemble the final user-side prompt from query + numbered sources."""
    if not sources:
        context_block = "(no context retrieved)"
    else:
        blocks = []
        for i, s in enumerate(sources, start=1):
            blocks.append(f"[{i}] {s.name} ({s.type}):\n{s.text}")
        context_block = "\n\n".join(blocks)

    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above. "
        "If the answer is not contained in the context, reply exactly: I don't know."
    )


def _ollama_generate_stream(prompt: str, model: str = LLM_MODEL) -> Iterator[str]:
    """Yield tokens from Ollama's ``/api/generate`` streaming endpoint."""
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {"temperature": 0.2},
    }
    try:
        with requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama generate failed ({resp.status_code}): {resp.text[:200]}"
                )
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = obj.get("response", "")
                if token:
                    yield token
                if obj.get("done"):
                    break
    except requests.RequestException as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. Is `ollama serve` running? ({e})"
        ) from e


def answer_stream(
    store: VectorStore,
    query: str,
) -> tuple[Iterator[str], list[Source], RouteType, bool]:
    """Stream an answer for ``query``.

    Returns a 4-tuple ``(token_iter, sources, route, grounded)``. When the
    grounding guard rejects the query, ``token_iter`` yields the canonical
    ``"I don't know."`` response without invoking the LLM.
    """
    route_type = route(query)
    sources = retrieve(store, query, route_type)
    grounded = is_grounded(sources)

    if not grounded:
        def _idk() -> Iterator[str]:
            yield IDK_RESPONSE
        return _idk(), sources, route_type, grounded

    prompt = build_prompt(query, sources)
    return _ollama_generate_stream(prompt), sources, route_type, grounded


def answer(store: VectorStore, query: str) -> RagResult:
    """Convenience wrapper that materializes the streamed answer."""
    tokens, sources, route_type, grounded = answer_stream(store, query)
    text = "".join(tokens).strip()
    if not text:
        text = IDK_RESPONSE
    return RagResult(answer=text, sources=sources, route=route_type, grounded=grounded)
