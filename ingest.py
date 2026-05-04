"""Ingestion pipeline: Wikipedia -> clean -> chunk -> embed -> ChromaDB.

Run as a script:

    python ingest.py            # idempotent: skips entities already indexed
    python ingest.py --reset    # wipe the collection and re-ingest everything

Per `.cursorrules`, the chunker is hand-written (no LangChain splitters). The
``wikipedia`` package is used purely as a thin HTTP client for the Wikipedia
API — it does not perform any RAG-relevant logic.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
from typing import Any

import requests

from entities import all_entities
from vector_store import VectorStore


CHUNK_SIZE: int = 900
CHUNK_OVERLAP: int = 150
MAX_ARTICLE_CHARS: int = 30000
QUICK_MAX_ARTICLE_CHARS: int = 6000

WIKIPEDIA_API: str = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_PAGE_BASE: str = "https://en.wikipedia.org/wiki/"

USER_AGENT: str = (
    "LocalWikipediaRAG/1.0 (BLG483E educational project; "
    "contact: student@itu.edu.tr) python-requests"
)


_SECTION_BLACKLIST = (
    "References",
    "External links",
    "Further reading",
    "See also",
    "Notes",
    "Bibliography",
    "Citations",
    "Sources",
)


def clean_text(text: str) -> str:
    """Strip Wikipedia boilerplate sections and collapse whitespace."""
    cleaned_lines: list[str] = []
    skip = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("==") and stripped.endswith("=="):
            heading = stripped.strip("= ").strip()
            skip = heading in _SECTION_BLACKLIST
            continue
        if skip:
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter — good enough for chunk boundaries."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split ``text`` into overlapping ~``size``-char chunks at sentence boundaries.

    Strategy:
    - Walk sentences and accumulate them until adding the next would exceed
      ``size``. That bundle becomes one chunk.
    - For overlap, carry the tail ``overlap`` characters of the previous
      chunk into the next one. This preserves continuity across boundaries
      without splitting mid-word.
    - Sentences longer than ``size`` are hard-split into windows.
    """
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in sentences:
        if len(sentence) > size:
            _flush()
            for start in range(0, len(sentence), size - overlap):
                chunks.append(sentence[start : start + size].strip())
            continue

        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
        else:
            _flush()
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                current = f"{tail} {sentence}".strip()
            else:
                current = sentence

    _flush()
    return chunks


def _wiki_get(params: dict[str, str]) -> dict[str, Any] | None:
    """GET against Wikipedia's MediaWiki API with a proper User-Agent.

    Returns the parsed JSON body, or ``None`` on transport / parse error.
    Wikipedia's policy requires a descriptive UA; the default ``requests``
    UA gets rate-limited or 403'd, so we set ``USER_AGENT`` explicitly.
    """
    try:
        resp = requests.get(
            WIKIPEDIA_API,
            params=params,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30,
        )
    except requests.RequestException as e:
        print(f"  ! HTTP error: {e}", flush=True)
        return None
    if resp.status_code != 200:
        print(f"  ! HTTP {resp.status_code} from Wikipedia: {resp.text[:120]}", flush=True)
        return None
    try:
        return resp.json()
    except ValueError as e:
        print(f"  ! bad JSON from Wikipedia: {e}", flush=True)
        return None


def fetch_wikipedia_article(title: str) -> tuple[str, str] | None:
    """Fetch a Wikipedia article via the MediaWiki API.

    Uses ``prop=extracts&explaintext=1`` to get the plain-text body and
    follows redirects automatically. Returns ``(content, url)`` or ``None``.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
        "formatversion": "2",
    }
    data = _wiki_get(params)
    if not data:
        return None

    pages = data.get("query", {}).get("pages") or []
    if not pages:
        print(f"  ! no pages returned for '{title}'", flush=True)
        return None

    page = pages[0]
    if page.get("missing"):
        print(f"  ! page '{title}' does not exist", flush=True)
        return None

    extract = page.get("extract") or ""
    if not extract.strip():
        print(f"  ! empty extract for '{title}'", flush=True)
        return None

    final_title = page.get("title") or title
    url = WIKIPEDIA_PAGE_BASE + urllib.parse.quote(final_title.replace(" ", "_"))
    return extract, url


def ingest_entity(
    store: VectorStore,
    name: str,
    title: str,
    entity_type: str,
    skip_existing: bool = True,
    max_chars: int = MAX_ARTICLE_CHARS,
) -> int:
    """Fetch + chunk + store one entity. Returns number of chunks added."""
    if skip_existing and store.has_entity(name):
        print(f"  = '{name}' already indexed, skipping")
        return 0

    print(f"  -> fetching '{title}' ...", end=" ", flush=True)
    fetched = fetch_wikipedia_article(title)
    if fetched is None:
        return 0
    content, url = fetched

    cleaned = clean_text(content)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    chunks = chunk_text(cleaned)
    print(f"{len(chunks)} chunks", flush=True)

    if not chunks:
        return 0

    metadata: dict[str, Any] = {
        "type": entity_type,
        "title": title,
        "url": url,
    }
    started = time.time()
    n = store.add_chunks(name=name, chunks=chunks, base_metadata=metadata)
    print(f"     embedded {n} chunks in {time.time() - started:.1f}s", flush=True)
    return n


def run(reset: bool = False, quick: bool = False, limit: int | None = None) -> None:
    """Run ingestion over the entity catalog.

    ``quick`` truncates each article to ``QUICK_MAX_ARTICLE_CHARS`` (~7 chunks
    per entity, ~3 minutes total instead of ~12). ``limit`` ingests only the
    first N entities (useful for smoke tests).
    """
    store = VectorStore()
    if reset:
        print("Resetting vector store ...")
        store.reset()

    max_chars = QUICK_MAX_ARTICLE_CHARS if quick else MAX_ARTICLE_CHARS
    if quick:
        print(f"QUICK mode: capping each article at {max_chars} chars")

    started = time.time()
    total_chunks = 0
    failures: list[str] = []

    entities = all_entities()
    if limit is not None:
        entities = entities[:limit]
        print(f"LIMIT: ingesting first {limit} entities only")

    for name, title, entity_type in entities:
        print(f"[{entity_type}] {name}")
        try:
            added = ingest_entity(
                store, name, title, entity_type, max_chars=max_chars
            )
            total_chunks += added
        except Exception as e:
            print(f"  ! error ingesting '{name}': {e}")
            failures.append(name)

    elapsed = time.time() - started
    counts = store.count_by_type()
    print()
    print(f"Done in {elapsed:.1f}s. Added {total_chunks} new chunks.")
    print(f"Collection now holds: people={counts['person']}, places={counts['place']}")
    if failures:
        print(f"Failures ({len(failures)}): {', '.join(failures)}")
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia data into the local RAG vector store.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before ingesting.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Quick mode: cap each article at {QUICK_MAX_ARTICLE_CHARS} chars (~3 min total).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest only the first N entities (useful for smoke tests).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(reset=args.reset, quick=args.quick, limit=args.limit)
