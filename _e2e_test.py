"""Quick end-to-end retrieval check.

Hits the live RAG pipeline only with queries whose target entities are
already in the index. Useful for validating that retrieval and the
grounding guard behave correctly without waiting for a full re-ingest.

Usage::

    python _e2e_test.py            # auto-pick queries that match indexed entities
    python _e2e_test.py --full     # always run the full PDF query set
"""

from __future__ import annotations

import argparse

from rag_engine import answer
from vector_store import VectorStore


FULL_QUERIES: list[tuple[str, str]] = [
    ("Who was Albert Einstein and what is he known for?", "person:Einstein"),
    ("What did Marie Curie discover?", "person:Curie"),
    ("Why is Nikola Tesla famous?", "person:Tesla"),
    ("What is Frida Kahlo known for?", "person:Kahlo"),
    ("Where is the Eiffel Tower located?", "place:Eiffel"),
    ("What was the Colosseum used for?", "place:Colosseum"),
    ("Where is Mount Everest?", "place:Everest"),
    ("Which famous place is located in Turkey?", "place:HagiaSophia"),
    ("Compare Albert Einstein and Nikola Tesla.", "person:both"),
    ("Compare the Eiffel Tower and the Statue of Liberty.", "place:both"),
    ("Who is the president of Mars?", "FAILURE"),
    ("Tell me about a random unknown person John Doe.", "FAILURE"),
]


def _pick_runnable(store: VectorStore) -> list[tuple[str, str]]:
    """Return the subset of FULL_QUERIES whose target entities are indexed."""
    res = store.collection.get(include=["metadatas"])
    indexed_names = {m.get("name") for m in res.get("metadatas") or [] if m}

    runnable: list[tuple[str, str]] = []
    for q, tag in FULL_QUERIES:
        if tag == "FAILURE":
            runnable.append((q, tag))
            continue
        target = tag.split(":", 1)[1].lower()
        if any(target.lower() in (n or "").lower() for n in indexed_names):
            runnable.append((q, tag))
    return runnable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run all queries")
    args = parser.parse_args()

    store = VectorStore()
    print(f"Index: {store.count()} chunks total")
    counts = store.count_by_type()
    print(f"  people={counts['person']}  places={counts['place']}\n")

    queries = FULL_QUERIES if args.full else _pick_runnable(store)
    if not queries:
        print("No matching entities indexed yet. Run `python ingest.py` first.")
        return

    print(f"Running {len(queries)} queries...\n")
    for query, tag in queries:
        print(f"Q ({tag}): {query}")
        result = answer(store, query)
        top_src = result.sources[0].name if result.sources else "<none>"
        top_dist = f"{result.sources[0].distance:.3f}" if result.sources else "n/a"
        snippet = result.answer.replace("\n", " ")[:220]
        print(
            f"  route={result.route}  grounded={result.grounded}  "
            f"top={top_src} (d={top_dist})"
        )
        print(f"  A: {snippet}\n")


if __name__ == "__main__":
    main()
