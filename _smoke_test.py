"""Offline smoke test: validates pure-Python logic (router, chunker, entities)
without touching ChromaDB or Ollama. Run with: python3 _smoke_test.py
"""

from __future__ import annotations

import sys


def test_entities() -> None:
    from entities import PEOPLE, PLACES, all_entities, match_entities

    assert len(PEOPLE) >= 20, f"need 20 people, got {len(PEOPLE)}"
    assert len(PLACES) >= 20, f"need 20 places, got {len(PLACES)}"

    mandatory_people = {
        "Albert Einstein", "Marie Curie", "Leonardo da Vinci", "William Shakespeare",
        "Ada Lovelace", "Nikola Tesla", "Lionel Messi", "Cristiano Ronaldo",
        "Taylor Swift", "Frida Kahlo",
    }
    mandatory_places = {
        "Eiffel Tower", "Great Wall of China", "Taj Mahal", "Grand Canyon",
        "Machu Picchu", "Colosseum", "Hagia Sophia", "Statue of Liberty",
        "Pyramids of Giza", "Mount Everest",
    }
    have_people = {n for n, _ in PEOPLE}
    have_places = {n for n, _ in PLACES}
    assert mandatory_people.issubset(have_people), mandatory_people - have_people
    assert mandatory_places.issubset(have_places), mandatory_places - have_places

    m = match_entities("Compare Lionel Messi and Cristiano Ronaldo")
    assert "Lionel Messi" in m["person"] and "Cristiano Ronaldo" in m["person"], m
    m = match_entities("Where is the Eiffel Tower located?")
    assert m["place"] == ["Eiffel Tower"], m
    m = match_entities("Who is the president of Mars?")
    assert m["person"] == [] and m["place"] == [], m

    assert len(all_entities()) == len(PEOPLE) + len(PLACES)
    print("entities OK")


def test_chunker() -> None:
    from ingest import chunk_text, clean_text

    text = (
        "Albert Einstein was a theoretical physicist. He developed the theory "
        "of relativity. He received the Nobel Prize in 1921. " * 20
    )
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 350, f"chunk too long: {len(c)}"
    overlap_seen = any(
        chunks[i][-30:] in chunks[i + 1][:80] for i in range(len(chunks) - 1)
    )
    assert overlap_seen or len(chunks) < 3
    print(f"chunker OK ({len(chunks)} chunks)")

    dirty = "Hello.\n\n== References ==\n[1] foo\n[2] bar\n== See also ==\nstuff"
    cleaned = clean_text(dirty)
    assert "References" not in cleaned and "foo" not in cleaned
    assert "Hello" in cleaned
    print("clean_text OK")


def test_router() -> None:
    from rag_engine import route

    assert route("Who was Albert Einstein?") == "person", route("Who was Albert Einstein?")
    assert route("Where is the Eiffel Tower located?") == "place"
    assert route("Compare Albert Einstein and the Eiffel Tower") == "both"
    assert route("Compare Lionel Messi and Cristiano Ronaldo") == "person"
    assert route("Compare the Eiffel Tower and the Statue of Liberty") == "place"
    r = route("Tell me something interesting")
    assert r in {"person", "place", "both"}
    print("router OK")


def main() -> int:
    try:
        test_entities()
        test_chunker()
        test_router()
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    print("\nAll offline smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
