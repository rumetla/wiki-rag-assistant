"""Canonical entity catalog for the Local Wikipedia RAG Assistant.

Defines the people and places the system ingests from Wikipedia and exposes
helpers used by the rule-based query router in `rag_engine.py`.

Each entry is a tuple ``(canonical_name, wikipedia_title)``:
- ``canonical_name`` is the human-readable label shown in sources and used
  for keyword routing (case-insensitive substring match).
- ``wikipedia_title`` is the exact title passed to the Wikipedia API. We keep
  these decoupled because some Wikipedia titles are disambiguated (e.g.
  ``Mercury (planet)``) while users still type the canonical short form.
"""

from __future__ import annotations

import re
from typing import Literal


EntityType = Literal["person", "place"]


PEOPLE: list[tuple[str, str]] = [
    ("Albert Einstein", "Albert Einstein"),
    ("Marie Curie", "Marie Curie"),
    ("Leonardo da Vinci", "Leonardo da Vinci"),
    ("William Shakespeare", "William Shakespeare"),
    ("Ada Lovelace", "Ada Lovelace"),
    ("Nikola Tesla", "Nikola Tesla"),
    ("Lionel Messi", "Lionel Messi"),
    ("Cristiano Ronaldo", "Cristiano Ronaldo"),
    ("Taylor Swift", "Taylor Swift"),
    ("Frida Kahlo", "Frida Kahlo"),
    ("Isaac Newton", "Isaac Newton"),
    ("Charles Darwin", "Charles Darwin"),
    ("Stephen Hawking", "Stephen Hawking"),
    ("Mahatma Gandhi", "Mahatma Gandhi"),
    ("Nelson Mandela", "Nelson Mandela"),
    ("Mustafa Kemal Ataturk", "Mustafa Kemal Atatürk"),
    ("Cleopatra", "Cleopatra"),
    ("Vincent van Gogh", "Vincent van Gogh"),
    ("Ludwig van Beethoven", "Ludwig van Beethoven"),
    ("Michael Jordan", "Michael Jordan"),
]


PLACES: list[tuple[str, str]] = [
    ("Eiffel Tower", "Eiffel Tower"),
    ("Great Wall of China", "Great Wall of China"),
    ("Taj Mahal", "Taj Mahal"),
    ("Grand Canyon", "Grand Canyon"),
    ("Machu Picchu", "Machu Picchu"),
    ("Colosseum", "Colosseum"),
    ("Hagia Sophia", "Hagia Sophia"),
    ("Statue of Liberty", "Statue of Liberty"),
    ("Pyramids of Giza", "Giza pyramid complex"),
    ("Mount Everest", "Mount Everest"),
    ("Petra", "Petra"),
    ("Stonehenge", "Stonehenge"),
    ("Sagrada Familia", "Sagrada Família"),
    ("Acropolis of Athens", "Acropolis of Athens"),
    ("Angkor Wat", "Angkor Wat"),
    ("Christ the Redeemer", "Christ the Redeemer (statue)"),
    ("Niagara Falls", "Niagara Falls"),
    ("Mount Fuji", "Mount Fuji"),
    ("Topkapi Palace", "Topkapı Palace"),
    ("Burj Khalifa", "Burj Khalifa"),
]


PERSON_NAMES: set[str] = {name.lower() for name, _ in PEOPLE}
PLACE_NAMES: set[str] = {name.lower() for name, _ in PLACES}


PERSON_KEYWORDS: set[str] = {
    "who",
    "whose",
    "born",
    "birth",
    "died",
    "death",
    "person",
    "people",
    "scientist",
    "physicist",
    "artist",
    "painter",
    "writer",
    "author",
    "musician",
    "singer",
    "footballer",
    "player",
    "discovered",
    "invented",
    "wrote",
    "famous for",
    "known for",
    "her",
    "his",
}


PLACE_KEYWORDS: set[str] = {
    "where",
    "located",
    "location",
    "place",
    "city",
    "country",
    "monument",
    "landmark",
    "building",
    "tower",
    "wall",
    "mountain",
    "river",
    "wonder",
    "ancient",
    "tourist",
    "visit",
}


def _name_matches(name: str, query_lower: str) -> bool:
    """Return True if ``name`` is mentioned in ``query_lower``.

    Tries two strategies:
    1. Full canonical name as a substring (e.g. "albert einstein" in query).
    2. Distinctive last token as a whole-word match (e.g. "einstein" → match,
       but "tower" alone does NOT match because it's too short / generic).

    The last-token rule lets users say *"Where is Everest?"* and still hit
    the *Mount Everest* entity, while avoiding false positives from common
    short words.
    """
    name_lower = name.lower()
    if name_lower in query_lower:
        return True
    last_token = name_lower.split()[-1]
    if len(last_token) < 5:
        return False
    return re.search(rf"\b{re.escape(last_token)}\b", query_lower) is not None


def match_entities(query: str) -> dict[str, list[str]]:
    """Return canonical entity names mentioned in ``query``.

    Uses substring + last-token matching against the canonical name lists.
    We deliberately keep this rule-based per `.cursorrules`
    ("Keyword based or rule-based approaches are acceptable").

    Returns a dict with ``"person"`` and ``"place"`` keys mapping to lists
    of canonical names found.
    """
    q = query.lower()
    found_people = [name for name, _ in PEOPLE if _name_matches(name, q)]
    found_places = [name for name, _ in PLACES if _name_matches(name, q)]
    return {"person": found_people, "place": found_places}


def all_entities() -> list[tuple[str, str, EntityType]]:
    """Return the full catalog as ``(canonical_name, wiki_title, type)``."""
    items: list[tuple[str, str, EntityType]] = []
    for name, title in PEOPLE:
        items.append((name, title, "person"))
    for name, title in PLACES:
        items.append((name, title, "place"))
    return items
