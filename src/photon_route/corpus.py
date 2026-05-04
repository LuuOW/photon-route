"""Document model + day-1 fixture corpus.

The fixture mirrors qrouter's so the two repos stay directly comparable:
identical input strings, identical ranking interface, different substrate
underneath.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


_FIXTURE: list[Document] = [
    Document(
        text="photons travel through optical fibers without losing coherence",
        meta={"id": "fixture-001", "topic": "fiber-optics"},
    ),
    Document(
        text="superposition lets a quantum bit represent multiple states at once",
        meta={"id": "fixture-002", "topic": "qubits"},
    ),
    Document(
        text="atoms absorb light at specific frequencies",
        meta={"id": "fixture-003", "topic": "spectroscopy"},
    ),
    Document(
        text="cats observe boxes with opening lids",
        meta={"id": "fixture-004", "topic": "measurement"},
    ),
    Document(
        text="qubits entangle across distant detectors",
        meta={"id": "fixture-005", "topic": "entanglement"},
    ),
]


def load_fixture() -> list[Document]:
    """Return a fresh list of fixture documents."""
    return [Document(text=d.text, meta=dict(d.meta)) for d in _FIXTURE]
