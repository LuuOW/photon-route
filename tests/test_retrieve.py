"""ScoredDoc shape + rank_against ordering, with a stub fidelity.

We don't import strawberryfields here — the goal is to test the
ranking/sorting logic deterministically without paying the import cost.
The actual Gaussian-state fidelity is exercised by the live HF Space at
deploy time; numerical correctness of thewalrus.quantum.fidelity is
their responsibility, not ours.
"""

from dataclasses import dataclass

import pytest


@dataclass
class _FakeDoc:
    text: str


@dataclass
class _FakeEncoded:
    doc: _FakeDoc
    score_proxy: float


def test_score_doc_dataclass_shape():
    from photon_route.retrieve import ScoredDoc

    s = ScoredDoc(doc=_FakeEncoded(doc=_FakeDoc(text="x"), score_proxy=0.0), score=0.42)
    assert s.score == pytest.approx(0.42)


def test_rank_against_sorts_descending(monkeypatch):
    """Substitute encode_one + gaussian_fidelity with deterministic stubs;
    verify rank_against returns documents sorted by score, descending."""
    from photon_route import retrieve

    corpus = [
        _FakeEncoded(doc=_FakeDoc(text="a"), score_proxy=0.1),
        _FakeEncoded(doc=_FakeDoc(text="b"), score_proxy=0.9),
        _FakeEncoded(doc=_FakeDoc(text="c"), score_proxy=0.5),
    ]

    def fake_encode_one(text):
        return object()

    def fake_fidelity(q, d):
        return d.score_proxy

    monkeypatch.setattr(retrieve, "encode_one", fake_encode_one)
    monkeypatch.setattr(retrieve, "gaussian_fidelity", fake_fidelity)

    out = retrieve.rank_against(corpus, "anything", top_k=2)
    assert [r.doc.doc.text for r in out] == ["b", "c"]
    assert out[0].score == pytest.approx(0.9)
