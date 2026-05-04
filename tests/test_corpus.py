"""Fixture sanity. No SF/thewalrus imports — runs anywhere."""

from photon_route.corpus import Document, load_fixture


def test_fixture_loads_five_documents():
    docs = load_fixture()
    assert len(docs) == 5
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.text for d in docs)
    assert all("id" in d.meta for d in docs)


def test_fixture_returns_independent_copies():
    a = load_fixture()
    b = load_fixture()
    a[0].meta["mutated"] = True
    assert "mutated" not in b[0].meta
