"""photon-route — continuous-variable photonic retrieval."""

__version__ = "0.1.0"

from photon_route.corpus import Document, load_fixture
from photon_route.encode import EncodedDoc, encode_corpus, encode_one
from photon_route.retrieve import ScoredDoc, gaussian_fidelity, rank_against

__all__ = [
    "Document",
    "EncodedDoc",
    "ScoredDoc",
    "encode_corpus",
    "encode_one",
    "gaussian_fidelity",
    "load_fixture",
    "rank_against",
]
