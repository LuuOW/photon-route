"""Encode text into a continuous-variable Strawberry Fields program.

Each word contributes deterministic squeezing + displacement parameters
(SHA-256 of the word) to one of N_MODES bosonic modes, alternating by
position. After all words are placed, a beam-splitter network mixes the
modes; its angles depend on sentence length, providing a coarse stand-in
for compositional structure.

Day-1 status: parameters are hash-bound, NOT learned. Distinct words
produce distinct programs and therefore distinct Gaussian states. This
gives a meaningful (deterministic, non-trivial) geometry sufficient for
end-to-end pipeline testing. Real research replaces the hash with a
trained parameter dict over an eval set.

Strawberry Fields is loaded lazily so importing photon_route at process
start (e.g. for the FastAPI stub mode) doesn't pay the SF + numpy import
cost when no encoding is requested.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any

from photon_route.corpus import Document

N_MODES = int(os.environ.get("PHOTON_ROUTE_N_MODES", "2"))
MAX_SQUEEZE = float(os.environ.get("PHOTON_ROUTE_MAX_SQUEEZE", "0.5"))
MAX_DISPLACE = float(os.environ.get("PHOTON_ROUTE_MAX_DISPLACE", "1.0"))


@dataclass
class EncodedDoc:
    """A document plus its Gaussian state. The state is an opaque SF
    object; the retrieval layer treats it as a black-box exposing
    `.means()` and `.cov()`."""

    doc: Document
    state: Any  # strawberryfields.backends.GaussianState


def _word_params(word: str) -> tuple[float, float, float, float]:
    """SHA-256(word) -> (r, phi_s, d_mag, d_phase) deterministically.

    r        in [0, MAX_SQUEEZE]    squeezing magnitude
    phi_s    in [0, 2*pi)           squeezing phase
    d_mag    in [0, MAX_DISPLACE]   displacement magnitude
    d_phase  in [0, 2*pi)           displacement phase

    Strawberry Fields' Dgate takes (magnitude, phase) on real arguments;
    complex displacements were deprecated. Squeezing magnitudes are
    capped well below 1 to keep photon numbers reasonable for the
    Gaussian backend.
    """
    h = hashlib.sha256(word.encode("utf-8")).digest()
    parts: list[float] = []
    for i in range(4):
        chunk = int.from_bytes(h[i * 8 : (i + 1) * 8], "big")
        parts.append((chunk % 10**9) / 10**9)
    r = parts[0] * MAX_SQUEEZE
    phi_s = parts[1] * 2 * math.pi
    d_mag = parts[2] * MAX_DISPLACE
    d_phase = parts[3] * 2 * math.pi
    return r, phi_s, d_mag, d_phase


def encode_one(text: str) -> Any:
    """Build and run an N_MODES-mode SF program; return the Gaussian state."""
    import strawberryfields as sf
    from strawberryfields import ops

    words = [w for w in text.lower().split() if w]
    if not words:
        raise ValueError("empty text")

    prog = sf.Program(N_MODES)

    with prog.context as q:
        for i, w in enumerate(words):
            r, phi_s, d_mag, d_phase = _word_params(w)
            mode = q[i % N_MODES]
            ops.Sgate(r, phi_s) | mode
            ops.Dgate(d_mag, d_phase) | mode
        if N_MODES >= 2:
            theta = (len(words) % 16) * (math.pi / 16)
            phi_bs = ((len(words) * 7) % 16) * (math.pi / 16)
            ops.BSgate(theta, phi_bs) | (q[0], q[1])

    eng = sf.Engine("gaussian")
    result = eng.run(prog)
    return result.state


def encode_corpus(items: list[Document] | list[str]) -> list[EncodedDoc]:
    """Encode every document in `items`."""
    out: list[EncodedDoc] = []
    for item in items:
        doc = Document(text=item, meta={}) if isinstance(item, str) else item
        out.append(EncodedDoc(doc=doc, state=encode_one(doc.text)))
    return out
