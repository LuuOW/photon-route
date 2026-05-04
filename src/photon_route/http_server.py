"""FastAPI server for photon-route.

Detects on import whether the CV stack (strawberryfields + thewalrus) is
importable. If yes, /rank computes real Gaussian-state fidelity against
the fixture. If no, /rank returns the fixture in natural order with
score=0.0 — lets us verify deployment plumbing before the heavy stack.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from photon_route import __version__
from photon_route.corpus import Document, load_fixture

BACKEND: str
_corpus_encoded: list[Any] | None = None
_IMPORT_ERROR: str | None = None

try:
    import strawberryfields  # noqa: F401
    import thewalrus  # noqa: F401

    from photon_route.encode import encode_corpus
    from photon_route.retrieve import rank_against

    BACKEND = "gaussian"
except Exception as _e:
    BACKEND = "stub"
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    print(f"[photon-route] CV stack failed to import → backend=stub: {_IMPORT_ERROR}", flush=True)
    encode_corpus = None  # type: ignore[assignment]
    rank_against = None  # type: ignore[assignment]


app = FastAPI(
    title="photon-route",
    description=(
        "Continuous-variable photonic retrieval. Each document is encoded "
        "as a Gaussian state in Strawberry Fields; ranking is closed-form "
        "Gaussian-state fidelity (Banchi-Braunstein-Pirandola)."
    ),
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _ensure_corpus() -> None:
    global _corpus_encoded
    if _corpus_encoded is None and BACKEND == "gaussian":
        _corpus_encoded = encode_corpus(load_fixture())  # type: ignore[misc]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "photon-route",
        "version": __version__,
        "backend": BACKEND,
        "n_modes": int(os.environ.get("PHOTON_ROUTE_N_MODES", "2")),
        "endpoints": ["/", "/health", "/version", "/rank", "/docs"],
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "backend": BACKEND}
    if _IMPORT_ERROR:
        out["import_error"] = _IMPORT_ERROR
    return out


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__, "backend": BACKEND}


@app.get("/rank")
def rank(
    q: str = Query(..., min_length=1, max_length=500, description="query text"),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    if BACKEND == "stub":
        docs: list[Document] = load_fixture()[:top_k]
        return {
            "query": q,
            "backend": BACKEND,
            "results": [
                {"rank": i + 1, "score": 0.0, "text": d.text, "meta": d.meta}
                for i, d in enumerate(docs)
            ],
        }

    _ensure_corpus()
    try:
        scored = rank_against(_corpus_encoded, q, top_k=top_k)  # type: ignore[arg-type]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rank failed: {e!r}") from e

    return {
        "query": q,
        "backend": BACKEND,
        "results": [
            {
                "rank": i + 1,
                "score": round(r.score, 6),
                "text": r.doc.doc.text,
                "meta": r.doc.doc.meta,
            }
            for i, r in enumerate(scored)
        ],
    }
