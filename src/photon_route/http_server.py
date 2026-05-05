"""FastAPI server for photon-route.

Three retrieval backends, all sharing the same 5-doc fixture corpus:

  v1        — original SHA-256 → SF gaussian backend (StrawberryFields)
  sha_init  — pure-numpy v2 encoder, identical SHA-256 init (Step 0
              equivalence: produces (mu, sigma) byte-identical to v1
              modulo dtype). Useful as a no-torch baseline.
  trained   — pure-numpy v2 encoder loaded from /app/weights.npz, the
              artifact produced by `space.train`. Only available when
              the Docker image was built with the training stage.

Importing the CV stack (strawberryfields + thewalrus) is required for
v1/sha_init/trained to function — fidelity scoring is closed-form
Gaussian-state fidelity (Banchi-Braunstein-Pirandola). If the import
fails, /rank degrades to the stub mode (deployment plumbing only).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from photon_route import __version__
from photon_route.corpus import Document, load_fixture

BACKEND_AVAILABLE: list[str] = []
_IMPORT_ERROR: str | None = None
_v1_corpus: list[Any] | None = None
_v2_sha_corpus: list[tuple[np.ndarray, np.ndarray, str, dict]] | None = None
_v2_trained_corpus: list[tuple[np.ndarray, np.ndarray, str, dict]] | None = None
_v2_sha_encoder = None
_v2_trained_encoder = None
WEIGHTS_PATH = Path(os.environ.get("PHOTON_ROUTE_WEIGHTS", "/app/weights.npz"))

try:
    import strawberryfields  # noqa: F401
    import thewalrus  # noqa: F401
    from thewalrus.quantum import fidelity as _tw_fidelity

    from photon_route.encode import encode_corpus as _v1_encode_corpus
    from photon_route.retrieve import rank_against as _v1_rank_against
    from photon_route.v2 import Encoder, sha_params_v1_compat
    from photon_route.v2.encoder import dict_params_fn

    BACKEND_AVAILABLE.append("v1")
    BACKEND_AVAILABLE.append("sha_init")

    _v2_sha_encoder = Encoder(params_fn=sha_params_v1_compat)

    if WEIGHTS_PATH.exists():
        _data = np.load(WEIGHTS_PATH, allow_pickle=True)
        _table = {
            str(w): np.asarray(p, dtype=np.float64)
            for w, p in zip(_data["words"], _data["params"])
        }
        _unk = np.asarray(_data["unk"], dtype=np.float64) if "unk" in _data.files else None
        _v2_trained_encoder = Encoder(params_fn=dict_params_fn(_table, unk=_unk))
        BACKEND_AVAILABLE.append("trained")
        print(f"[photon-route] loaded trained weights: |V|={len(_table)}", flush=True)
    else:
        print(f"[photon-route] no trained weights at {WEIGHTS_PATH}; trained backend disabled", flush=True)

except Exception as _e:
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    print(f"[photon-route] CV stack failed → backend=stub: {_IMPORT_ERROR}", flush=True)


def _has_cv() -> bool:
    return "v1" in BACKEND_AVAILABLE


DEFAULT_BACKEND = "trained" if "trained" in BACKEND_AVAILABLE else (
    "sha_init" if "sha_init" in BACKEND_AVAILABLE else "stub"
)


app = FastAPI(
    title="photon-route",
    description=(
        "Continuous-variable photonic retrieval. Each document is encoded "
        "as a Gaussian state; ranking is closed-form Gaussian-state "
        "fidelity (Banchi-Braunstein-Pirandola). Three swappable encoders: "
        "v1 (SF + SHA), sha_init (numpy + SHA), trained (numpy + learned)."
    ),
    version=__version__,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _ensure_v1_corpus() -> list[Any]:
    global _v1_corpus
    if _v1_corpus is None:
        _v1_corpus = _v1_encode_corpus(load_fixture())
    return _v1_corpus


def _encode_corpus_numpy(encoder) -> list[tuple[np.ndarray, np.ndarray, str, dict]]:
    return [
        (*encoder.encode(d.text), d.text, d.meta)
        for d in load_fixture()
    ]


def _ensure_v2_sha_corpus():
    global _v2_sha_corpus
    if _v2_sha_corpus is None:
        _v2_sha_corpus = _encode_corpus_numpy(_v2_sha_encoder)
    return _v2_sha_corpus


def _ensure_v2_trained_corpus():
    global _v2_trained_corpus
    if _v2_trained_corpus is None and _v2_trained_encoder is not None:
        _v2_trained_corpus = _encode_corpus_numpy(_v2_trained_encoder)
    return _v2_trained_corpus


def _safe_fidelity(mu_a, sg_a, mu_b, sg_b) -> float:
    try:
        f = _tw_fidelity(mu_a, sg_a, mu_b, sg_b)
        val = float(f.real if hasattr(f, "real") else f)
        return max(0.0, min(1.0, val))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return 0.0


def _rank_v2(encoder, encoded_corpus, query: str, top_k: int):
    mu_q, sg_q = encoder.encode(query)
    scored = [
        (_safe_fidelity(mu_q, sg_q, mu_d, sg_d), text, meta)
        for (mu_d, sg_d, text, meta) in encoded_corpus
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "name": "photon-route",
        "version": __version__,
        "backends_available": BACKEND_AVAILABLE or ["stub"],
        "default_backend": DEFAULT_BACKEND,
        "n_modes": int(os.environ.get("PHOTON_ROUTE_N_MODES", "2")),
        "weights_path": str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None,
        "endpoints": ["/", "/health", "/version", "/rank", "/docs"],
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": True,
        "backends_available": BACKEND_AVAILABLE or ["stub"],
        "default_backend": DEFAULT_BACKEND,
        "weights_loaded": "trained" in BACKEND_AVAILABLE,
    }
    if _IMPORT_ERROR:
        out["import_error"] = _IMPORT_ERROR
    return out


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__, "default_backend": DEFAULT_BACKEND}


@app.get("/weights.npz")
def weights_download():
    """Serve the trained weights.npz so the eval harness on the VM can
    score `--backend v2 --weights weights.npz` without rebuilding the
    whole training pipeline locally. Read-only, baked at build time."""
    if not WEIGHTS_PATH.exists():
        raise HTTPException(status_code=404, detail="no trained weights on this build")
    return FileResponse(
        WEIGHTS_PATH,
        media_type="application/octet-stream",
        filename="weights.npz",
    )


@app.get("/rank")
def rank(
    q: str = Query(..., min_length=1, max_length=500, description="query text"),
    top_k: int = Query(5, ge=1, le=20),
    backend: str = Query(
        DEFAULT_BACKEND,
        description="encoder: v1 | sha_init | trained (falls back to default if unavailable)",
    ),
) -> dict[str, Any]:
    if not _has_cv():
        docs: list[Document] = load_fixture()[:top_k]
        return {
            "query": q,
            "backend": "stub",
            "results": [
                {"rank": i + 1, "score": 0.0, "text": d.text, "meta": d.meta}
                for i, d in enumerate(docs)
            ],
        }

    if backend not in BACKEND_AVAILABLE:
        backend = DEFAULT_BACKEND

    try:
        if backend == "v1":
            scored = _v1_rank_against(_ensure_v1_corpus(), q, top_k=top_k)
            results = [
                {"rank": i + 1, "score": round(r.score, 6),
                 "text": r.doc.doc.text, "meta": r.doc.doc.meta}
                for i, r in enumerate(scored)
            ]
        else:
            enc = _v2_trained_encoder if backend == "trained" else _v2_sha_encoder
            corpus = (
                _ensure_v2_trained_corpus() if backend == "trained"
                else _ensure_v2_sha_corpus()
            )
            scored = _rank_v2(enc, corpus, q, top_k)
            results = [
                {"rank": i + 1, "score": round(s, 6), "text": text, "meta": meta}
                for i, (s, text, meta) in enumerate(scored)
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rank failed: {e!r}") from e

    return {"query": q, "backend": backend, "results": results}
