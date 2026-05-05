"""Score retrieval quality on the eval corpus.

Reports Recall@k and nDCG@k for any encoder that exposes encode(text) ->
(mu, sigma). Intended to make trained-v2 vs SHA-init-v2 vs v1-via-SF
comparable apples-to-apples.

Usage:
    python -m eval.run                       # SHA-init v2 (default)
    python -m eval.run --weights weights.npz # trained v2
    python -m eval.run --backend v1          # v1-via-strawberryfields
"""

from __future__ import annotations

# scipy 1.17 removed `simps`; SF 0.23 still imports it. Shim before SF
# is loaded by any v1 backend code path. Mirrors tests/conftest.py.
import scipy.integrate as _si  # noqa: E402
if not hasattr(_si, "simps"):
    _si.simps = _si.simpson  # type: ignore[attr-defined]

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from eval.fetch import fetch_all, verify_against_manifest


def _ensure_path():
    here = Path(__file__).resolve().parent.parent / "src"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def make_v2_encoder(weights_path: Path | None):
    _ensure_path()
    from photon_route.v2 import Encoder, sha_params_v1_compat
    from photon_route.v2.encoder import dict_params_fn

    if weights_path is None:
        return Encoder(params_fn=sha_params_v1_compat)
    data = np.load(weights_path, allow_pickle=True)  # words is dtype=object
    table = {str(w): np.asarray(v, np.float64) for w, v in zip(data["words"], data["params"])}
    unk = np.asarray(data["unk"], np.float64) if "unk" in data.files else None
    return Encoder(params_fn=dict_params_fn(table, unk=unk))


def make_v1_encoder():
    _ensure_path()
    from photon_route.encode import encode_one

    class _V1:
        def encode(self, text):
            s = encode_one(text)
            return np.asarray(s.means(), np.float64), np.asarray(s.cov(), np.float64)

    return _V1()


def fidelity(mu_a, sg_a, mu_b, sg_b):
    from thewalrus.quantum import fidelity as tw_fidelity

    f = tw_fidelity(mu_a, sg_a, mu_b, sg_b)
    val = float(f.real if hasattr(f, "real") else f)
    return max(0.0, min(1.0, val))


def encode_corpus(encoder, abstracts: dict[str, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {arxiv_id: encoder.encode(text) for arxiv_id, text in abstracts.items()}


def rank(encoder, query: str, encoded_corpus: dict) -> list[tuple[str, float]]:
    mu_q, sg_q = encoder.encode(query)
    scored = []
    for arxiv_id, (mu_d, sg_d) in encoded_corpus.items():
        try:
            s = fidelity(mu_q, sg_q, mu_d, sg_d)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            s = 0.0
        scored.append((arxiv_id, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def recall_at_k(ranked: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    top = {a for a, _ in ranked[:k]}
    return len(top & relevant) / len(relevant)


def ndcg_at_k(ranked: list[tuple[str, float]], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = 0.0
    for i, (arxiv_id, _) in enumerate(ranked[:k], start=1):
        if arxiv_id in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["v2", "v1"], default="v2")
    ap.add_argument("--weights", type=Path, default=None,
                    help="path to trained v2 weights (.npz with 'words', 'params', optional 'unk')")
    ap.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus_ids.json")
    ap.add_argument("--relevance", type=Path, default=Path(__file__).parent / "relevance.json")
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.json")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip manifest hash verification (manifest may be empty)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of human text")
    args = ap.parse_args()

    ids = json.loads(args.corpus.read_text("utf-8"))["ids"]
    relevance = json.loads(args.relevance.read_text("utf-8"))["queries"]

    abstracts = fetch_all(ids)
    if not args.skip_verify and json.loads(args.manifest.read_text("utf-8")).get("hashes"):
        bad = verify_against_manifest(abstracts, args.manifest)
        if bad:
            sys.exit(f"manifest mismatch: {list(bad)[:5]} (and {max(0, len(bad) - 5)} more)")

    encoder = make_v1_encoder() if args.backend == "v1" else make_v2_encoder(args.weights)

    t0 = time.time()
    encoded = encode_corpus(encoder, abstracts)
    encode_secs = time.time() - t0

    per_query = []
    for q in relevance:
        ranked = rank(encoder, q["query"], encoded)
        rel = set(q["relevant_ids"])
        row = {"query": q["query"], "ranked": [a for a, _ in ranked[:max(args.ks)]]}
        for k in args.ks:
            row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"ndcg@{k}"]   = ndcg_at_k(ranked, rel, k)
        per_query.append(row)

    aggregate = {f"recall@{k}": float(np.mean([q[f"recall@{k}"] for q in per_query])) for k in args.ks}
    aggregate.update({f"ndcg@{k}": float(np.mean([q[f"ndcg@{k}"] for q in per_query])) for k in args.ks})

    payload = {
        "backend": args.backend,
        "weights": str(args.weights) if args.weights else None,
        "n_corpus": len(ids),
        "n_queries": len(relevance),
        "encode_seconds": encode_secs,
        "per_query": per_query,
        "aggregate": aggregate,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"backend={args.backend} weights={args.weights} corpus={len(ids)} queries={len(relevance)} encode={encode_secs:.2f}s")
        for q in per_query:
            cells = " ".join(f"{m}={q[m]:.3f}" for m in q if m.startswith(("recall", "ndcg")))
            print(f"  {q['query'][:48]:<48s}  {cells}")
        print("aggregate: " + " ".join(f"{m}={aggregate[m]:.3f}" for m in aggregate))


if __name__ == "__main__":
    main()
