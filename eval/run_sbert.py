"""SBERT (all-MiniLM-L6-v2) baseline against the same eval set.

Mean-pooled 384-d sentence embedding, cosine similarity. Establishes the
modern dense-retrieval ceiling for the photon-route eval. Runs entirely
on CPU in a few seconds for this corpus size.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval.fetch import fetch_all, verify_against_manifest


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, a in enumerate(ranked_ids[:k], start=1)
        if a in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--corpus",    type=Path, default=Path(__file__).parent / "corpus_ids.json")
    ap.add_argument("--relevance", type=Path, default=Path(__file__).parent / "relevance.json")
    ap.add_argument("--manifest",  type=Path, default=Path(__file__).parent / "manifest.json")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    ids = json.loads(args.corpus.read_text("utf-8"))["ids"]
    queries = json.loads(args.relevance.read_text("utf-8"))["queries"]
    abstracts = fetch_all(ids)
    bad = verify_against_manifest(abstracts, args.manifest)
    if bad:
        raise SystemExit(f"manifest mismatch: {list(bad)[:3]}")

    print(f"loading {args.model}...")
    model = SentenceTransformer(args.model)

    docs_in_order = [abstracts[i] for i in ids]
    doc_emb = model.encode(docs_in_order, normalize_embeddings=True, show_progress_bar=False)
    q_emb   = model.encode([q["query"] for q in queries], normalize_embeddings=True, show_progress_bar=False)

    per_query = []
    for qi, q in enumerate(queries):
        sims = doc_emb @ q_emb[qi]  # cosine since both are normalized
        order = np.argsort(-sims)
        ranked_ids = [ids[i] for i in order]
        rel = set(q["relevant_ids"])
        row = {"query": q["query"], "ranked": ranked_ids[: max(args.ks)]}
        for k in args.ks:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, rel, k)
            row[f"ndcg@{k}"]   = ndcg_at_k(ranked_ids, rel, k)
        per_query.append(row)

    aggregate = {f"recall@{k}": float(np.mean([q[f"recall@{k}"] for q in per_query])) for k in args.ks}
    aggregate.update(
        {f"ndcg@{k}": float(np.mean([q[f"ndcg@{k}"] for q in per_query])) for k in args.ks}
    )

    print(f"backend=sbert/{args.model.split('/')[-1]}  corpus={len(ids)} queries={len(queries)}")
    for q in per_query:
        cells = " ".join(f"{m}={q[m]:.3f}" for m in q if m.startswith(("recall", "ndcg")))
        print(f"  {q['query'][:48]:<48s}  {cells}")
    print("aggregate: " + " ".join(f"{m}={aggregate[m]:.3f}" for m in aggregate))


if __name__ == "__main__":
    main()
