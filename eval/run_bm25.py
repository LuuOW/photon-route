"""BM25 baseline against the same eval set photon-route uses.

Drops in next to eval.run so apples-to-apples on Recall@k / nDCG@k.
Pure-stdlib BM25 — no external IR library — to keep the dependency
surface identical to the rest of eval/.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from eval.fetch import fetch_all, verify_against_manifest


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [d.lower().split() for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / self.N
        df: Counter = Counter()
        for t in self.toks:
            for w in set(t):
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        d = self.toks[doc_index]
        tf = Counter(d)
        s = 0.0
        for w in query.lower().split():
            if w not in self.idf:
                continue
            f = tf[w]
            denom = f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            s += self.idf[w] * f * (self.k1 + 1) / max(denom, 1e-9)
        return s


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
    ap.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus_ids.json")
    ap.add_argument("--relevance", type=Path, default=Path(__file__).parent / "relevance.json")
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.json")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    args = ap.parse_args()

    ids = json.loads(args.corpus.read_text("utf-8"))["ids"]
    queries = json.loads(args.relevance.read_text("utf-8"))["queries"]
    abstracts = fetch_all(ids)
    bad = verify_against_manifest(abstracts, args.manifest)
    if bad:
        raise SystemExit(f"manifest mismatch: {list(bad)[:3]}")

    docs_in_order = [abstracts[i] for i in ids]
    bm25 = BM25(docs_in_order)

    per_query = []
    for q in queries:
        scored = sorted(
            ((bm25.score(q["query"], i), ids[i]) for i in range(len(ids))),
            key=lambda x: -x[0],
        )
        ranked_ids = [doc_id for _, doc_id in scored]
        rel = set(q["relevant_ids"])
        row = {"query": q["query"], "ranked": ranked_ids[: max(args.ks)]}
        for k in args.ks:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, rel, k)
        per_query.append(row)

    aggregate = {
        f"recall@{k}": float(np.mean([q[f"recall@{k}"] for q in per_query])) for k in args.ks
    }
    aggregate.update(
        {f"ndcg@{k}": float(np.mean([q[f"ndcg@{k}"] for q in per_query])) for k in args.ks}
    )
    print(f"backend=bm25 corpus={len(ids)} queries={len(queries)}")
    for q in per_query:
        cells = " ".join(
            f"{m}={q[m]:.3f}" for m in q if m.startswith(("recall", "ndcg"))
        )
        print(f"  {q['query'][:48]:<48s}  {cells}")
    print("aggregate: " + " ".join(f"{m}={aggregate[m]:.3f}" for m in aggregate))


if __name__ == "__main__":
    main()
