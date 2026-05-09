"""Read sweep_results.csv, decide whether SBERT-photon's gain is real.

Three questions answered:

  Q1. Is SBERT-photon's *full* (squeezing on) test nDCG@10 beating raw
      SBERT (0.385) by a margin larger than its split-to-split stdev?
      → if yes, the photonic structure adds robust signal over a
        pretrained backbone; if no, the headline 0.502 was lucky.

  Q2. Does the squeezing layer specifically pay? Compare full vs
      no-squeeze across the same seeds. Paired difference > stdev → yes.

  Q3. Is there a generalization tax? Train nDCG@10 minus test nDCG@10.
      Smaller gap = better generalization. Useful for sanity.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


SBERT_ALONE_NDCG10 = 0.385  # raw all-MiniLM-L6-v2 on the same eval, no training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path(__file__).resolve().parent.parent / "sweep_results.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(args.csv.open()))
    for r in rows:
        for k, v in list(r.items()):
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass

    full = sorted([r for r in rows if int(r["no_squeeze"]) == 0], key=lambda r: r["seed"])
    nosq = sorted([r for r in rows if int(r["no_squeeze"]) == 1], key=lambda r: r["seed"])

    def stat(rs, key):
        vals = [r[key] for r in rs if r[key] == r[key]]  # drop NaN
        if len(vals) < 2:
            return (vals[0] if vals else float("nan"), 0.0, len(vals))
        return statistics.mean(vals), statistics.stdev(vals), len(vals)

    print(f"loaded {len(rows)} runs from {args.csv}")
    print(f"  full (squeezing on):  n={len(full)}")
    print(f"  no-squeeze:           n={len(nosq)}")

    print("\n=== Q1. Does SBERT-photon (full) robustly beat raw SBERT? ===")
    m, s, n = stat(full, "test_ndcg10")
    delta = m - SBERT_ALONE_NDCG10
    z_like = delta / s if s > 0 else float("inf")
    verdict_q1 = "✓ YES — gain exceeds split-to-split noise" if delta > s and m > SBERT_ALONE_NDCG10 else "✗ noisy or no gain"
    print(f"  full mean test nDCG@10 = {m:.3f} ± {s:.3f}  (n={n})")
    print(f"  vs raw SBERT (no training) = {SBERT_ALONE_NDCG10}")
    print(f"  Δ = {delta:+.3f}   (Δ/σ ≈ {z_like:+.2f})")
    print(f"  → {verdict_q1}")

    print("\n=== Q2. Does the squeezing layer specifically pay? ===")
    paired_seeds = sorted(set(r["seed"] for r in full) & set(r["seed"] for r in nosq))
    pairs = []
    for s_id in paired_seeds:
        f = next((r for r in full if r["seed"] == s_id), None)
        n = next((r for r in nosq if r["seed"] == s_id), None)
        if f and n:
            pairs.append((s_id, f["test_ndcg10"], n["test_ndcg10"]))
    if not pairs:
        print("  no paired seeds — cannot compute")
    else:
        diffs = [f - nq for _, f, nq in pairs]
        m_d  = statistics.mean(diffs)
        s_d  = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        verdict_q2 = "✓ YES — squeezing helps consistently" if m_d > s_d and m_d > 0.01 else (
            "✗ NO — squeezing adds nothing or hurts" if m_d <= 0 else
            "≈ marginal — within noise"
        )
        print(f"  paired (full − no_squeeze) test nDCG@10:")
        for s_id, f, nq in pairs:
            print(f"    seed {int(s_id)}:  full={f:.3f}  no_sq={nq:.3f}  Δ={f-nq:+.3f}")
        print(f"  mean Δ = {m_d:+.3f} ± {s_d:.3f}  (n={len(pairs)})")
        print(f"  → {verdict_q2}")

    print("\n=== Q3. Generalization tax (train − test nDCG@10) ===")
    for label, rs in [("full", full), ("no-squeeze", nosq)]:
        if not rs:
            continue
        gaps = [r["train_ndcg10"] - r["test_ndcg10"] for r in rs]
        m_g = statistics.mean(gaps)
        s_g = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
        print(f"  {label:>10}: gap = {m_g:.3f} ± {s_g:.3f}")

    print("\nFull table:")
    print(f"{'seed':>4} {'mode':>10} {'train_nDCG':>10} {'test_nDCG':>10} {'test_R@1':>10} {'test_R@10':>10}")
    for r in full + nosq:
        mode = "no_squeeze" if int(r["no_squeeze"]) else "full"
        print(f"{int(r['seed']):>4} {mode:>10} {r['train_ndcg10']:>10.3f} "
              f"{r['test_ndcg10']:>10.3f} {r.get('test_recall1', float('nan')):>10.3f} "
              f"{r.get('test_recall10', float('nan')):>10.3f}")


if __name__ == "__main__":
    main()
