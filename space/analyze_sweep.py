"""Read sweep_results.csv, decide whether SBERT-photon's gain is real
and whether photon-number-distribution metric (A3-Simple) outperforms
Gaussian-state-overlap (BBP fidelity) on the same trained encoder.

Q1. Does SBERT-photon (full, gaussian metric) robustly beat raw SBERT?
Q2. Does the squeezing layer specifically pay (full vs no-squeeze)?
Q3. A3-Simple: does photon-prob metric > gaussian metric on same encoder?
Q4. Generalization tax (train − test nDCG@10).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


SBERT_ALONE_NDCG10 = 0.385


def stat(rs, key):
    vals = [r[key] for r in rs if r[key] == r[key]]
    if len(vals) < 2:
        return (vals[0] if vals else float("nan"), 0.0, len(vals))
    return statistics.mean(vals), statistics.stdev(vals), len(vals)


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

    print(f"loaded {len(rows)} runs from {args.csv}")
    print(f"  full (squeezing on):  n={len(full)}")
    print(f"  no-squeeze:           n={len(nosq)}")

    print("\n=== Q1. SBERT-photon (full, gaussian metric) vs raw SBERT (0.385) ===")
    m, s, n = stat(full, "test_gaussian_n10")
    delta = m - SBERT_ALONE_NDCG10
    z = delta / s if s > 0 else float("inf")
    verdict = "✓ YES" if delta > s and m > SBERT_ALONE_NDCG10 else "✗ noisy or no gain"
    print(f"  full mean test nDCG@10 (gaussian) = {m:.3f} ± {s:.3f}  (n={n})")
    print(f"  Δ vs raw SBERT = {delta:+.3f}  (Δ/σ ≈ {z:+.2f})  → {verdict}")

    print("\n=== Q2. Squeezing pays? (paired full − no_squeeze, gaussian metric) ===")
    paired = []
    for r in full:
        n_row = next((x for x in nosq if x["seed"] == r["seed"]), None)
        if n_row:
            paired.append((int(r["seed"]), r["test_gaussian_n10"], n_row["test_gaussian_n10"]))
    diffs = [a - b for _, a, b in paired]
    m_d = statistics.mean(diffs) if diffs else float("nan")
    s_d = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    for sid, a, b in paired:
        print(f"    seed {sid}:  full={a:.3f}  no_sq={b:.3f}  Δ={a-b:+.3f}")
    verdict = ("✓ YES" if m_d > s_d and m_d > 0.01 else
               "✗ NO" if m_d <= 0 else "≈ within noise")
    print(f"  mean Δ = {m_d:+.3f} ± {s_d:.3f}  → {verdict}")

    print("\n=== Q3. A3-Simple: photon-prob > gaussian metric on same encoder? ===")
    for label, rs in [("full", full), ("no-squeeze", nosq)]:
        if not rs:
            continue
        # Paired per-seed: same encoder, two metrics on test set
        diffs = [r["test_photon_prob_n10"] - r["test_gaussian_n10"] for r in rs]
        m_d = statistics.mean(diffs)
        s_d = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        m_g, _, _ = stat(rs, "test_gaussian_n10")
        m_p, _, _ = stat(rs, "test_photon_prob_n10")
        verdict = ("✓ photon-prob wins" if m_d > s_d and m_d > 0.01 else
                   "✗ photon-prob loses" if m_d < -0.01 else
                   "≈ tie within noise")
        print(f"  {label:>10}: gaussian={m_g:.3f}  photon_prob={m_p:.3f}  Δ={m_d:+.3f} ± {s_d:.3f}  → {verdict}")

    print("\n=== Q4. Generalization tax (train − test, gaussian metric) ===")
    for label, rs in [("full", full), ("no-squeeze", nosq)]:
        if not rs:
            continue
        gaps = [r["train_gaussian_n10"] - r["test_gaussian_n10"] for r in rs]
        m_g = statistics.mean(gaps)
        s_g = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
        print(f"  {label:>10}: gap = {m_g:.3f} ± {s_g:.3f}")

    print("\nFull table:")
    headers = ("seed", "mode", "train_g_n10", "test_g_n10", "train_p_n10", "test_p_n10")
    print("  " + "  ".join(f"{h:>11}" for h in headers))
    for r in full + nosq:
        mode = "no_squeeze" if int(r["no_squeeze"]) else "full"
        cells = (
            int(r["seed"]), mode,
            r["train_gaussian_n10"], r["test_gaussian_n10"],
            r["train_photon_prob_n10"], r["test_photon_prob_n10"],
        )
        print("  " + "  ".join(
            f"{c:>11}" if isinstance(c, str) else f"{c:>11.3f}" if isinstance(c, float) else f"{c:>11}"
            for c in cells
        ))


if __name__ == "__main__":
    main()
