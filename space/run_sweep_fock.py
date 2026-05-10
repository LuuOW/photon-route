"""5-split sweep for the A3-Real Fock-basis trainer.

Mirrors space/run_sweep.py but for train_sbert_fock.py (non-Gaussian
heralded encoder). Outputs one CSV row per (seed) — no squeeze ablation
since the Fock encoder structure already includes a learnable TMS gate
and learnable squeezing; the equivalent ablation is herald_n=0 (heralding
on vacuum keeps the state Gaussian).
"""
from __future__ import annotations

import argparse
import csv
import io
import contextlib
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_split(rel_payload, n_test, seed):
    rng = random.Random(seed)
    queries = list(rel_payload["queries"])
    rng.shuffle(queries)
    return (
        {**rel_payload, "queries": queries[n_test:]},
        {**rel_payload, "queries": queries[:n_test]},
    )


def run_one(seed, herald_n, steps, cutoff, relevance_path, n_test, log_dir):
    import space.train_sbert_fock as ts

    rel_payload = json.loads(relevance_path.read_text("utf-8"))
    train_rel, test_rel = make_split(rel_payload, n_test=n_test, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        train_p = Path(tmp) / "rel_train.json"
        test_p  = Path(tmp) / "rel_test.json"
        train_p.write_text(json.dumps(train_rel, indent=2))
        test_p.write_text(json.dumps(test_rel, indent=2))
        import argparse as _ap
        args = _ap.Namespace(
            cutoff=cutoff, herald_n=herald_n,
            steps=steps, lr=1e-2, weight_decay=1e-3, temperature=0.5,
            negatives=8, clip=1.0, seed=seed, log_every=50,
            relevance=str(train_p),
            eval_train_rel=str(train_p), eval_test_rel=str(test_p),
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ts.train(args)
        out = buf.getvalue()
    log_path = log_dir / f"fock_seed{seed}_n{herald_n}.log"
    log_path.write_text(out, encoding="utf-8")
    summary_line = next(
        (l for l in out.splitlines() if l.startswith("SUMMARY_JSON=")), ""
    )
    summary = json.loads(summary_line.split("=", 1)[1]) if summary_line else {}
    train_agg = summary.get("train/fock", {})
    test_agg  = summary.get("test/fock", {})
    return {
        "seed": seed,
        "herald_n": herald_n,
        "cutoff": cutoff,
        "train_ndcg10":  train_agg.get("ndcg@10",   float("nan")),
        "test_ndcg10":   test_agg.get("ndcg@10",    float("nan")),
        "train_recall10":train_agg.get("recall@10", float("nan")),
        "test_recall10": test_agg.get("recall@10",  float("nan")),
        "train_recall1": train_agg.get("recall@1",  float("nan")),
        "test_recall1":  test_agg.get("recall@1",   float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relevance", type=Path, default=ROOT / "eval" / "relevance_expanded.json")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--steps",  type=int, default=200)
    ap.add_argument("--cutoff", type=int, default=6)
    ap.add_argument("--herald-ns", type=int, nargs="+", default=[1, 0],
                    help="Ancilla outcomes to test. herald_n=0 keeps state Gaussian; herald_n=1 makes it non-Gaussian.")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "sweep_fock_results.csv")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "sweep_fock_logs")
    args = ap.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        for hn in args.herald_ns:
            print(f"\n{'='*72}\nseed={seed}  herald_n={hn}\n{'='*72}")
            row = run_one(seed=seed, herald_n=hn, steps=args.steps,
                          cutoff=args.cutoff, relevance_path=args.relevance,
                          n_test=args.n_test, log_dir=args.log_dir)
            print(f"  → train n10={row['train_ndcg10']:.3f}  test n10={row['test_ndcg10']:.3f}")
            results.append(row)

    fieldnames = list(results[0].keys())
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {len(results)} rows → {args.out_csv}")

    import statistics
    def stat(rs, key):
        vals = [r[key] for r in rs if r[key] == r[key]]
        return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals))

    print("\nAggregates:")
    for hn in args.herald_ns:
        rs = [r for r in results if r["herald_n"] == hn]
        m, s, n = stat(rs, "test_ndcg10")
        label = "non-Gaussian (herald=1)" if hn == 1 else "Gaussian (herald=0)" if hn == 0 else f"herald={hn}"
        print(f"  {label:>26}: test nDCG@10 = {m:.3f} ± {s:.3f}   (n={n})")


if __name__ == "__main__":
    main()
