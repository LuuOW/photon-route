"""5-split × 2-mode sweep over the SBERT-backed photon-route trainer.

Designed to run on cloud CI (e.g. GitHub Actions ubuntu-latest, free tier),
NOT locally. Output is a CSV of (split_seed, no_squeeze, train_ndcg10, test_ndcg10, ...)
that gets uploaded as a workflow artifact for the user to read.

For each random split seed:
  1. Pick 2 of the eval queries as held-out test, rest as train.
  2. Train SBERTPhoton (full vs --no-squeeze).
  3. Evaluate on train and test.
  4. Append result row.

The point: we want error bars, not a point estimate. If the +30% nDCG@10
the SBERT-backed run got on one specific 4/2 split is real, it should hold
across multiple random splits. If it varies wildly, the headline was a
2-query coincidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def make_split(rel_payload: dict, n_test: int, seed: int) -> tuple[dict, dict]:
    """Returns (train_relevance, test_relevance) as separate JSON-able dicts."""
    rng = random.Random(seed)
    queries = list(rel_payload["queries"])
    rng.shuffle(queries)
    test = queries[:n_test]
    train = queries[n_test:]
    return (
        {**rel_payload, "queries": train},
        {**rel_payload, "queries": test},
    )


def run_one(seed: int, no_squeeze: bool, steps: int, relevance_path: Path,
            n_test: int, log_dir: Path) -> dict:
    """Train + eval one configuration; return summary dict for the CSV row."""
    import space.train_sbert as ts

    rel_payload = json.loads(relevance_path.read_text("utf-8"))
    train_rel, test_rel = make_split(rel_payload, n_test=n_test, seed=seed)

    with tempfile.TemporaryDirectory() as tmp:
        train_p = Path(tmp) / "rel_train.json"
        test_p  = Path(tmp) / "rel_test.json"
        train_p.write_text(json.dumps(train_rel, indent=2))
        test_p.write_text(json.dumps(test_rel, indent=2))

        # Build a Namespace mimicking train_sbert's CLI args so we can call
        # train(args) directly without subprocess. Faster + captures Python
        # exceptions cleanly.
        import argparse as _ap
        args = _ap.Namespace(
            steps=steps, lr=1e-2, weight_decay=1e-3, temperature=2.0,
            negatives=8, clip=1.0, seed=seed, log_every=50,
            relevance=str(train_p),
            eval_train_rel=str(train_p), eval_test_rel=str(test_p),
            no_squeeze=no_squeeze,
        )

        # Capture stdout to recover SUMMARY_JSON line.
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ts.train(args)
        out = buf.getvalue()

    # Persist log
    log_path = log_dir / f"seed{seed}_nosqz{int(no_squeeze)}.log"
    log_path.write_text(out, encoding="utf-8")

    summary_line = next(
        (l for l in out.splitlines() if l.startswith("SUMMARY_JSON=")), ""
    )
    summary = json.loads(summary_line.split("=", 1)[1]) if summary_line else {}

    def g(key, metric_key):
        return summary.get(f"{key}/{metric_key}", {})

    row = {"seed": seed, "no_squeeze": int(no_squeeze)}
    for split in ("train", "test"):
        for metric in ("gaussian", "photon_prob"):
            agg = g(split, metric)
            for m in ("ndcg@10", "recall@10", "recall@1"):
                short = m.replace("@", "").replace("recall", "r").replace("ndcg", "n")
                row[f"{split}_{metric}_{short}"] = agg.get(m, float("nan"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relevance", type=Path, default=ROOT / "eval" / "relevance_expanded.json",
                    help="Default to the title-expanded set so each split has more train signal.")
    ap.add_argument("--seeds",   type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-test",  type=int, default=4,
                    help="Held-out test queries per split. With expanded relevance (26 q), 4 test is ~15%.")
    ap.add_argument("--steps",   type=int, default=200)
    ap.add_argument("--out-csv", type=Path, default=ROOT / "sweep_results.csv")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "sweep_logs")
    args = ap.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        for no_squeeze in [False, True]:
            print(f"\n{'='*72}\nseed={seed} no_squeeze={no_squeeze}\n{'='*72}")
            row = run_one(
                seed=seed, no_squeeze=no_squeeze, steps=args.steps,
                relevance_path=args.relevance, n_test=args.n_test, log_dir=args.log_dir,
            )
            print(f"  → gaussian:    train n10={row['train_gaussian_n10']:.3f}  test n10={row['test_gaussian_n10']:.3f}")
            print(f"     photon_prob: train n10={row['train_photon_prob_n10']:.3f}  test n10={row['test_photon_prob_n10']:.3f}")
            results.append(row)

    # Write CSV
    fieldnames = list(results[0].keys())
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {len(results)} rows → {args.out_csv}")

    # Aggregate stats
    import statistics
    def stat(rows, key):
        vals = [r[key] for r in rows]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    print("\nAggregates over seeds:")
    for ns in [False, True]:
        rows = [r for r in results if r["no_squeeze"] == int(ns)]
        if not rows:
            continue
        label = "no-squeeze" if ns else "full"
        for metric in ("gaussian", "photon_prob"):
            m, s = stat(rows, f"test_{metric}_n10")
            print(f"  {label:>10}/{metric:>11}: test nDCG@10 = {m:.3f} ± {s:.3f}   (n={len(rows)})")


if __name__ == "__main__":
    main()
