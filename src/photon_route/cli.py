"""Command-line entry: `photon-route "<query>" [--top-k N]`."""

from __future__ import annotations

import argparse
import json
import sys

from photon_route.corpus import load_fixture
from photon_route.encode import encode_corpus
from photon_route.retrieve import rank_against


def main() -> int:
    p = argparse.ArgumentParser(prog="photon-route")
    p.add_argument("query", help="text to rank fixture documents against")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args()

    corpus = encode_corpus(load_fixture())
    results = rank_against(corpus, args.query, top_k=args.top_k)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "rank": i + 1,
                        "score": r.score,
                        "text": r.doc.doc.text,
                        "meta": r.doc.doc.meta,
                    }
                    for i, r in enumerate(results)
                ],
                indent=2,
            )
        )
    else:
        for i, r in enumerate(results, 1):
            print(f"{i:>2}. {r.score:.4f}  {r.doc.doc.text}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
