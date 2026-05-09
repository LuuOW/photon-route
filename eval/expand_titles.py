"""Expand the eval relevance set with title-as-query pairs.

For each arXiv ID in corpus_ids.json, fetch og:title from the abstract
page (same scrape pattern as eval.fetch but a different meta tag), and
emit one query whose only relevant document is that paper.

Output: eval/relevance_expanded.json — original 6 multi-positive queries
plus 20 single-positive title queries = 26 total. Increases trainer
signal 4× without any human labeling.

This script does NOT touch the existing relevance.json. It writes a
sibling file the trainer / eval harness opt into via --relevance.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARXIV_ABS = "https://arxiv.org/abs/"
_OG_TITLE = re.compile(
    r'<meta\s+(?:property|name)="og:title"\s+content="([^"]*)"',
    re.IGNORECASE,
)
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_arxiv_prefix(title: str) -> str:
    """og:title comes back as '[2304.12717] Quantum natural language ...';
    strip the '[id]' prefix so the query is just the paper title."""
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", title).strip()


def fetch_title(arxiv_id: str, timeout: float = 30.0) -> str:
    url = ARXIV_ABS + arxiv_id
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    m = _OG_TITLE.search(body)
    if not m:
        raise RuntimeError(f"og:title not found for {arxiv_id}")
    return _strip_arxiv_prefix(_normalize(html.unescape(m.group(1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus",       type=Path, default=ROOT / "eval" / "corpus_ids.json")
    ap.add_argument("--in-relevance", type=Path, default=ROOT / "eval" / "relevance.json")
    ap.add_argument("--out",          type=Path, default=ROOT / "eval" / "relevance_expanded.json")
    ap.add_argument("--cache",        type=Path, default=Path.home() / ".cache" / "photon-route" / "titles")
    ap.add_argument("--sleep",        type=float, default=0.5)
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    ids = json.loads(args.corpus.read_text("utf-8"))["ids"]
    base = json.loads(args.in_relevance.read_text("utf-8"))

    titles = {}
    for j, i in enumerate(ids):
        cache_path = args.cache / f"{i}.title"
        if cache_path.exists():
            titles[i] = cache_path.read_text("utf-8").strip()
            continue
        t = fetch_title(i)
        cache_path.write_text(t, encoding="utf-8")
        titles[i] = t
        print(f"[{j+1:2d}/{len(ids)}] {i}: {t[:60]}")
        if j + 1 < len(ids):
            time.sleep(args.sleep)

    title_queries = [
        {"query": titles[i], "relevant_ids": [i], "kind": "title"}
        for i in ids
    ]
    out_payload = {
        **base,
        "queries": [
            *[{**q, "kind": "topical"} for q in base["queries"]],
            *title_queries,
        ],
    }
    args.out.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(out_payload['queries'])} queries → {args.out}")


if __name__ == "__main__":
    main()
