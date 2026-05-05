"""Fetch arxiv abstracts on demand, verify against the frozen manifest.

The repo deliberately does not commit abstract text — see project memory
(no_arxiv_storage). Only IDs + SHA-256 hashes are versioned. Runners
fetch text at eval time and abort on hash mismatch so retrieval numbers
remain reproducible against the exact snapshot.

Cache layout: $PHOTON_EVAL_CACHE (default ~/.cache/photon-route/eval/),
one .txt file per arxiv ID. The cache is content-addressed implicitly:
manifest.hashes[id] is the authoritative SHA-256.

Network: scrapes the og:description meta tag from arxiv.org/abs/<id>
HTML pages (CDN-cached via Google Frontend, no per-IP rate limit in
practice). The official export.arxiv.org/api endpoint is rate-limited
to ~1 req / 3s and easily 429s during eval runs, so it isn't used.
A browser-like User-Agent is required: arxiv.org returns HTTP 406 to
non-browser UAs from datacenter IPs (caught HF Space build failure
2026-05-05).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

ARXIV_ABS = "https://arxiv.org/abs/"
DEFAULT_CACHE = Path(os.environ.get("PHOTON_EVAL_CACHE", str(Path.home() / ".cache/photon-route/eval")))
_OG_DESC = re.compile(
    r'<meta\s+(?:property|name)="og:description"\s+content="([^"]*)"',
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_one(arxiv_id: str, timeout: float = 30.0, max_retries: int = 4) -> str:
    """Fetch one abstract via abs-page scrape. Returns normalized abstract text."""
    url = ARXIV_ABS + arxiv_id
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 32.0)
                continue
            raise last_err  # type: ignore[misc]
    m = _OG_DESC.search(body)
    if not m:
        raise RuntimeError(f"og:description not found for {arxiv_id}")
    raw = html.unescape(m.group(1))
    return _normalize(raw)


def fetch_all(
    ids: Iterable[str],
    cache_dir: Path | None = None,
    sleep_between: float = 0.5,
) -> dict[str, str]:
    """Return {id: abstract}. Cached entries are read from disk; missing ones
    are scraped one by one from arxiv.org/abs/<id> with a small delay so we
    don't hammer the CDN."""
    cache_dir = cache_dir or DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = list(ids)
    out: dict[str, str] = {}
    missing: list[str] = []
    for i in ids:
        p = cache_dir / f"{i}.txt"
        if p.exists():
            out[i] = p.read_text("utf-8")
        else:
            missing.append(i)
    for j, arxiv_id in enumerate(missing):
        text = _fetch_one(arxiv_id)
        (cache_dir / f"{arxiv_id}.txt").write_text(text, encoding="utf-8")
        out[arxiv_id] = text
        if j + 1 < len(missing):
            time.sleep(sleep_between)
    return out


def verify_against_manifest(
    abstracts: dict[str, str], manifest_path: Path
) -> dict[str, str]:
    """Returns {} on success, or {id: actual_hash} for mismatches."""
    manifest = json.loads(manifest_path.read_text("utf-8"))
    expected = manifest.get("hashes", {})
    if not expected:
        return {}
    bad: dict[str, str] = {}
    for arxiv_id, text in abstracts.items():
        actual = sha256_text(text)
        if expected.get(arxiv_id) != actual:
            bad[arxiv_id] = actual
    return bad


def freeze_manifest(
    abstracts: dict[str, str],
    manifest_path: Path,
    source_url: str = ARXIV_ABS,
) -> None:
    """Write a fresh manifest; intended to be run once to lock the snapshot."""
    payload = {
        "schema_version": 1,
        "description": json.loads(manifest_path.read_text("utf-8")).get(
            "description", ""
        ) if manifest_path.exists() else "",
        "snapshot_taken_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_source": source_url,
        "hash_algo": "sha256",
        "hashes": {k: sha256_text(v) for k, v in sorted(abstracts.items())},
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path(__file__).parent / "corpus_ids.json")
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.json")
    ap.add_argument("--freeze", action="store_true",
                    help="Overwrite manifest with hashes of currently fetched abstracts")
    args = ap.parse_args()

    ids = json.loads(args.corpus.read_text("utf-8"))["ids"]
    abstracts = fetch_all(ids)
    print(f"fetched {len(abstracts)} / {len(ids)} abstracts")
    if args.freeze:
        freeze_manifest(abstracts, args.manifest)
        print(f"wrote manifest with {len(abstracts)} hashes -> {args.manifest}")
    else:
        bad = verify_against_manifest(abstracts, args.manifest)
        if bad:
            print(f"HASH MISMATCH on {len(bad)} ids: {bad}")
            raise SystemExit(2)
        print("manifest verified" if json.loads(args.manifest.read_text("utf-8")).get("hashes") else "manifest empty (run with --freeze)")
