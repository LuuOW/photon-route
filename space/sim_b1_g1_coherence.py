"""B1 sim — does g^(1)-style coherence time τ_c discriminate candidates
better than meridian's current 3-bin Shannon entropy `cross_domain`?

Loudon eq 3.1.3: g^(1)(τ) = ⟨E*(t) E(t+τ)⟩ / ⟨|E|²⟩.
For a chaotic source, |g^(1)(τ)| decays exponentially with characteristic
τ_c = (∫|g^(1)(τ)|² dτ).

Treat each candidate's keyword stream as a chaotic light source where
each token at position t is a "wavetrain at frequency ω_token". The
autocorrelation of one-hot token vectors gives an effective τ_c that
scales with vocabulary diversity.

This is a self-contained synthetic-data sim. No external corpus / no
cloud compute required. Runs in <1 s.
"""
from __future__ import annotations

import math
import numpy as np
from collections import Counter


def cross_domain_proxy(tokens: list[str], systems: dict[str, set[str]]) -> float:
    """Mirror meridian's existing computation: Shannon entropy / log(3)
    over hits in {forge, signal, mind} term lists, normalized to [0, 1]."""
    affinity = {sys: 0 for sys in systems}
    for t in tokens:
        for sys, terms in systems.items():
            if t in terms:
                affinity[sys] += 1
    total = sum(affinity.values()) or 1
    probs = [n / total for n in affinity.values() if n > 0]
    H = -sum(p * math.log(p) for p in probs)
    return H / math.log(3) if H else 0.0


def coherence_time(tokens: list[str], window: int = 8) -> float:
    """Empirical g^(1)-style coherence time of a token stream.

    Treat the sequence as a discrete-time signal where each token is a
    distinct mode. g^(1)(τ) = (# matched-token pairs at offset τ) /
    (# matched at τ=0). τ_c = sum_{τ≥1} |g^(1)(τ)|² up to a window.

    Pure-stdlib, normalised so τ_c ∈ [0, window].
    """
    n = len(tokens)
    if n < 2:
        return 0.0
    g0 = sum(1 for t in tokens) or 1  # τ=0 normalisation = total length
    tau_c = 0.0
    for tau in range(1, min(window, n)):
        matches = sum(1 for i in range(n - tau) if tokens[i] == tokens[i + tau])
        gtau = matches / g0
        tau_c += gtau * gtau
    return tau_c


# ─── Synthetic candidates with realistic body lengths ──────────────────────
SYSTEMS = {
    "forge":  {"build", "compile", "deploy", "ci", "container", "image", "binary",
               "docker", "kubernetes", "package", "release", "monorepo"},
    "signal": {"data", "stream", "ingest", "pipeline", "etl", "kafka", "queue",
               "throughput", "latency", "broker", "subscriber", "publish"},
    "mind":   {"llm", "embed", "embedding", "model", "transformer", "agent",
               "reasoning", "prompt", "context", "rag", "fine", "tune"},
}


def make_candidate(label: str, vocab_pool: list[str], length: int = 250,
                   alpha: float = 1.0, seed: int = 0) -> list[str]:
    """Generate length tokens drawn Zipfian from vocab_pool. alpha controls
    head heaviness; alpha=1.0 ≈ thermal; alpha→∞ ≈ heavy concentrated."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / (np.arange(1, len(vocab_pool) + 1) ** alpha)
    weights /= weights.sum()
    return list(rng.choice(vocab_pool, size=length, p=weights))


def main():
    # 9 archetypes spanning body lengths and topical vs scattered patterns.
    forge_terms  = sorted(SYSTEMS["forge"])
    signal_terms = sorted(SYSTEMS["signal"])
    mind_terms   = sorted(SYSTEMS["mind"])
    cross_terms  = forge_terms + signal_terms + mind_terms
    cases = [
        ("focused-forge",       forge_terms,  300, 1.0, 1),
        ("focused-signal",      signal_terms, 300, 1.0, 2),
        ("focused-mind",        mind_terms,   300, 1.0, 3),
        ("cross-forge-signal",  forge_terms + signal_terms, 300, 1.0, 4),
        ("cross-mind-signal",   mind_terms + signal_terms,  300, 1.0, 5),
        ("cross-three-systems", cross_terms,  300, 1.0, 6),
        ("scattered-cross",     cross_terms,  300, 0.5, 7),  # less Zipfian, more uniform
        ("very-narrow",         forge_terms[:3], 300, 2.0, 8),  # 3 dominant words
        ("very-broad",          cross_terms,  300, 0.3, 9),  # near-uniform
    ]

    print(f"{'archetype':>22}  {'len':>5}  {'cross_domain':>13}  {'τ_c (g^(1))':>12}")
    print("-" * 64)
    rows = []
    for label, pool, length, alpha, seed in cases:
        toks = make_candidate(label, pool, length=length, alpha=alpha, seed=seed)
        cd = cross_domain_proxy(toks, SYSTEMS)
        tc = coherence_time(toks)
        print(f"{label:>22}  {length:>5}  {cd:>13.3f}  {tc:>12.3f}")
        rows.append((label, cd, tc))

    print("\nDiscrimination check (variance across archetypes, higher = better signal):")
    cd_vals = [r[1] for r in rows]
    tc_vals = [r[2] for r in rows]
    print(f"  cross_domain:  std = {np.std(cd_vals):.3f}, range = [{min(cd_vals):.3f}, {max(cd_vals):.3f}]")
    print(f"  τ_c (g^(1)):   std = {np.std(tc_vals):.3f}, range = [{min(tc_vals):.3f}, {max(tc_vals):.3f}]")

    # CV (coefficient of variation) — higher = more discriminative on its own scale
    cd_cv = np.std(cd_vals) / max(np.mean(cd_vals), 1e-9)
    tc_cv = np.std(tc_vals) / max(np.mean(tc_vals), 1e-9)
    print(f"\n  CV (std/mean): cross_domain={cd_cv:.3f}  τ_c={tc_cv:.3f}")
    if tc_cv > cd_cv * 1.2:
        print("  → τ_c is more discriminative than cross_domain — B1 stands.")
    elif tc_cv < cd_cv * 0.8:
        print("  → τ_c is LESS discriminative than cross_domain — B1 fails.")
    else:
        print("  → τ_c and cross_domain have similar discrimination — B1 is a wash.")


if __name__ == "__main__":
    main()
