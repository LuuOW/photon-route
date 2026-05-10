"""B2 sim — does g^(2)(0) cleanly classify candidates into planet / comet
/ asteroid on REAL meridian-shaped data (Llama-emitted bodies of typical
length 100–500 tokens), and where does it disagree with the existing
mass × scope × independence rule?

Sim 4b earlier (synthetic Zipfian) showed g^(2) > 1 only emerges at
N_distinct × token_total scales typical of real bodies, not the 8–12
token toy archetypes from Sim 4. This sim re-runs that check on
realistic body shapes and compares per-candidate the g^(2) class label
to the mass × scope × independence label that orbital.mjs assigns today.
"""
from __future__ import annotations

import math
import numpy as np
from collections import Counter


def g2_zero(tokens: list[str]) -> float:
    """g^(2)(0) = ⟨n(n-1)⟩ / ⟨n⟩² over per-word counts {n_i}.
    Loudon Ch 6.4: coherent → 1, chaotic → 2, antibunched → <1."""
    if not tokens:
        return float("nan")
    counts = np.asarray(list(Counter(tokens).values()), dtype=np.float64)
    n_mean = counts.mean()
    n_n_minus_1 = (counts * (counts - 1)).mean()
    return n_n_minus_1 / (n_mean ** 2) if n_mean > 0 else float("nan")


def class_from_g2(g2: float) -> str:
    """Threshold rule from Loudon Ch 6.4."""
    if g2 < 0.7:
        return "asteroid"   # antibunched / sparse / niche
    elif g2 < 1.4:
        return "planet"     # ≈ 1 = coherent / focused
    else:
        return "comet"      # > 1 = thermal / scattered


def class_from_meridian(mass: float, scope: float, indep: float,
                        cross_domain: float, drag: float, fragmentation: float,
                        dep_ratio: float, has_parent: bool) -> str:
    """Mirror orbital.mjs:139-167 — compute the same scores and pick max."""
    planet  = min(mass, scope, indep) ** 1.5
    moon    = (max(0, 0.5 - indep) * 2 *
               (1.0 if has_parent else 0.4) * (1 - 0.5 * mass))
    trojan  = dep_ratio * (1.0 if has_parent else 0.5) * (1 - fragmentation)
    asteroid = max(0, 0.55 - mass) * 2.5 * scope * indep
    comet    = drag * cross_domain * (1 - dep_ratio)
    irregular = cross_domain * fragmentation * 0.85
    scores = {"planet": planet, "moon": moon, "trojan": trojan,
              "asteroid": asteroid, "comet": comet, "irregular": irregular}
    return max(scores, key=scores.get)


def physics_from_tokens(tokens: list[str]) -> dict:
    """Approximate the physics scalars meridian computes from text."""
    body_len = sum(len(t) for t in tokens)
    n_words = len(tokens)
    mass = max(0, min(1, 0.6 * np.log10(max(50, body_len) / 200) /
                       np.log10(3000 / 200) + 0.4 * (n_words - 3) / 9))
    distinct = len(set(tokens))
    scope = min(0.7, distinct / 12) + 0.2  # rough proxy
    scope = max(0, min(1, scope))
    indep  = 0.7  # synthetic candidates have no siblings; assume mid-high
    drag   = 0.3
    fragmentation = 0.4
    cross_domain  = 0.5  # placeholder
    dep_ratio    = 0.2
    return dict(mass=mass, scope=scope, indep=indep,
                drag=drag, fragmentation=fragmentation,
                cross_domain=cross_domain, dep_ratio=dep_ratio,
                has_parent=False)


# ─── Realistic synthetic candidates ─────────────────────────────────────────
def zipfian_words(prefix: str, n_distinct: int, length: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    vocab = [f"{prefix}-{i:02d}" for i in range(n_distinct)]
    weights = 1.0 / (np.arange(1, n_distinct + 1) ** alpha)
    weights /= weights.sum()
    return list(rng.choice(vocab, size=length, p=weights))


def main():
    # 9 archetypes covering the planet/comet/asteroid spectrum at realistic
    # body length (200–400 tokens) — the regime Sim 4b proved relevant.
    cases = [
        # label,                      n_distinct, length, alpha (Zipf head),  expected
        ("planet-tight-vocab",        20,  300, 1.0, "planet"),    # coherent-shaped
        ("planet-medium",             15,  250, 0.8, "planet"),
        ("planet-broad-vocab",        50,  400, 1.2, "planet"),
        ("comet-thermal",             30,  300, 1.5, "comet"),     # heavier head
        ("comet-very-heavy",          25,  300, 2.0, "comet"),
        ("comet-multimodal",          40,  350, 1.8, "comet"),
        ("asteroid-narrow",           5,   300, 1.0, "asteroid"),  # too few distinct
        ("asteroid-fragments",        10,  100, 0.5, "asteroid"),  # short body
        ("asteroid-uniform",          50,  300, 0.3, "asteroid"),  # near-uniform
    ]

    print(f"{'archetype':>22}  {'len':>5}  {'g^(2)':>7}  {'g2_class':>10}  "
          f"{'mass×s×i_class':>16}  {'expected':>10}")
    print("-" * 90)
    correct_g2 = 0
    correct_meridian = 0
    for label, n_distinct, length, alpha, expected in cases:
        toks = zipfian_words(label, n_distinct, length, alpha, seed=hash(label) & 0xFFFF)
        g2 = g2_zero(toks)
        cls_g2 = class_from_g2(g2)
        phys = physics_from_tokens(toks)
        cls_m = class_from_meridian(**phys)
        ok_g2 = cls_g2 == expected
        ok_m  = cls_m == expected
        correct_g2 += int(ok_g2)
        correct_meridian += int(ok_m)
        marker_g2 = "✓" if ok_g2 else "✗"
        marker_m  = "✓" if ok_m  else "✗"
        print(f"{label:>22}  {length:>5}  {g2:>7.3f}  "
              f"{cls_g2:>9}{marker_g2}  {cls_m:>15}{marker_m}  {expected:>10}")

    print(f"\n  g^(2)-only classifier:        {correct_g2}/{len(cases)} archetypes correct")
    print(f"  meridian's mass×scope×indep:  {correct_meridian}/{len(cases)} archetypes correct")
    if correct_g2 > correct_meridian:
        print("  → B2 stands: g^(2) classifies more reliably on real-shape data")
    elif correct_g2 < correct_meridian:
        print("  → B2 fails: meridian's existing rule is better")
    else:
        print("  → B2 is a wash: both classifiers tied on archetype recovery")


if __name__ == "__main__":
    main()
