"""SHA-256 → (r, phi_s, alpha_q, alpha_p) mapping for v1 numerical equivalence.

This is *not* the long-term plan — it's a deterministic param source that
mirrors v1.encode.encode_one's polar parametrization, converted to the
Cartesian-displacement form that v2 uses internally. Feeding this through
the v2 encoder produces (mu, sigma) numerically identical to running v1
through the SF gaussian backend (verified by tests/test_v2_equivalence.py).

The reason this exists:
  * Step 0 of the v2 plan: prove the v2 inference path is structurally
    equivalent to v1 *before* changing semantics. If the encoder is
    correct under SHA-init, then the only thing learning has to do is
    move points in the (r, phi_s, alpha_q, alpha_p) space.
  * Ablation baseline: after training, compare trained-v2 vs SHA-init-v2
    using the same encoder. Differences are attributable purely to the
    parameter table.

Constants mirror v1's encode.py defaults:
  MAX_SQUEEZE  = 0.5
  MAX_DISPLACE = 1.0

The Dgate prefactor sqrt(2*hbar) = 2 (hbar=2 in SF) converts polar
(d_mag, d_phase) → Cartesian (alpha_q, alpha_p):
  alpha_q = 2 * d_mag * cos(d_phase)
  alpha_p = 2 * d_mag * sin(d_phase)
"""

from __future__ import annotations

import functools
import hashlib
import math

import numpy as np

MAX_SQUEEZE = 0.5
MAX_DISPLACE = 1.0
# sqrt(2 * hbar) with hbar=2 (SF gaussian backend default)
_DGATE_PREFACTOR = math.sqrt(2.0 * 2.0)


@functools.lru_cache(maxsize=None)
def sha_params_v1_compat(word: str) -> np.ndarray:
    """Same SHA-256 derivation as v1.encode._word_params, returned as
    (r, phi_s, alpha_q, alpha_p) ready for the Cartesian-displacement v2 encoder.

    Cached because SHA-256 is deterministic and the same words appear many
    times across queries and corpus passes.
    """
    h = hashlib.sha256(word.encode("utf-8")).digest()
    parts = []
    for i in range(4):
        chunk = int.from_bytes(h[i * 8 : (i + 1) * 8], "big")
        parts.append((chunk % 10**9) / 1e9)
    r = parts[0] * MAX_SQUEEZE
    phi_s = parts[1] * 2 * math.pi
    d_mag = parts[2] * MAX_DISPLACE
    d_phase = parts[3] * 2 * math.pi
    alpha_q = _DGATE_PREFACTOR * d_mag * math.cos(d_phase)
    alpha_p = _DGATE_PREFACTOR * d_mag * math.sin(d_phase)
    return np.array([r, phi_s, alpha_q, alpha_p], dtype=np.float64)
