"""Symplectic gate operations, hbar=2.

All arithmetic is in **qqpp ordering** (q_0, q_1, ..., q_{N-1},
p_0, p_1, ..., p_{N-1}) — the convention shared by thewalrus.symplectic
and SF's BaseGaussianState.means()/cov() (which SF documents as xxpp).

Why this matters:
  * thewalrus.symplectic.{squeezing, beam_splitter, expand} produce
    qqpp matrices.
  * SF's `state.cov()` / `state.means()` return qqpp (xxpp).
  * v2 matches v1 (which uses SF state.cov() output) for the Step-0
    equivalence invariant — same ordering throughout, no permutation.

The xpxp helpers below are kept for callers that need to interop with
formats that use xpxp (q_0, p_0, q_1, p_1, ...), but the encoder does
not use them.

All matrices are 2N × 2N float arrays. They act on state via:
    mu    -> S @ mu
    sigma -> S @ sigma @ S.T
"""

from __future__ import annotations

import math

import numpy as np
from thewalrus.symplectic import beam_splitter, expand, squeezing


def vacuum(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """Vacuum: mu = 0, sigma = I_{2N}. Same in qqpp and xpxp."""
    return np.zeros(2 * n_modes), np.eye(2 * n_modes)


def sgate_matrix_qqpp(n_modes: int, mode: int, r: float, phi: float) -> np.ndarray:
    return expand(squeezing(r, phi), [mode], n_modes)


def bsgate_matrix_qqpp(
    n_modes: int, mode_a: int, mode_b: int, theta: float, phi: float
) -> np.ndarray:
    return expand(beam_splitter(theta, phi), [mode_a, mode_b], n_modes)


def apply_symplectic(
    mu: np.ndarray, sigma: np.ndarray, S: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return S @ mu, S @ sigma @ S.T


def apply_displacement_qqpp(
    mu: np.ndarray, n_modes: int, mode: int, alpha_q: float, alpha_p: float
) -> np.ndarray:
    """Dgate as shift on means in qqpp: q_mode at index `mode`, p_mode at `mode + N`."""
    out = mu.copy()
    out[mode] += alpha_q
    out[mode + n_modes] += alpha_p
    return out


def qqpp_to_xpxp_perm(n_modes: int) -> np.ndarray:
    """Permutation array: xpxp[i] = qqpp[perm[i]].

    For N=2: perm = [0, 2, 1, 3]
        xpxp index 0 (q_0) ← qqpp index 0
        xpxp index 1 (p_0) ← qqpp index 2
        xpxp index 2 (q_1) ← qqpp index 1
        xpxp index 3 (p_1) ← qqpp index 3
    """
    perm = np.empty(2 * n_modes, dtype=int)
    for k in range(n_modes):
        perm[2 * k] = k
        perm[2 * k + 1] = k + n_modes
    return perm


def to_xpxp(
    mu: np.ndarray, sigma: np.ndarray, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Permute (mu, sigma) from internal qqpp to public xpxp."""
    perm = qqpp_to_xpxp_perm(n_modes)
    return mu[perm], sigma[np.ix_(perm, perm)]


def terminal_bs_params(n_words: int) -> tuple[float, float]:
    """Length-based terminal BS schedule, mirrors v1.encode.encode_one exactly.

    Path-A keeps this length-only schedule (audit point #2 deferred to v2.1).
    """
    theta = (n_words % 16) * (math.pi / 16)
    phi_bs = ((n_words * 7) % 16) * (math.pi / 16)
    return theta, phi_bs
