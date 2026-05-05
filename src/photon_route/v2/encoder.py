"""Path-A inference encoder: text → Gaussian state via a per-word param table.

Pure forward pass: word lookup → (r, phi_s, alpha_q, alpha_p) → symplectic
Sgate + displacement shift, repeated per word with mode cycling, then a
terminal beam splitter. All operations are numpy linear algebra on
2N-vectors and 2N×2N matrices; no torch, no autograd, no Monte Carlo.

Ordering is qqpp throughout (q_0,...,q_{N-1}, p_0,...,p_{N-1}). This
matches both thewalrus.symplectic (gate matrices) and SF
BaseGaussianState.means()/cov() (xxpp/qqpp), so the returned (mu, sigma)
are byte-identical to the v1 SF path under SHA-init.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from photon_route.v2 import ops

ParamsFn = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class EncoderConfig:
    n_modes: int = 2


def vacuum_state(n_modes: int) -> tuple[np.ndarray, np.ndarray]:
    return ops.vacuum(n_modes)


def dict_params_fn(
    table: Mapping[str, np.ndarray],
    unk: np.ndarray | None = None,
) -> ParamsFn:
    unk_vec = np.zeros(4, dtype=np.float64) if unk is None else np.asarray(unk, np.float64)

    def fn(word: str) -> np.ndarray:
        v = table.get(word)
        return np.asarray(v, np.float64) if v is not None else unk_vec

    return fn


class Encoder:
    """Numpy-only Gaussian encoder. Same gate algebra as v1, learnable params."""

    def __init__(self, params_fn: ParamsFn, config: EncoderConfig | None = None):
        self.params_fn = params_fn
        self.config = config or EncoderConfig()

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """text -> (mu, sigma) in qqpp ordering, matching SF state.means()/cov()."""
        words = [w for w in text.lower().split() if w]
        if not words:
            raise ValueError("empty text")
        n = self.config.n_modes
        mu, sigma = ops.vacuum(n)

        for i, word in enumerate(words):
            r, phi_s, alpha_q, alpha_p = (float(x) for x in self.params_fn(word))
            k = i % n
            S = ops.sgate_matrix_qqpp(n, k, r, phi_s)
            mu, sigma = ops.apply_symplectic(mu, sigma, S)
            mu = ops.apply_displacement_qqpp(mu, n, k, alpha_q, alpha_p)

        if n >= 2:
            theta, phi_bs = ops.terminal_bs_params(len(words))
            BS = ops.bsgate_matrix_qqpp(n, 0, 1, theta, phi_bs)
            mu, sigma = ops.apply_symplectic(mu, sigma, BS)

        return mu, sigma

    def encode_corpus(
        self, items: list[str] | list[tuple[str, dict]]
    ) -> list[tuple[np.ndarray, np.ndarray, str, dict]]:
        out = []
        for item in items:
            if isinstance(item, str):
                text, meta = item, {}
            else:
                text, meta = item
            mu, sigma = self.encode(text)
            out.append((mu, sigma, text, meta))
        return out
