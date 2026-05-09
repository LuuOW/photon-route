"""Train the v2 photon-route encoder. Runs on the HF Space (16 GB CPU),
NEVER on the meridian-vm (notepad, 952 MB).

What this does:

  1. Load the eval corpus (arxiv IDs + frozen manifest) and fetch abstracts
     on demand via eval.fetch (cached under $PHOTON_EVAL_CACHE).
  2. Build a vocab over query words + abstract words (lowercase, whitespace).
  3. Initialize an `nn.Embedding(|vocab|, 4)` table with SHA-256 derived
     params so the untrained model already matches the v1 baseline.
  4. Forward pass = the v2 encoder, but in torch:
       per-word: r = MAX_SQ * sigmoid(raw[0])   in [0, MAX_SQ]
                 phi_s = atan2(sin(raw[1]), cos(raw[1]))    [-π, π]
                 alpha_q = sqrt(2*hbar) * tanh(raw[2])      [-2, 2]  hbar=2
                 alpha_p = sqrt(2*hbar) * tanh(raw[3])
       gates: thewalrus-faithful symplectic squeeze + displacement,
              terminal length-driven beam splitter, all qqpp ordering.
  5. Loss = InfoNCE over Bhattacharyya-coefficient similarities, with
     photon-number regularization (||μ||² + ||V - I||²).
  6. Dump weights as `weights.npz` keyed by word → (r, phi_s, alpha_q,
     alpha_p), directly consumable by photon_route.v2 numpy encoder.

Why Bhattacharyya, not Banchi-Braunstein-Pirandola: BBP fidelity for
Gaussian states involves a matrix square root that is expensive and
numerically unstable to backprop through for tiny states. The
Bhattacharyya coefficient between two Gaussian distributions is a
smooth, closed-form, differentiable lower bound on their quantum
fidelity — fine as a training surrogate; eval-time scoring still uses
real BBP fidelity via thewalrus.quantum.fidelity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.fetch import fetch_all, verify_against_manifest  # noqa: E402

# Constants pinned to v1 conventions
N_MODES = 2
MAX_SQUEEZE = 0.5
MAX_DISPLACE = 1.0
HBAR = 2.0
DGATE_PREFACTOR = math.sqrt(2.0 * HBAR)  # 2.0


# ---------------------------------------------------------------------------
# vocab + SHA-256 initialization
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return [w for w in text.lower().split() if w]


def sha_init_raw(word: str) -> np.ndarray:
    """Return the four pre-reparametrization "raw" logits whose
    sigmoid/tanh gives the v1 SHA-256 (r, phi_s, alpha_q, alpha_p).

    sigmoid(raw_r)   * MAX_SQ        = r       (in [0, MAX_SQ])
    sigmoid(raw_phi) * 2π            = phi_s   (in [0, 2π))
    tanh(raw_aq)     * sqrt(2 hbar)  = alpha_q (in [-2, 2])
    tanh(raw_ap)     * sqrt(2 hbar)  = alpha_p

    Since sigmoid and tanh are invertible we can compute the exact raw
    that reproduces the SHA-init point in parameter space.
    """
    h = hashlib.sha256(word.encode("utf-8")).digest()
    parts = [(int.from_bytes(h[i * 8 : (i + 1) * 8], "big") % 10**9) / 1e9 for i in range(4)]
    r       = parts[0] * MAX_SQUEEZE
    phi_s   = parts[1] * 2 * math.pi
    d_mag   = parts[2] * MAX_DISPLACE
    d_phase = parts[3] * 2 * math.pi
    alpha_q = DGATE_PREFACTOR * d_mag * math.cos(d_phase)
    alpha_p = DGATE_PREFACTOR * d_mag * math.sin(d_phase)
    # invert: sigmoid^-1(x) = log(x / (1 - x))
    eps = 1e-6
    p_r   = max(eps, min(1 - eps, r / MAX_SQUEEZE))
    p_phi = max(eps, min(1 - eps, phi_s / (2 * math.pi)))
    raw_r   = math.log(p_r / (1 - p_r))
    raw_phi = math.log(p_phi / (1 - p_phi))
    # tanh^-1(x) = 0.5 * log((1+x)/(1-x))
    t_q = max(-1 + eps, min(1 - eps, alpha_q / DGATE_PREFACTOR))
    t_p = max(-1 + eps, min(1 - eps, alpha_p / DGATE_PREFACTOR))
    raw_aq = 0.5 * math.log((1 + t_q) / (1 - t_q))
    raw_ap = 0.5 * math.log((1 + t_p) / (1 - t_p))
    return np.array([raw_r, raw_phi, raw_aq, raw_ap], dtype=np.float64)


def reparametrize(raw: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """raw: (..., 4) → (r, phi_s, alpha_q, alpha_p) in their physical
    ranges, all differentiable wrt raw."""
    r       = MAX_SQUEEZE  * torch.sigmoid(raw[..., 0])
    phi_s   = (2 * math.pi) * torch.sigmoid(raw[..., 1])
    alpha_q = DGATE_PREFACTOR * torch.tanh(raw[..., 2])
    alpha_p = DGATE_PREFACTOR * torch.tanh(raw[..., 3])
    return r, phi_s, alpha_q, alpha_p


# ---------------------------------------------------------------------------
# torch ports of the symplectic gates (qqpp), faithful to thewalrus
# ---------------------------------------------------------------------------

def _eye2N(n: int, ref: Tensor) -> Tensor:
    return torch.eye(2 * n, dtype=ref.dtype, device=ref.device)


def squeezing_qqpp(n: int, k: int, r: Tensor, phi: Tensor) -> Tensor:
    """Single-mode squeeze on mode k of n, qqpp ordering.

    Block = [[cosh r - sinh r cos φ, -sinh r sin φ],
             [-sinh r sin φ,          cosh r + sinh r cos φ]]
    placed at (k, k+n).
    """
    S = _eye2N(n, r).clone()
    cr, sr = torch.cosh(r), torch.sinh(r)
    cp, sp = torch.cos(phi), torch.sin(phi)
    S[k,     k    ] = cr - sr * cp
    S[k,     k + n] = -sr * sp
    S[k + n, k    ] = -sr * sp
    S[k + n, k + n] = cr + sr * cp
    return S


def beam_splitter_qqpp(n: int, a: int, b: int, theta: Tensor, phi: Tensor) -> Tensor:
    """Two-mode BS via thewalrus interferometer construction in qqpp.

    U = [[cos θ, -e^{-iφ} sin θ], [e^{iφ} sin θ, cos θ]]
    S = [[Re U, -Im U], [Im U, Re U]] embedded at (a, b, a+n, b+n).
    """
    S = _eye2N(n, theta).clone()
    ct, st = torch.cos(theta), torch.sin(theta)
    cp, sp = torch.cos(phi),   torch.sin(phi)
    # Re(U) = [[ct, -st cp], [st cp, ct]];  Im(U) = [[0, st sp], [st sp, 0]]
    # qq block (top-left of qqpp embedding):
    S[a,     a    ] = ct;        S[a,     b    ] = -st * cp
    S[b,     a    ] = st * cp;   S[b,     b    ] = ct
    # qp block (top-right):
    S[a,     a + n] = 0.0;        S[a,     b + n] = -st * sp
    S[b,     a + n] = -st * sp;   S[b,     b + n] = 0.0
    # pq block (bottom-left, =Im U):
    S[a + n, a    ] = 0.0;        S[a + n, b    ] = st * sp
    S[b + n, a    ] = st * sp;    S[b + n, b    ] = 0.0
    # pp block (bottom-right, =Re U):
    S[a + n, a + n] = ct;         S[a + n, b + n] = -st * cp
    S[b + n, a + n] = st * cp;    S[b + n, b + n] = ct
    return S


def encode_torch(
    text: str,
    vocab: dict[str, int],
    embedding: nn.Embedding,
    n_modes: int = N_MODES,
) -> tuple[Tensor, Tensor]:
    words = tokenize(text)
    if not words:
        raise ValueError("empty text")
    raw = embedding(torch.tensor(
        [vocab.get(w, 0) for w in words], dtype=torch.long, device=embedding.weight.device,
    ))  # (T, 4)
    r, phi_s, alpha_q, alpha_p = reparametrize(raw)

    mu = torch.zeros(2 * n_modes, dtype=embedding.weight.dtype, device=embedding.weight.device)
    sigma = _eye2N(n_modes, embedding.weight)

    for i in range(len(words)):
        k = i % n_modes
        S = squeezing_qqpp(n_modes, k, r[i], phi_s[i])
        mu = S @ mu
        sigma = S @ sigma @ S.T
        # Cartesian Dgate: shift means at (k, k+N)
        shift = torch.zeros_like(mu)
        shift[k]           = alpha_q[i]
        shift[k + n_modes] = alpha_p[i]
        mu = mu + shift

    if n_modes >= 2:
        # length-only schedule, NOT learned (matches v1)
        L = len(words)
        theta = torch.tensor((L % 16) * (math.pi / 16),
                             dtype=embedding.weight.dtype, device=embedding.weight.device)
        phi_bs = torch.tensor(((L * 7) % 16) * (math.pi / 16),
                              dtype=embedding.weight.dtype, device=embedding.weight.device)
        BS = beam_splitter_qqpp(n_modes, 0, 1, theta, phi_bs)
        mu = BS @ mu
        sigma = BS @ sigma @ BS.T

    return mu, sigma


# ---------------------------------------------------------------------------
# Bhattacharyya surrogate fidelity (Gaussian-Gaussian)
# ---------------------------------------------------------------------------

def bhattacharyya_distance(
    mu_a: Tensor, sg_a: Tensor, mu_b: Tensor, sg_b: Tensor, ridge: float = 1e-3,
) -> Tensor:
    """Bhattacharyya distance D_B between two Gaussians (means + covs).

    D_B = (1/8) Δμᵀ V⁻¹ Δμ + (1/2) log(det V / sqrt(det A · det B)),
    with V = (A + B)/2, A = Σ_a + ridge·I, B = Σ_b + ridge·I.
    Lower = more similar; ≥ 0 for proper SPD inputs.
    Returned clamped to [0, 50] for downstream softmax/exp stability.

    Used as a contrastive *logit* (-D / temperature) — cheaper and far
    more numerically stable than F_B = exp(-D), which underflows for
    well-separated Gaussians and amplifies slogdet noise.
    """
    d = sg_a.shape[0]
    eye = torch.eye(d, dtype=sg_a.dtype, device=sg_a.device)
    A = sg_a + ridge * eye
    B = sg_b + ridge * eye
    V = 0.5 * (A + B)
    delta = mu_a - mu_b
    sol = torch.linalg.solve(V, delta)
    quad = (delta * sol).sum()
    log_det_V = torch.linalg.slogdet(V)[1]
    log_det_A = torch.linalg.slogdet(A)[1]
    log_det_B = torch.linalg.slogdet(B)[1]
    D = 0.125 * quad + 0.5 * (log_det_V - 0.5 * (log_det_A + log_det_B))
    return torch.clamp(D, min=0.0, max=50.0)


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rel_path  = Path(args.relevance) if args.relevance else ROOT / "eval" / "relevance.json"
    cids_path = ROOT / "eval" / "corpus_ids.json"
    man_path  = ROOT / "eval" / "manifest.json"

    relevance = json.loads(rel_path.read_text("utf-8"))["queries"]
    ids       = json.loads(cids_path.read_text("utf-8"))["ids"]

    print(f"[train] fetching {len(ids)} abstracts...", flush=True)
    abstracts = fetch_all(ids)
    bad = verify_against_manifest(abstracts, man_path)
    if bad:
        sys.exit(f"manifest mismatch on {len(bad)} ids: {list(bad)[:3]}")

    # vocab over query + abstract tokens
    words: set[str] = set()
    for q in relevance:
        words.update(tokenize(q["query"]))
    for t in abstracts.values():
        words.update(tokenize(t))
    vocab = {w: i for i, w in enumerate(sorted(words))}
    print(f"[train] vocab |V| = {len(vocab)}", flush=True)

    # float64 throughout — slogdet of a near-singular squeezed-state covariance
    # in float32 emits NaN that propagates through cross_entropy. Float64 absorbs
    # the conditioning loss from many sequential Sgate compositions.
    embedding = nn.Embedding(len(vocab), 4, dtype=torch.float64)
    with torch.no_grad():
        for w, i in vocab.items():
            embedding.weight[i] = torch.from_numpy(sha_init_raw(w))

    optim = torch.optim.AdamW(embedding.parameters(), lr=args.lr, weight_decay=1e-4)

    rng = np.random.default_rng(args.seed)
    queries = [(q["query"], set(q["relevant_ids"])) for q in relevance]
    all_ids = list(ids)

    t_start = time.time()
    for step in range(1, args.steps + 1):
        optim.zero_grad()

        loss_sum = torch.zeros((), dtype=torch.float64)
        loss_components = {"info_nce": 0.0, "photon": 0.0}

        # Encode each abstract once per step (was 9× per query before — 54
        # encodings/step → 26). Weights change every step so the cache is
        # per-step only, not amortized across steps.
        doc_states = {a: encode_torch(t, vocab, embedding) for a, t in abstracts.items()}

        for query, rel_set in queries:
            mu_q, sg_q = encode_torch(query, vocab, embedding)

            # one positive (random pick from relevant set)
            pos_id = rng.choice(sorted(rel_set))
            mu_p, sg_p = doc_states[pos_id]

            # negatives: K random non-relevant ids
            negs = rng.choice(
                [i for i in all_ids if i not in rel_set],
                size=min(args.negatives, len(all_ids) - len(rel_set)),
                replace=False,
            )
            d_pos = bhattacharyya_distance(mu_q, sg_q, mu_p, sg_p)
            d_negs = torch.stack([
                bhattacharyya_distance(mu_q, sg_q, *doc_states[n]) for n in negs
            ])
            # Use distance directly as a (negative) logit. Smaller D → larger
            # logit → higher probability for that class. Standard contrastive
            # form: cross_entropy(-D / temp, target=positive).
            logits = -torch.cat([d_pos.unsqueeze(0), d_negs]) / args.temperature
            target = torch.zeros((), dtype=torch.long)
            ce = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
            loss_sum = loss_sum + ce

            loss_components["info_nce"] += ce.item()

        loss_sum = loss_sum / len(queries)

        # photon-number regularization across the vocab — keeps |α|, r small
        r_v       = MAX_SQUEEZE * torch.sigmoid(embedding.weight[:, 0])
        alpha_q_v = DGATE_PREFACTOR * torch.tanh(embedding.weight[:, 2])
        alpha_p_v = DGATE_PREFACTOR * torch.tanh(embedding.weight[:, 3])
        n_photons = (torch.sinh(r_v) ** 2 + 0.25 * (alpha_q_v**2 + alpha_p_v**2) / HBAR).mean()
        photon_pen = args.photon_lambda * n_photons
        loss_components["photon"] = photon_pen.item()

        total = loss_sum + photon_pen
        total.backward()
        torch.nn.utils.clip_grad_norm_(embedding.parameters(), max_norm=args.clip)
        optim.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"[train] step {step}/{args.steps}  "
                f"info_nce={loss_components['info_nce']/len(queries):.4f}  "
                f"photon={loss_components['photon']:.4f}  "
                f"total={total.item():.4f}  "
                f"elapsed={time.time()-t_start:.1f}s",
                flush=True,
            )

    # serialize: store the *physical* params (r, phi_s, alpha_q, alpha_p)
    # so the v2 numpy encoder can consume them directly with no torch.
    embedding.eval()
    with torch.no_grad():
        raw = embedding.weight.detach()
        r, phi_s, alpha_q, alpha_p = reparametrize(raw)
        params = torch.stack([r, phi_s, alpha_q, alpha_p], dim=-1).cpu().numpy().astype(np.float64)

    words_array = np.array(sorted(vocab, key=vocab.get), dtype=object)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        words=words_array,
        params=params,
        unk=np.zeros(4, dtype=np.float64),
        meta=np.array(json.dumps({
            "n_modes": N_MODES,
            "max_squeeze": MAX_SQUEEZE,
            "max_displace": MAX_DISPLACE,
            "vocab_size": len(vocab),
            "training_steps": args.steps,
            "lr": args.lr,
            "temperature": args.temperature,
            "photon_lambda": args.photon_lambda,
            "negatives_per_query": args.negatives,
            "seed": args.seed,
        }), dtype=object),
    )
    print(f"[train] wrote {out_path}  vocab_size={len(vocab)}  params shape={params.shape}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "weights.npz")
    ap.add_argument("--relevance", type=str, default=None,
                    help="path to alternate relevance.json (e.g. for held-out splits)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-3)
    # D-scale logits: with D in [0, 50], temp=0.1 made -D/temp logits up to
    # -500, exploding cross_entropy + gradients. temp=5 keeps logit magnitudes
    # in a sensible range (~0-10) so AdamW can converge instead of oscillate.
    ap.add_argument("--temperature", type=float, default=5.0)
    ap.add_argument("--photon-lambda", type=float, default=1e-2)
    ap.add_argument("--negatives", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
