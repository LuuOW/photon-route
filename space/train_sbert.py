"""SBERT-backed photon-route encoder. The language model does language;
the photonic gates do the structured projection.

Architecture:
    text → frozen SentenceTransformer (all-MiniLM-L6-v2, 384-d, no grad)
         → Linear(384 → 4N + 2N)  [trainable]
         → 4N displacement outputs (αq, αp per mode) + 2N squeezing outputs (r, φ per mode)
         → photonic gates (Sgate + Dgate per mode)
         → 2N-d Gaussian state (μ, σ) at hbar=2

Trainable surface:
    Linear(384 → 6N) ≈ 384·6N + 6N params  (6 numbers per mode: αq, αp, r, φ_s, plus 2 future)
    For N=2: 384·8 + 8 = 3,080 params total
    vs word-level photon-route: |V|·4 = 5,772 (and grows with vocab)

Loss is the same InfoNCE-on-Bhattacharyya as space/train.py so the comparison
is apples-to-apples on the encoder, not the loss.

Holdout discipline: load --relevance from a file. The eval driver
(eval/run.py) does NOT support sbert weights yet; this module ships its
own evaluator alongside the trainer for now.
"""
from __future__ import annotations

import argparse
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

N_MODES = 2
HBAR = 2.0
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_DIM = 384


# ─── differentiable single-mode squeezing in qqpp ─────────────────────────────
def _eye2N(n: int, ref: Tensor) -> Tensor:
    return torch.eye(2 * n, dtype=ref.dtype, device=ref.device)


def squeezing_qqpp(n: int, k: int, r: Tensor, phi: Tensor) -> Tensor:
    S = _eye2N(n, r).clone()
    cr, sr = torch.cosh(r), torch.sinh(r)
    cp, sp = torch.cos(phi), torch.sin(phi)
    S[k,     k    ] = cr - sr * cp
    S[k,     k + n] = -sr * sp
    S[k + n, k    ] = -sr * sp
    S[k + n, k + n] = cr + sr * cp
    return S


class SBERTPhoton(nn.Module):
    """Frozen SBERT → Linear → photonic state.

    The Linear emits 6 numbers per mode (αq, αp, r, φ_s, plus 2 reserved).
    Currently 4 are used; spare dims are zero'd by their learnable weight
    converging to small values, so unused capacity self-prunes.
    """

    def __init__(self, n_modes: int = N_MODES, max_squeeze: float = 0.5,
                 max_displace: float = 1.0, no_squeeze: bool = False):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.n = n_modes
        self.max_sq = max_squeeze
        self.max_disp = max_displace
        self.no_squeeze = no_squeeze
        self.dgate_prefactor = math.sqrt(2.0 * HBAR)
        self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
        for p in self.sbert.parameters():
            p.requires_grad = False
        # float32 throughout — MPS (Apple-Silicon GPU) doesn't support float64;
        # cast to float64 at the eval-fidelity boundary (numpy + thewalrus).
        # Squeezing magnitudes are bounded ≤ 0.5 so the covariance stays
        # well-conditioned and float32 slogdet is numerically safe.
        self.proj = nn.Linear(SBERT_DIM, 4 * n_modes, dtype=torch.float32)
        # Small-random init (NOT zeros). Zero init puts every text at the
        # same vacuum state, so all pairwise distances equal zero, gradients
        # vanish, and loss stays at log(N+1) = saddle point forever.
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def encode_features(self, texts: list[str]) -> Tensor:
        """Run frozen SBERT, return (B, 384) float32 on CPU."""
        with torch.no_grad():
            emb = self.sbert.encode(
                texts, normalize_embeddings=True, convert_to_numpy=False,
                show_progress_bar=False,
            )
            emb = torch.stack([e for e in emb]) if isinstance(emb, list) else emb
            return emb.to(torch.float32).cpu()

    def state_from_text(self, text: str) -> tuple[Tensor, Tensor]:
        feat = self.encode_features([text])           # (1, 384)
        out  = self.proj(feat)[0]                     # (4N,)
        # Decompose: per-mode (αq, αp, raw_r, raw_phi).
        # tanh-bound squeezing magnitude to [0, max_sq]; phi free.
        per_mode = out.view(self.n, 4)
        alpha_q = self.dgate_prefactor * torch.tanh(per_mode[:, 0])
        alpha_p = self.dgate_prefactor * torch.tanh(per_mode[:, 1])
        if self.no_squeeze:
            r     = torch.zeros(self.n, dtype=feat.dtype)
            phi_s = torch.zeros(self.n, dtype=feat.dtype)
        else:
            r     = self.max_sq * torch.sigmoid(per_mode[:, 2])
            phi_s = (2 * math.pi) * torch.sigmoid(per_mode[:, 3])

        mu = torch.zeros(2 * self.n, dtype=feat.dtype)
        sigma = _eye2N(self.n, feat)
        for k in range(self.n):
            if not self.no_squeeze:
                S = squeezing_qqpp(self.n, k, r[k], phi_s[k])
                mu = S @ mu
                sigma = S @ sigma @ S.T
            shift = torch.zeros_like(mu)
            shift[k]            = alpha_q[k]
            shift[k + self.n]   = alpha_p[k]
            mu = mu + shift
        return mu, sigma


def bhattacharyya_distance(mu_a, sg_a, mu_b, sg_b, ridge: float = 1e-3) -> Tensor:
    d = sg_a.shape[0]
    eye = torch.eye(d, dtype=sg_a.dtype, device=sg_a.device)
    A = sg_a + ridge * eye
    B = sg_b + ridge * eye
    V = 0.5 * (A + B)
    delta = mu_a - mu_b
    quad = (delta * torch.linalg.solve(V, delta)).sum()
    log_det_V = torch.linalg.slogdet(V)[1]
    log_det_A = torch.linalg.slogdet(A)[1]
    log_det_B = torch.linalg.slogdet(B)[1]
    D = 0.125 * quad + 0.5 * (log_det_V - 0.5 * (log_det_A + log_det_B))
    return torch.clamp(D, min=0.0, max=50.0)


def gaussian_fidelity_eval(mu_a: np.ndarray, sg_a: np.ndarray,
                            mu_b: np.ndarray, sg_b: np.ndarray) -> float:
    """thewalrus closed-form for eval-time scoring (not used in loss)."""
    from thewalrus.quantum import fidelity as tw_fidelity
    f = tw_fidelity(mu_a, sg_a, mu_b, sg_b, hbar=HBAR)
    val = float(f.real if hasattr(f, "real") else f)
    return max(0.0, min(1.0, val))


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked_ids, relevant, k):
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, a in enumerate(ranked_ids[:k], start=1) if a in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else float("nan")


def evaluate(model: SBERTPhoton, abstracts, ids, queries, ks=(1, 3, 5, 10)) -> dict:
    model.eval()
    with torch.no_grad():
        doc_states = {a: model.state_from_text(t) for a, t in abstracts.items()}
    rows = []
    for q in queries:
        with torch.no_grad():
            mu_q, sg_q = model.state_from_text(q["query"])
        scored = []
        for a in ids:
            mu_d, sg_d = doc_states[a]
            f = gaussian_fidelity_eval(
                mu_q.cpu().numpy().astype(np.float64), sg_q.cpu().numpy().astype(np.float64),
                mu_d.cpu().numpy().astype(np.float64), sg_d.cpu().numpy().astype(np.float64),
            )
            scored.append((f, a))
        scored.sort(key=lambda x: -x[0])
        ranked_ids = [a for _, a in scored]
        rel = set(q["relevant_ids"])
        row = {"query": q["query"], "ranked": ranked_ids[: max(ks)]}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, rel, k)
            row[f"ndcg@{k}"]   = ndcg_at_k(ranked_ids, rel, k)
        rows.append(row)
    aggregate = {f"recall@{k}": float(np.mean([r[f"recall@{k}"] for r in rows])) for k in ks}
    aggregate.update({f"ndcg@{k}": float(np.mean([r[f"ndcg@{k}"] for r in rows])) for k in ks})
    return {"per_query": rows, "aggregate": aggregate}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rel_path  = Path(args.relevance) if args.relevance else ROOT / "eval" / "relevance.json"
    cids_path = ROOT / "eval" / "corpus_ids.json"
    man_path  = ROOT / "eval" / "manifest.json"

    train_relevance = json.loads(rel_path.read_text("utf-8"))["queries"]
    ids = json.loads(cids_path.read_text("utf-8"))["ids"]
    print(f"[sbert] fetching {len(ids)} abstracts...", flush=True)
    abstracts = fetch_all(ids)
    bad = verify_against_manifest(abstracts, man_path)
    if bad:
        sys.exit(f"manifest mismatch: {list(bad)[:3]}")

    model = SBERTPhoton(n_modes=N_MODES, no_squeeze=args.no_squeeze)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[sbert] trainable params = {n_trainable}  no_squeeze={args.no_squeeze}", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    queries = [(q["query"], set(q["relevant_ids"])) for q in train_relevance]

    t0 = time.time()
    for step in range(1, args.steps + 1):
        optim.zero_grad()
        # Encode all docs once per step (weights change every step).
        doc_states = {a: model.state_from_text(t) for a, t in abstracts.items()}

        loss_sum = torch.zeros((), dtype=torch.float32)
        for query_text, rel_set in queries:
            mu_q, sg_q = model.state_from_text(query_text)
            pos_id = rng.choice(sorted(rel_set))
            mu_p, sg_p = doc_states[pos_id]
            negs = rng.choice(
                [i for i in ids if i not in rel_set],
                size=min(args.negatives, len(ids) - len(rel_set)), replace=False,
            )
            d_pos = bhattacharyya_distance(mu_q, sg_q, mu_p, sg_p)
            d_negs = torch.stack([bhattacharyya_distance(mu_q, sg_q, *doc_states[n]) for n in negs])
            logits = -torch.cat([d_pos.unsqueeze(0), d_negs]) / args.temperature
            ce = F.cross_entropy(logits.unsqueeze(0), torch.zeros((), dtype=torch.long).unsqueeze(0))
            loss_sum = loss_sum + ce
        loss_sum = loss_sum / len(queries)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=args.clip,
        )
        optim.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"[sbert] step {step}/{args.steps}  loss={loss_sum.item():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

    # final eval against whichever relevance file the user asks for
    eval_paths = []
    if args.eval_train_rel:
        eval_paths.append(("train", Path(args.eval_train_rel)))
    if args.eval_test_rel:
        eval_paths.append(("test",  Path(args.eval_test_rel)))
    if not eval_paths:
        eval_paths.append(("all", rel_path))
    summary = {}
    for label, p in eval_paths:
        rels = json.loads(p.read_text("utf-8"))["queries"]
        report = evaluate(model, abstracts, ids, rels)
        print(f"\n=== {label.upper()} EVAL ({len(rels)} queries) ===")
        for r in report["per_query"]:
            cells = " ".join(f"{m}={r[m]:.3f}" for m in r if m.startswith(("recall", "ndcg")))
            print(f"  {r['query'][:48]:<48s}  {cells}")
        print("aggregate: " + " ".join(f"{m}={report['aggregate'][m]:.3f}" for m in report["aggregate"]))
        summary[label] = report["aggregate"]
    # Sentinel line for downstream parsers (run_sweep.py).
    print(f"\nSUMMARY_JSON={json.dumps(summary)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--negatives", type=int, default=8)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--relevance",      type=str, default=None,
                    help="training relevance.json (e.g. /tmp/rel_train.json)")
    ap.add_argument("--eval-train-rel", type=str, default=None,
                    help="optional separate train-eval set for in-sample numbers")
    ap.add_argument("--eval-test-rel",  type=str, default=None,
                    help="optional held-out eval set for generalization numbers")
    ap.add_argument("--no-squeeze", action="store_true",
                    help="ablation: force r=0 (displacement-only). Tests whether the squeezing layer specifically pays.")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
