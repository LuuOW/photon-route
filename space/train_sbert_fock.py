"""A3-Real — non-Gaussian SBERT-photon trainer.

Architecture:
    text → frozen SBERT 384d → Linear(384, 6) → photonic params
        → 2-mode (signal + ancilla) cutoff-D Fock-basis encoder
        → unitary U = D_signal(α) S_signal(r,φ) S_2(τ,θ)  applied to |0,0⟩
        → project ancilla onto |1⟩ (single-photon herald)
        → normalised pure state |ψ_sig⟩ ∈ ℂ^D, NON-GAUSSIAN
    score(q, d) = |⟨ψ_q,sig | ψ_d,sig⟩|²

Why this is genuinely new vs space/train_sbert.py: with α small and r mild,
the heralded-on-|1⟩ signal mode contains a single-photon contribution. A
single-photon Fock state has Wigner-negative regions — non-Gaussian. The
similarity |⟨ψ_q|ψ_d⟩|² is *not* representable as a Gaussian-RBF kernel
on any finite-d projection of the inputs (Sim 1's negative result for the
Gaussian path).

Loss: InfoNCE on -log(score) (i.e. score is the affinity logit). Cached
SBERT features. Same eval/relevance as the Gaussian trainer for direct
head-to-head numbers.
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

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_DIM = 384
HERALD_N = 1               # ancilla outcome we herald on; single-photon → genuinely non-Gaussian


# ─── Truncated bosonic operators in Fock basis ────────────────────────────
def annihilation_op(D: int) -> Tensor:
    """a |n⟩ = √n |n-1⟩  (D-dim truncation; lossy at the top)."""
    a = torch.zeros(D, D, dtype=torch.complex128)
    for n in range(1, D):
        a[n - 1, n] = math.sqrt(n)
    return a


def kron(A: Tensor, B: Tensor) -> Tensor:
    return torch.kron(A, B)


# ─── Generators of the unitaries ────────────────────────────────────────────
def displace_generator(a: Tensor, alpha: Tensor) -> Tensor:
    """G_D = α a† − α* a; applied as exp(G_D) gives D(α). α is complex scalar."""
    return alpha * a.conj().T - torch.conj(alpha) * a


def squeeze_generator(a: Tensor, zeta: Tensor) -> Tensor:
    """G_S = (1/2)(ζ* a² − ζ a†²); applied as exp(G_S) gives S(ζ). ζ = r e^{iφ}."""
    a2 = a @ a
    return 0.5 * (torch.conj(zeta) * a2 - zeta * a2.conj().T)


def two_mode_squeeze_generator(a: Tensor, b: Tensor, xi: Tensor) -> Tensor:
    """G_{TMS} = ξ* a b − ξ a† b†. Acts on joint signal⊗ancilla space.
    Inputs are full-dim joint operators a, b (e.g. a = a_signal ⊗ I_anc)."""
    return torch.conj(xi) * (a @ b) - xi * (a.conj().T @ b.conj().T)


# ─── Encoder ────────────────────────────────────────────────────────────────
class SBERTPhotonFock(nn.Module):
    """SBERT → Linear(384, 6) → 2-mode Fock encoder → herald → 1-mode pure state.

    The 6 outputs decompose to (αq, αp, r, φ_s, τ, θ):
      α = αq + i·αp                      (signal displacement)
      ζ = r e^{iφ_s},  r ∈ [0, 0.5]      (signal squeezing)
      ξ = τ e^{iθ},    τ ∈ [0, 0.5]      (two-mode squeezing toward ancilla)
    """

    def __init__(self, cutoff: int = 6, max_squeeze: float = 0.5,
                 max_displace: float = 1.5, herald_n: int = HERALD_N):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.D = cutoff
        self.max_sq = max_squeeze
        self.max_disp = max_displace
        self.herald_n = herald_n
        if herald_n >= cutoff:
            raise ValueError(f"herald_n={herald_n} must be < cutoff={cutoff}")
        self.sbert = SentenceTransformer(SBERT_MODEL_NAME)
        for p in self.sbert.parameters():
            p.requires_grad = False
        # Trainable surface
        self.proj = nn.Linear(SBERT_DIM, 6, dtype=torch.float32)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        # Pre-compute truncated bosonic operators (constants — no grad needed)
        a = annihilation_op(cutoff)
        I = torch.eye(cutoff, dtype=torch.complex128)
        # Joint-space (signal ⊗ ancilla): a_s = a ⊗ I, b_a = I ⊗ a
        self.register_buffer("a_signal_full", kron(a, I))
        self.register_buffer("b_anc_full",     kron(I, a))
        self.register_buffer("a_signal_local", a)  # for solo signal-side gates if ever needed
        # Initial vacuum |0,0⟩ in joint Fock basis (D² vector)
        psi0 = torch.zeros(cutoff * cutoff, dtype=torch.complex128)
        psi0[0] = 1.0  # index (0, 0) → flat 0
        self.register_buffer("vacuum", psi0)

    def encode_features(self, texts: list[str]) -> Tensor:
        with torch.no_grad():
            emb = self.sbert.encode(
                texts, normalize_embeddings=True, convert_to_numpy=False,
                show_progress_bar=False,
            )
            emb = torch.stack([e for e in emb]) if isinstance(emb, list) else emb
            return emb.to(torch.float32).cpu()

    def state_from_features(self, feat: Tensor) -> Tensor:
        """Returns the heralded signal-mode state |ψ_sig⟩ ∈ ℂ^D, normalized.
        Shape: (D,) complex128.
        """
        out = self.proj(feat)  # (6,) float32
        out = out.to(torch.float64)
        # decompose with bounded reparametrizations
        alpha_q = self.max_disp * torch.tanh(out[0])
        alpha_p = self.max_disp * torch.tanh(out[1])
        r       = self.max_sq  * torch.sigmoid(out[2])
        phi_s   = (2 * math.pi) * torch.sigmoid(out[3])
        tau     = self.max_sq  * torch.sigmoid(out[4])
        theta   = (2 * math.pi) * torch.sigmoid(out[5])
        # Build complex parameters
        alpha = torch.complex(alpha_q, alpha_p)
        zeta  = torch.complex(r * torch.cos(phi_s), r * torch.sin(phi_s))
        xi    = torch.complex(tau * torch.cos(theta), tau * torch.sin(theta))
        # Generators in joint space
        G_TMS = two_mode_squeeze_generator(
            self.a_signal_full, self.b_anc_full, xi,
        )
        G_S   = squeeze_generator(self.a_signal_full, zeta)
        G_D   = displace_generator(self.a_signal_full, alpha)
        # Apply unitaries: |ψ⟩ = D · S · S_2 · |0,0⟩
        U_TMS = torch.linalg.matrix_exp(G_TMS)
        U_S   = torch.linalg.matrix_exp(G_S)
        U_D   = torch.linalg.matrix_exp(G_D)
        psi = U_TMS @ self.vacuum
        psi = U_S   @ psi
        psi = U_D   @ psi
        # Project ancilla onto |herald_n⟩.  Joint flat index = signal*D + ancilla.
        # Pick rows where ancilla == herald_n: rows = [signal*D + herald_n for signal in 0..D-1]
        D = self.D
        idx = torch.arange(D, device=psi.device, dtype=torch.long) * D + self.herald_n
        psi_sig = psi[idx]
        # Normalize (heralding probability is the squared norm; we drop it)
        norm = torch.linalg.vector_norm(psi_sig)
        psi_sig = psi_sig / torch.clamp(norm, min=1e-12)
        return psi_sig

    def state_from_text(self, text: str) -> Tensor:
        feat = self.encode_features([text])[0]
        return self.state_from_features(feat)


def overlap_squared(psi_a: Tensor, psi_b: Tensor) -> Tensor:
    """|⟨ψ_a|ψ_b⟩|². Pure-state fidelity since both are heralded pure."""
    inner = torch.vdot(psi_a, psi_b)
    return (inner.real ** 2 + inner.imag ** 2)


def recall_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranked, relevant, k):
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, a in enumerate(ranked[:k], start=1) if a in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal > 0 else float("nan")


def evaluate(model, abstracts, ids, queries, ks=(1, 3, 5, 10)) -> dict:
    model.eval()
    with torch.no_grad():
        doc_states = {a: model.state_from_text(t) for a, t in abstracts.items()}
    rows = []
    for q in queries:
        with torch.no_grad():
            psi_q = model.state_from_text(q["query"])
        scored = []
        for a in ids:
            psi_d = doc_states[a]
            scored.append((float(overlap_squared(psi_q, psi_d).item()), a))
        scored.sort(key=lambda x: -x[0])
        ranked = [a for _, a in scored]
        rel = set(q["relevant_ids"])
        row = {"query": q["query"], "ranked": ranked[: max(ks)]}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, rel, k)
            row[f"ndcg@{k}"]   = ndcg_at_k(ranked, rel, k)
        rows.append(row)
    agg = {f"recall@{k}": float(np.mean([r[f"recall@{k}"] for r in rows])) for k in ks}
    agg.update({f"ndcg@{k}": float(np.mean([r[f"ndcg@{k}"] for r in rows])) for k in ks})
    return {"per_query": rows, "aggregate": agg}


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rel_path  = Path(args.relevance) if args.relevance else ROOT / "eval" / "relevance.json"
    cids_path = ROOT / "eval" / "corpus_ids.json"
    man_path  = ROOT / "eval" / "manifest.json"
    train_relevance = json.loads(rel_path.read_text("utf-8"))["queries"]
    ids = json.loads(cids_path.read_text("utf-8"))["ids"]
    print(f"[fock] fetching {len(ids)} abstracts...", flush=True)
    abstracts = fetch_all(ids)
    bad = verify_against_manifest(abstracts, man_path)
    if bad:
        sys.exit(f"manifest mismatch: {list(bad)[:3]}")

    model = SBERTPhotonFock(cutoff=args.cutoff, herald_n=args.herald_n)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fock] cutoff={args.cutoff}  herald_n={args.herald_n}  trainable={n_trainable}", flush=True)

    # Cache features
    print(f"[fock] caching SBERT features ({len(abstracts)} docs + "
          f"{len(train_relevance)} queries)...", flush=True)
    doc_feats = {a: model.encode_features([t])[0] for a, t in abstracts.items()}
    query_feats = {q["query"]: model.encode_features([q["query"]])[0]
                    for q in train_relevance}

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    queries = [(q["query"], set(q["relevant_ids"])) for q in train_relevance]

    t0 = time.time()
    for step in range(1, args.steps + 1):
        optim.zero_grad()
        # Recompute every state (proj weights change every step). Cached
        # SBERT features feed straight into state_from_features.
        doc_psi = {a: model.state_from_features(doc_feats[a]) for a in abstracts}

        loss_sum = torch.zeros((), dtype=torch.float64)
        for query_text, rel_set in queries:
            psi_q = model.state_from_features(query_feats[query_text])
            pos_id = rng.choice(sorted(rel_set))
            psi_pos = doc_psi[pos_id]
            negs = rng.choice(
                [i for i in ids if i not in rel_set],
                size=min(args.negatives, len(ids) - len(rel_set)), replace=False,
            )
            f_pos = overlap_squared(psi_q, psi_pos)
            f_negs = torch.stack([overlap_squared(psi_q, doc_psi[n]) for n in negs])
            # InfoNCE: log P(pos) = log(f_pos / Σ f). Fidelities are in [0,1] so use as logits directly.
            logits = torch.cat([f_pos.unsqueeze(0), f_negs]) / args.temperature
            ce = F.cross_entropy(logits.unsqueeze(0), torch.zeros((), dtype=torch.long).unsqueeze(0))
            loss_sum = loss_sum + ce
        loss_sum = loss_sum / len(queries)
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=args.clip,
        )
        optim.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"[fock] step {step}/{args.steps}  loss={loss_sum.item():.4f}  "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

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
        summary[f"{label}/fock"] = report["aggregate"]
    print(f"\nSUMMARY_JSON={json.dumps(summary)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff",       type=int, default=6,
                    help="Fock-basis truncation per mode. Joint dim = cutoff².")
    ap.add_argument("--herald-n",     type=int, default=HERALD_N,
                    help="Ancilla photon-number outcome to project onto.")
    ap.add_argument("--steps",        type=int, default=200)
    ap.add_argument("--lr",           type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--temperature",  type=float, default=0.5)
    ap.add_argument("--negatives",    type=int, default=8)
    ap.add_argument("--clip",         type=float, default=1.0)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--log-every",    type=int, default=20)
    ap.add_argument("--relevance",      type=str, default=None)
    ap.add_argument("--eval-train-rel", type=str, default=None)
    ap.add_argument("--eval-test-rel",  type=str, default=None)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
