# photon-route design notes

A research diary, not a spec. Update as the work moves.

## Why this exists

Test the hypothesis that **continuous-variable photonic state geometry**
— text encoded as Gaussian states (squeezing + displacement on
beam-splitter-mixed bosonic modes), retrieval via closed-form
Gaussian-state fidelity — produces qualitatively different ranking
behavior than the discrete-variable QNLP approach (qrouter) and than
classical dense embeddings, on small specialized corpora.

This is the CV sister to qrouter. The two share a worldview
("compositional retrieval via quantum overlap") but not a substrate.
qrouter rides the lambeq + IQP + qubit-Hilbert path; photon-route rides
the Strawberry Fields + Gaussian + Fock path. The point of running both
is the comparison: where does CV win, where does DV win, where do they
tie, and what does the energy delta look like.

## Scope at day 1

Pre-arXiv toy fixture (the same 5 short quant-ph one-liners as qrouter)
that the demo can rank without any network or HF download. 2 modes
(N_MODES=2) to keep simulation latency invisible. Gaussian-only — no
non-Gaussian elements yet. Hash-bound parameters so the geometry is
deterministic and word-identity-encoding without learning.

## Encoding

Per word w:

    h = SHA-256(w)
    r        = (h[0:8]  / 2^64) * MAX_SQUEEZE       # squeezing magnitude
    phi      = (h[8:16] / 2^64) * 2pi               # squeezing phase
    alpha_re = (h[16:24] / 2^64 - 0.5) * 2 * MAX_DISPLACE
    alpha_im = (h[24:32] / 2^64 - 0.5) * 2 * MAX_DISPLACE

Words are placed onto modes alternately by position (i % N_MODES),
applying Sgate(r, phi) then Dgate(alpha_re + i*alpha_im). After all
words, a BSgate with sentence-length-derived angles mixes the modes.

The Gaussian backend evaluates the program; the resulting GaussianState
exposes mean vector mu (length 2*N_MODES) and covariance matrix V
(2*N_MODES x 2*N_MODES). This pair fully specifies the state.

## Retrieval

For two Gaussian states (mu_q, V_q) and (mu_d, V_d), fidelity is
computed via thewalrus.quantum.fidelity, which implements the
Banchi-Braunstein-Pirandola 2015 closed form. Symmetric, in [0, 1].

## Open design questions

- **Mode count.** N_MODES=2 is the minimum that supports a non-trivial
  beam splitter. Going higher costs O(N^2) for fidelity computation and
  more squeezing-budget overhead. Exploration: at what N does fidelity
  geometry start meaningfully separating semantically-similar from
  semantically-different sentences?

- **Squeezing budget.** MAX_SQUEEZE is conservatively 0.5 to keep
  photon numbers low and Gaussian backend numerically stable. Real
  hardware (Borealis) accepts larger squeezing but at the cost of
  detection inefficiency; choose MAX_SQUEEZE close to the hardware
  target rather than the simulator's comfort zone.

- **Compositional grammar in CV.** lambeq's pregroup-grammar →
  tensor-network pipeline does not have a direct CV analog. Active
  research areas:
    - Continuous DisCoCat (Karvonen 2023)
    - String diagrams in symmetric monoidal categories of CV systems
  For now we use the beam-splitter-of-sentence-length stand-in. This
  is the weakest link in the CV-QNLP claim and the most interesting
  place to push.

- **Gaussian classical-simulability.** Pure Gaussian programs are
  efficient to simulate classically. The day-1 demo therefore captures
  CV substrate but not quantum advantage. To escape this, Phase 1 adds
  PNR measurement on a subset of modes, conditioning the rest on the
  click pattern. The conditioned state is non-Gaussian and not
  classically efficient in general.

- **Eval set.** Same gap as qrouter — no benchmark exists for
  "CV-QNLP retrieval on quant-ph". We will build one: ~200-500
  query-doc pairs over arXiv quant-ph titles + abstracts. Once it
  exists it can score qrouter (DV), photon-route (CV), and a classical
  bge-m3 baseline on the same axis. Plot is publishable.

## Phases

### Phase 0 — Gaussian-only CV pipeline (THIS SCAFFOLD)

- 5-doc fixture, hash-bound params, 2 modes
- thewalrus closed-form fidelity ranking
- /rank endpoint live
- Pipeline verification, not a research result
- ~1-2 weeks of iteration to declare done

### Phase 1 — non-Gaussian element + training

- Add PNR measurement on m of the modes; condition the doc state on
  the click pattern. State is non-Gaussian, not classically efficient.
- Build the eval set (~200-500 query-doc pairs).
- Replace hash-bound params with optimized params (gradient-free
  optimizer over fidelity loss is fine at this scale; no autodiff
  through the SF program needed).
- Benchmark vs qrouter (DV) and bge-m3 on the same eval. Three-way
  comparison plot.
- ~2-4 weeks.

### Phase 2 — Borealis run

- Compile the same SF program for the Xanadu Cloud Borealis target.
  Boson-sampling-style retrieval: doc/query become squeezing+BS
  programs, retrieval signal is a coincidence statistic on the photon
  counts.
- Simulator vs hardware comparison. One real-photons paper.
- Decided after Phase 1 results.

### Phase 3 — photonic reservoir on a desk (the long bet)

- A physical fiber-loop reservoir as the front-end. Words modulated
  onto laser intensity, fiber loop with SOA/EDFA nonlinearity provides
  feature extraction, homodyne readout feeds a still-classical
  retrieval head.
- ~$800-1500 BoM. Open-source FPGA control (LiteX or similar).
- The point at which "AI from real-world physics" stops being a
  metaphor.

## Where this points (longer arc)

If Phase 0-1 show interesting geometry, the natural composition is:

    photonic reservoir front-end (real photons)
       -> classical readout layer
          -> fidelity / overlap as the retrieval primitive

This is end-to-end light-doing-AI-work in a way no published QNLP or
photonic-reservoir paper has wired together. The composition is the
contribution, not either piece alone.

## Non-goals (for now)

- Beating BERT / E5 / bge-m3 on MTEB at scale.
- Production performance.
- Anything involving a full variational training loop with autodiff
  through the SF program until we have a baseline geometry to beat.
- Continuous DisCoCat as a hard requirement — we'll use coarse
  composition (beam-splitter mesh) until evidence says otherwise.
