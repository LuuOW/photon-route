---
title: photon-route
emoji: 💡
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Continuous-variable photonic retrieval (Strawberry Fields).
sleep_time: 3600
---

# photon-route

A continuous-variable (CV) photonic retrieval research artifact.

Each document and query is encoded as a small Strawberry Fields program — words
contribute squeezing and displacement parameters to alternating bosonic modes,
sentence length controls a beam-splitter mixing layer, and the resulting
Gaussian state is the document's representation in Fock space. Retrieval ranks
documents by closed-form Gaussian-state fidelity (Banchi-Braunstein-Pirandola)
between query and document states.

This is the CV sister to [qrouter](https://github.com/LuuOW/qrouter), which
takes the discrete-variable (DV / qubit-gate-model) path. Both share the same
fixture and ranking interface; the substrate is what changes.

## Status

Day-1 scaffold. The pipeline is end-to-end correct in shape; numerical results
are placeholders until:

- The encoding is replaced with a learned parameter map (today's params are
  SHA-256-derived from word identity)
- A real eval set (~200–500 query-doc pairs over arXiv quant-ph) is built
- A non-Gaussian element (PNR measurement on a subset of modes, conditioning
  the rest) is added to escape Gaussian classical-simulability
- The same program is run on Borealis via Xanadu Cloud for the
  simulated-vs-measured comparison

See `docs/design.md` for the research diary.

## Quickstart (local)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[photonic,dev]"
photon-route "what happens when light hits a barrier"
```

Or run the HTTP server:

```bash
uvicorn photon_route.http_server:app --port 7860
curl 'http://127.0.0.1:7860/rank?q=quantum+entanglement&top_k=3'
```

## Endpoints (HTTP)

- `GET /` — JSON banner with backend mode
- `GET /health` — `{"ok": true, "backend": "..."}` where backend is `gaussian`
  if Strawberry Fields imports cleanly, else `stub`
- `GET /rank?q=...&top_k=N` — ranked fixture documents
- `GET /version`, `GET /docs` — metadata + FastAPI Swagger

## License

MIT.
