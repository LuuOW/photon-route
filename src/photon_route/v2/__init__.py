"""Path-A v2 encoder: per-word lookup table → Gaussian state via numpy.

The inference path imports only numpy + thewalrus (which v1 already pulls in
for fidelity). No torch, no autograd at deployment time. The PyTorch-based
training loop lives in v2/training.py and runs offline; once trained, weights
are serialized as a plain dict[word, np.ndarray(4)] of (r, phi_s, alpha_q,
alpha_p) and consumed back through this same numpy encoder.

This preserves the artifact's original framing: closed-form linear algebra
on small symplectic matrices, every distance a determinant, no NN at
inference. The only thing that changes from v1 is *where the four numbers
per word come from* — SHA-256 in v1, a learned table in v2.

Status: Step 0 (SHA-init equivalence verified). Training is the next
deliverable; see docs/design.md for the multi-step plan.
"""

from photon_route.v2.encoder import Encoder, EncoderConfig, vacuum_state
from photon_route.v2.init_sha import sha_params_v1_compat

__all__ = ["Encoder", "EncoderConfig", "vacuum_state", "sha_params_v1_compat"]
