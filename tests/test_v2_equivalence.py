"""v2(SHA-init) must produce numerically-identical (mu, sigma) to v1.

This is the Step 0 invariant: the v2 inference path is structurally the same
as v1, just with the four-numbers-per-word source pluggable. If this test
passes, any retrieval-quality difference between trained-v2 and v1 is
attributable to the parameter table, not to the gate algebra.
"""

from __future__ import annotations

import numpy as np
import pytest

QUERIES = [
    "quantum",
    "quantum entanglement",
    "superposition lets a quantum bit represent multiple states at once",
    "photons travel through optical fibers without losing coherence",
    "spectroscopy reveals atomic energy levels through emitted light",
    "measurement collapses a quantum state into a classical outcome",
    "qubits entangle across distant detectors",
    "the cat in the box is both alive and dead until you look",
]


@pytest.mark.parametrize("text", QUERIES)
def test_v2_sha_matches_v1_state(text):
    """v2 encoder seeded from SHA-256 must match v1's SF state byte-for-byte
    (within 1e-9 absolute tolerance — both are double-precision linear algebra
    and there's no extra step in either path that should accumulate drift)."""
    from photon_route.encode import encode_one as v1_encode
    from photon_route.v2 import Encoder, sha_params_v1_compat

    state_v1 = v1_encode(text)
    mu_v1 = np.asarray(state_v1.means(), dtype=np.float64)
    sigma_v1 = np.asarray(state_v1.cov(), dtype=np.float64)

    encoder_v2 = Encoder(params_fn=sha_params_v1_compat)
    mu_v2, sigma_v2 = encoder_v2.encode(text)

    assert mu_v2.shape == mu_v1.shape, f"shape: {mu_v2.shape} vs {mu_v1.shape}"
    assert sigma_v2.shape == sigma_v1.shape

    np.testing.assert_allclose(
        mu_v2, mu_v1, atol=1e-9, rtol=1e-9,
        err_msg=f"mu mismatch on '{text}':\n  v1={mu_v1}\n  v2={mu_v2}",
    )
    np.testing.assert_allclose(
        sigma_v2, sigma_v1, atol=1e-9, rtol=1e-9,
        err_msg=f"sigma mismatch on '{text}'",
    )


def test_v2_fidelity_matches_v1_score():
    """End-to-end: v2-encoded state vs v2-encoded doc, scored with thewalrus,
    must equal v1-encoded query vs v1-encoded doc, also scored with thewalrus.
    This protects against subtle issues in mu/sigma that wouldn't surface in
    the raw-state test (e.g., transposition errors that happen to cancel in
    one component but bias the fidelity)."""
    from thewalrus.quantum import fidelity as tw_fidelity

    from photon_route.encode import encode_one as v1_encode
    from photon_route.v2 import Encoder, sha_params_v1_compat

    query = "quantum entanglement"
    doc = "qubits entangle across distant detectors"
    encoder_v2 = Encoder(params_fn=sha_params_v1_compat)

    mu_q1, sg_q1 = (np.asarray(x, np.float64) for x in (v1_encode(query).means(), v1_encode(query).cov()))
    mu_d1, sg_d1 = (np.asarray(x, np.float64) for x in (v1_encode(doc).means(), v1_encode(doc).cov()))
    mu_q2, sg_q2 = encoder_v2.encode(query)
    mu_d2, sg_d2 = encoder_v2.encode(doc)

    f1 = float(tw_fidelity(mu_q1, sg_q1, mu_d1, sg_d1))
    f2 = float(tw_fidelity(mu_q2, sg_q2, mu_d2, sg_d2))
    assert abs(f1 - f2) < 1e-9, f"fidelity drift: v1={f1!r}, v2={f2!r}"


def test_dict_params_fn_round_trips():
    """A learned-style table containing only the test-query words should
    produce the same encoded state as calling the SHA-256 fn directly,
    proving the dict-backed params source is wired correctly."""
    from photon_route.v2 import Encoder, sha_params_v1_compat
    from photon_route.v2.encoder import dict_params_fn

    text = "quantum entanglement"
    table = {w: sha_params_v1_compat(w) for w in text.split()}

    enc_sha = Encoder(params_fn=sha_params_v1_compat)
    enc_dict = Encoder(params_fn=dict_params_fn(table))

    mu_a, sg_a = enc_sha.encode(text)
    mu_b, sg_b = enc_dict.encode(text)
    np.testing.assert_allclose(mu_a, mu_b, atol=1e-12)
    np.testing.assert_allclose(sg_a, sg_b, atol=1e-12)


def test_unk_word_uses_unk_vector():
    """Words missing from the table should produce the unk vector's params
    (default: zeros → no squeeze, no displacement, mode is just identity)."""
    from photon_route.v2 import Encoder
    from photon_route.v2.encoder import dict_params_fn

    enc_zero_unk = Encoder(params_fn=dict_params_fn({}))
    mu, sigma = enc_zero_unk.encode("a b c")
    # All-zeros params: squeeze=0 → I, displace=0 → no shift. Only effect
    # is the terminal BS for word_count=3. State should still be Gaussian
    # with mu = 0 and sigma = BS @ I @ BS.T = I (BS is orthogonal).
    np.testing.assert_allclose(mu, np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(sigma, np.eye(4), atol=1e-12)
