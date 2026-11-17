"""Belief-propagation decoding of the 5G NR LDPC code against Sionna (Apache-2.0).

Sionna (<https://github.com/NVlabs/sionna>) decodes the rate-matched LDPC code of
TS 38.212 with ``LDPC5GDecoder``: check-node rule ``"boxplus-phi"`` (sum-product) or
``"minsum"``, flooding or layered schedule, a fixed number of iterations. comnum's
:class:`comnum.NrLdpc` encodes the same codewords (compared bit for bit in
``test_nr_pdsch_oracle.py``); here both decode the same noisy LLRs of BPSK over AWGN,
in double precision, 20 iterations, no clipping of the LLRs (``llr_max=None``), and
the decoded information bits are compared. Sionna's decisions are recorded fixtures
(``tests/oracle/sionna_ldpc.json`` and ``.npz``, see ``tests/oracle/__init__.py``)::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_sionna_ldpc_decoding.py

The information bits and the noise are drawn by comnum's generator (xoshiro256**, ADR
0002) from the seeds of ``CONFIGS``, so the test draws them again rather than reading
them; the LLRs of the first configuration are also recorded, for the card
``sionna_ldpc_decoding``. Sionna's LLR is :math:`\\ln P(1)/P(0)`: it is given the
opposite of comnum's.
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

# (k, E, Eb/N0 in dB, seed): rates 0.42, 0.67 and 0.5, near the waterfall
CONFIGS = [(500, 1200, 1.0, 5), (1000, 1500, 2.5, 6), (3000, 6000, 1.0, 7)]
BLOCKS = 40
RULES = [("sum_product", "boxplus-phi"), ("min_sum", "minsum")]
SCHEDULES = ["flooding", "layered"]


def draw(k, e, ebn0_db, seed):
    """The information bits and the LLRs (comnum's sign) of the blocks of a configuration."""
    code = comnum.NrLdpc(k, e)
    rng = comnum.Rng(seed)
    u = np.stack([np.asarray(comnum.bits(k, rng)) for _ in range(BLOCKS)])
    c = np.stack([np.asarray(code.encode(x)) for x in u]).astype(float)
    s2 = 1 / (2 * 10 ** (ebn0_db / 10) * k / e)
    noise = np.asarray(rng.gaussian(c.size)).reshape(c.shape)
    return u, 2 * ((1 - 2 * c) + np.sqrt(s2) * noise) / s2


def key(k, e, rule, schedule):
    return f"{k}_{e}_{rule}_{schedule}"


def _compute():
    import torch
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder

    out = {"decisions": {}}
    for k, e, ebn0, seed in CONFIGS:
        u, llr = draw(k, e, ebn0, seed)
        enc = LDPC5GEncoder(k, e, precision="double")
        for rule, cn in RULES:
            for schedule in SCHEDULES:
                dec = LDPC5GDecoder(enc, cn_update=cn, cn_schedule=schedule, num_iter=20, llr_max=None,
                                    precision="double", hard_out=True)  # fmt: skip
                bits = np.asarray(dec(torch.tensor(-llr)), np.uint8)
                out["decisions"][key(k, e, rule, schedule)] = np.packbits(bits, axis=1)
                if k == 500:  # the decisions of the blocks of the card, unpacked
                    out[f"card_{rule}_{schedule}"] = bits[:8]
        if k == 500:
            out["llr_500"] = llr[:8]
            out["info_500"] = u[:8]
    return out


@pytest.fixture(scope="module")
def sionna():
    return oracle("sionna_ldpc", _compute, tool="Sionna")


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("rule", [r for r, _ in RULES])
@pytest.mark.parametrize("config", CONFIGS)
def test_the_decoded_bits_are_those_of_sionna(sionna, config, rule, schedule):
    k, e, _ebn0, _seed = config
    u, llr = draw(*config)
    code = comnum.NrLdpc(k, e, rule=rule, max_iterations=20, layered=schedule == "layered")
    ours = np.stack([np.asarray(code.decode(x)) for x in llr])
    theirs = np.unpackbits(
        np.asarray(sionna["decisions"][key(k, e, rule, schedule)], np.uint8), axis=1
    )[:, :k]
    failed = np.any(ours != u, axis=1) & np.any(theirs != u, axis=1)
    differ = np.any(ours != theirs, axis=1)
    if rule == "min_sum" and schedule == "layered":
        # in the blocks that both fail, layered min-sum is chaotic: a relative change of
        # 1e-12 of the LLRs changes comnum's own word there, so the rounding of sums in
        # another order makes the two end on different words; every block one of them
        # decodes, the other decodes the same
        assert np.all(failed[differ]), config
    else:
        np.testing.assert_array_equal(ours, theirs)


def test_the_recorded_llrs_are_those_comnum_draws(sionna):
    u, llr = draw(*CONFIGS[0])
    # the Gaussian draws go through ln, sqrt and cos of the platform's libm, which
    # differ by one ulp between macOS and glibc: the LLRs drawn again agree to 1e-15
    # (the bits exactly), and the decoders are insensitive to it but for layered
    # min-sum where it fails, which the test already allows
    np.testing.assert_allclose(sionna["llr_500"], llr[:8], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(sionna["info_500"], u[:8])
    packed = np.asarray(sionna["decisions"][key(500, 1200, "sum_product", "flooding")], np.uint8)
    np.testing.assert_array_equal(
        sionna["card_sum_product_flooding"], np.unpackbits(packed, axis=1)[:8, :500]
    )
