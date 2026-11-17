"""A coded link of comnum, block by block, against the same link in Sionna.

The kernels of comnum are compared with Sionna one by one elsewhere; this test checks
how the blocks of a chain fit together, where errors of convention hide: the order of
the bits on the points, the definition of :math:`E_b/N_0` with the code rate, the sign
and the scale of the LLRs, the systematic bits of the decoded word. The declarative
chain

    fec_encode (LDPC, PEG, n = 1200, k = 600) -> map -> awgn_ebn0 -> demap -> fec_decode

runs with :meth:`comnum.Chain.trace` for QPSK, 16-QAM and 64-QAM; Sionna (NVIDIA,
Apache-2.0) is given what each block received and computes what it should give:

- ``Mapper`` with comnum's points as a custom constellation maps the coded bits;
- ``ebnodb2no`` gives the noise variance of :math:`E_b/N_0` for :math:`\\log_2 M` bits per
  symbol and the rate 1/2, on the nominal energy of the constellation (1);
- ``Demapper`` (``"app"``) computes the LLRs of the received samples (its sign is the
  opposite of comnum's);
- ``LDPCBPDecoder`` with comnum's parity-check matrix (sum-product, flooding, 20
  iterations, no clipping) decodes them; the first 600 bits of its word are the
  information bits (comnum's PEG code is systematic in front).

Sionna's outputs are recorded fixtures (``tests/oracle/sionna_chain.json`` and
``.npz``, see ``tests/oracle/__init__.py``)::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_sionna_chain.py

The chain draws its bits and its noise with comnum's generator: on another platform
the noise differs by one ulp (the libm), so the LLRs are compared to 1e-9 and the
decisions exactly.
"""

import numpy as np
import pytest

import comnum
from comnum import blocks as B
from oracle import oracle

N, K = 1200, 600
CODE = {
    "type": "ldpc",
    "n": N,
    "k": K,
    "rule": "sum_product",
    "layered": False,
    "iterations": 20,
}
BLOCKS = 12
# (constellation, Eb/N0 in dB, seed): near the waterfall of each, some blocks fail
CASES = [("QPSK", 1.5, 11), ("16-QAM", 4.0, 12), ("64-QAM", 7.5, 13)]


def trace(constellation, ebn0_db, seed):
    link = comnum.Chain(
        enc=B.FecEncode(code=CODE, blocks=BLOCKS),
        map=B.Map(constellation=constellation),
        ch=B.AwgnEbn0(ebn0_db=ebn0_db),
        demap=B.Demap(),
        dec=B.FecDecode(code=CODE),
    )
    return link.trace({}, comnum.Rng(seed))


def parity_check():
    code = comnum.LdpcCode.peg(N, K)
    h = np.zeros((N - K, N), dtype=np.int32)
    for i, row in enumerate(code.rows):
        h[i, row] = 1
    return h


def _compute():
    import torch
    from sionna.phy.fec.ldpc import LDPCBPDecoder
    from sionna.phy.mapping import Constellation, Demapper, Mapper
    from sionna.phy.utils import ebnodb2no

    h = parity_check()
    out = {}
    for name, ebn0, seed in CASES:
        tr = trace(name, ebn0, seed)
        f = tr["dec"]
        pts = np.asarray(f.constellation.points, complex)
        bits = int(np.log2(len(pts)))
        sc = Constellation("custom", bits, points=torch.tensor(pts), precision="double")
        x = np.asarray(
            Mapper(constellation=sc, precision="double")(
                torch.tensor(np.asarray(f.tx_bits), dtype=torch.float64)
            )
        )
        no = float(ebnodb2no(torch.tensor(ebn0, dtype=torch.float64), bits, 0.5))
        y = np.asarray(tr["ch"].samples)
        llr = -np.asarray(
            Demapper("app", constellation=sc, precision="double")(
                torch.tensor(y), torch.tensor(no, dtype=torch.float64)
            )
        )
        dec = LDPCBPDecoder(
            h, num_iter=20, cn_update="boxplus-phi", hard_out=True, llr_max=None, precision="double"
        )
        word = np.asarray(dec(torch.tensor(-np.asarray(f.llr).reshape(BLOCKS, N))), np.uint8)
        out[name] = {
            "symbols": x.ravel(),
            "noise_var": no,
            "llr": llr.ravel()[: 2 * N],  # the first two blocks
            "decoded": np.packbits(word[:, :K], axis=1),
            "valid": bool(np.all((h @ np.asarray(f.tx_bits).reshape(BLOCKS, N).T) % 2 == 0)),
        }
    return out


@pytest.fixture(scope="module")
def sionna():
    return oracle("sionna_chain", _compute, tool="Sionna")


@pytest.mark.parametrize(("name", "ebn0", "seed"), CASES)
def test_the_chain_is_the_link_of_sionna(sionna, name, ebn0, seed):
    ref = sionna[name]
    tr = trace(name, ebn0, seed)
    f = tr["dec"]
    # the coded bits are words of the code, systematic in front
    assert ref["valid"]
    words = np.asarray(f.tx_bits).reshape(BLOCKS, N)
    np.testing.assert_array_equal(words[:, :K].ravel(), np.asarray(f.payload))
    # the bits on the points
    np.testing.assert_allclose(np.asarray(tr["map"].samples), ref["symbols"], atol=1e-15)
    # E_b/N_0 with the rate, on the nominal energy (Sionna computes in float32)
    assert f.noise_var == pytest.approx(ref["noise_var"], rel=1e-6)
    # the LLRs of Sionna's demapper, given Sionna's noise variance
    np.testing.assert_allclose(np.asarray(f.llr)[: 2 * N], ref["llr"], rtol=1e-5, atol=1e-9)
    # the decoded information bits
    theirs = np.unpackbits(np.asarray(ref["decoded"], np.uint8), axis=1)[:, :K]
    np.testing.assert_array_equal(np.asarray(f.decoded).reshape(BLOCKS, K), theirs)
