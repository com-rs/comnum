"""The Gold scrambler and the interleavers of comnum against py3gpp and Sionna.

py3gpp (AGPL-3.0, <https://github.com/catkira/py3gpp>) reimplements the 5G functions of
MATLAB, among them ``nrPRBS``, the pseudo-random sequence of TS 38.211, Sec. 5.2.1.
Sionna (NVIDIA, Apache-2.0, <https://github.com/NVlabs/sionna>) implements the
scrambler of the transport blocks of 5G (``TB5GScrambler``, TS 38.211, Secs. 6.3.1.1
and 7.3.1.1), the interleaver of the LTE turbo code (``Turbo3GPPInterleaver``, the
quadratic permutation polynomials of TS 36.212, Table 5.1.3-3) and a row-column
interleaver (``RowColumnInterleaver``). Their outputs are recorded fixtures
(``tests/oracle/scrambling_interleaving.json`` and ``.npz``, see
``tests/oracle/__init__.py``)::

    COMNUM_RECORD_ORACLE=1 python -m pytest tests/oracle/test_scrambling_interleaving_oracle.py

Compared:

- the first 20 000 bits of the Gold sequence of eight values of :math:`c_{\\mathrm{init}}`
  (0, 1, the largest of 31 bits and five others) with :meth:`comnum.Scrambler.sequence`;
- the sequences that ``TB5GScrambler`` adds to the bits of a PDSCH codeword (indices 0
  and 1) and of a PUSCH, for four pairs :math:`(n_{\\mathrm{RNTI}}, n_{\\mathrm{ID}})`,
  with the sequence of :math:`c_{\\mathrm{init}} = n_{\\mathrm{RNTI}} 2^{15} + q 2^{14} +
  n_{\\mathrm{ID}}`;
- the 188 coefficient pairs :math:`(f_1, f_2)` of Sionna's table, through the
  interleavers they define (``Interleaver.from_name("lte", k)`` against
  ``Interleaver.qpp(k, f1, f2)``), and Sionna's permutation itself for twelve sizes from
  40 to 6144;
- the row-column interleaver of depth :math:`d` on :math:`n` positions, comnum's
  ``Interleaver.block(n / d, d)``, for five shapes.

The card ``sionna_interleaving`` and ``py3gpp_nr_prbs`` hold the comparisons.
"""

import numpy as np
import pytest

import comnum
from oracle import oracle

C_INIT = [0, 1, 0x1234, 20061, 65535, 1 << 30, (1 << 31) - 1, 1_234_567_890]
LENGTH = 20_000
PAIRS = [(1, 1), (65535, 1023), (4660, 0), (17, 500)]
SIZES = [40, 48, 104, 512, 1008, 1024, 2048, 3072, 4032, 5056, 6080, 6144]
SHAPES = [(12, 3), (60, 4), (100, 10), (1000, 8), (6144, 64)]


def _py3gpp():
    from py3gpp.nrPRBS import nrPRBS

    return {str(c): np.asarray(nrPRBS(c, LENGTH), np.uint8) for c in C_INIT}


def _sionna():
    from sionna.phy.fec.interleaving import RowColumnInterleaver, Turbo3GPPInterleaver
    from sionna.phy.fec.scrambling import TB5GScrambler

    n = 4096
    scr = {}
    for rnti, nid in PAIRS:
        for kind, q in (("PDSCH", 0), ("PDSCH", 1), ("PUSCH", 0)):
            s = TB5GScrambler(n_rnti=rnti, n_id=nid, channel_type=kind, codeword_index=q)
            seq = s(np.zeros((1, n), np.float32))  # zeros: the output is the sequence
            scr[f"{kind}_{q}_{rnti}_{nid}"] = np.asarray(seq, np.uint8)[0]
    turbo = Turbo3GPPInterleaver()
    coeffs = np.array([[k, f1, f2] for k, (f1, f2) in sorted(turbo.coeffs_dict.items())], np.int64)
    perms = {}
    for k in SIZES:
        x = np.arange(k, dtype=np.float32)[None, :]
        perms[str(k)] = np.asarray(turbo(x), np.int64)[0]
    rc = {}
    for n_pos, d in SHAPES:
        x = np.arange(n_pos, dtype=np.float32)[None, :]
        rc[f"{n_pos}_{d}"] = np.asarray(RowColumnInterleaver(row_depth=d)(x), np.int64)[0]
    return {"scrambler": scr, "lte_coeffs": coeffs, "lte_perm": perms, "row_column": rc}


@pytest.fixture(scope="module")
def prbs():
    return oracle("nr_prbs", _py3gpp, tool="py3gpp")


@pytest.fixture(scope="module")
def sionna():
    return oracle("sionna_interleaving", _sionna, tool="Sionna")


@pytest.mark.parametrize("c_init", C_INIT)
def test_the_gold_sequence_is_that_of_py3gpp(prbs, c_init):
    np.testing.assert_array_equal(comnum.Scrambler(c_init).sequence(LENGTH), prbs[str(c_init)])


@pytest.mark.parametrize(("rnti", "nid"), PAIRS)
@pytest.mark.parametrize(("kind", "q"), [("PDSCH", 0), ("PDSCH", 1), ("PUSCH", 0)])
def test_the_scrambling_of_a_codeword_is_that_of_sionna(sionna, kind, q, rnti, nid):
    ref = sionna["scrambler"][f"{kind}_{q}_{rnti}_{nid}"]
    c_init = rnti * 2**15 + q * 2**14 + nid
    np.testing.assert_array_equal(comnum.Scrambler(c_init).sequence(ref.size), ref)


def test_the_lte_table_is_that_of_sionna(sionna):
    coeffs = np.asarray(sionna["lte_coeffs"])
    assert len(coeffs) == 188
    for k, f1, f2 in coeffs:
        ours = comnum.Interleaver.from_name("lte", int(k)).perm
        np.testing.assert_array_equal(ours, comnum.Interleaver.qpp(int(k), int(f1), int(f2)).perm)


@pytest.mark.parametrize("k", SIZES)
def test_the_lte_permutation_is_that_of_sionna(sionna, k):
    np.testing.assert_array_equal(
        comnum.Interleaver.from_name("lte", k).perm, sionna["lte_perm"][str(k)]
    )


@pytest.mark.parametrize(("n", "d"), SHAPES)
def test_the_row_column_interleaver_is_that_of_sionna(sionna, n, d):
    np.testing.assert_array_equal(
        comnum.Interleaver.block(n // d, d).perm, sionna["row_column"][f"{n}_{d}"]
    )
